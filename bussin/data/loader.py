"""Sharded, memory-mapped, resumable data loading.

Tokens live as flat `uint16` streams (vocab <= 65,536, SPEC §6.7), 100M tokens
per shard. `np.memmap` means a session starts reading in milliseconds with no
RAM cost and no download -- which matters because on Kaggle the corpus is a
*mounted* dataset, not a fetched one.

The cursor is `(shard_index, token_offset, epoch)`. Eight bytes of state is all
it takes to resume mid-shard on different hardware, which is what makes the
relay possible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

TOKEN_DTYPE = np.uint16


@dataclass
class ShardInfo:
    path: str
    n_tokens: int
    stage: str = "S1"
    pool: str = "general"
    sha256: str | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class Cursor:
    shard: int = 0
    offset: int = 0
    epoch: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Cursor":
        if not d:
            return cls()
        return cls(shard=int(d.get("shard", 0)), offset=int(d.get("offset", 0)),
                   epoch=int(d.get("epoch", 0)))


class Manifest:
    """The shard index. Written by pipelines/06_shard.py."""

    def __init__(self, shards: list[ShardInfo], root: Path,
                 seq_len: int = 2048, vocab_size: int | None = None) -> None:
        self.shards = shards
        self.root = Path(root)
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        shards = [ShardInfo(**s) for s in d["shards"]]
        return cls(shards, root=p.parent, seq_len=d.get("seq_len", 2048),
                   vocab_size=d.get("vocab_size"))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"seq_len": self.seq_len, "vocab_size": self.vocab_size,
                 "n_shards": len(self.shards), "total_tokens": self.total_tokens,
                 "shards": [asdict(s) for s in self.shards]},
                indent=2,
            ),
            encoding="utf-8",
        )

    @property
    def total_tokens(self) -> int:
        return sum(s.n_tokens for s in self.shards)

    def filter_stage(self, stages: Sequence[str]) -> "Manifest":
        return Manifest([s for s in self.shards if s.stage in stages],
                        self.root, self.seq_len, self.vocab_size)

    def resolve(self, shard: ShardInfo) -> Path:
        p = Path(shard.path)
        return p if p.is_absolute() else self.root / p


class ShardReader:
    """Reads one shard as a memory map, keeping at most a few open."""

    def __init__(self, manifest: Manifest, max_open: int = 2) -> None:
        self.manifest = manifest
        self.max_open = max_open
        self._open: dict[int, np.memmap] = {}
        self._order: list[int] = []

    def get(self, idx: int) -> np.memmap:
        if idx in self._open:
            return self._open[idx]
        shard = self.manifest.shards[idx]
        path = self.manifest.resolve(shard)
        if not path.exists():
            raise FileNotFoundError(f"shard {idx} missing: {path}")
        mm = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
        self._open[idx] = mm
        self._order.append(idx)
        while len(self._order) > self.max_open:
            evict = self._order.pop(0)
            self._open.pop(evict, None)
        return mm

    def close(self) -> None:
        self._open.clear()
        self._order.clear()


class PackedDataset:
    """Yields fixed-length token windows, resumable from a cursor.

    Set `document_ids=True` to also emit per-token document ids, which the model
    turns into a block-diagonal mask so packed-but-unrelated documents cannot
    attend to each other (SPEC §8.5).
    """

    def __init__(
        self,
        manifest: Manifest,
        seq_len: int,
        cursor: Cursor | None = None,
        eos_token_id: int = 0,
        emit_document_ids: bool = True,
        shuffle_shards: bool = True,
        seed: int = 1337,
    ) -> None:
        self.manifest = manifest
        self.seq_len = seq_len
        self.cursor = cursor or Cursor()
        self.eos_token_id = eos_token_id
        self.emit_document_ids = emit_document_ids
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.reader = ShardReader(manifest)
        if not manifest.shards:
            raise ValueError("manifest contains no shards")

    def _shard_order(self, epoch: int) -> list[int]:
        order = list(range(len(self.manifest.shards)))
        if self.shuffle_shards:
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(order)
        return order

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        while True:
            order = self._shard_order(self.cursor.epoch)
            # Resume mid-epoch: skip shards already consumed this epoch.
            start_pos = self.cursor.shard if self.cursor.shard < len(order) else 0
            for pos in range(start_pos, len(order)):
                idx = order[pos]
                mm = self.reader.get(idx)
                offset = self.cursor.offset if pos == start_pos else 0
                n = len(mm)
                while offset + self.seq_len + 1 <= n:
                    window = np.asarray(mm[offset : offset + self.seq_len + 1], dtype=np.int64)
                    self.cursor.shard = pos
                    self.cursor.offset = offset + self.seq_len
                    out = {"input_ids": window[:-1], "labels": window[1:]}
                    if self.emit_document_ids:
                        out["document_ids"] = self._doc_ids(window[:-1])
                    yield out
                    offset += self.seq_len
                self.cursor.offset = 0
            self.cursor.epoch += 1
            self.cursor.shard = 0
            self.cursor.offset = 0

    def _doc_ids(self, ids: np.ndarray) -> np.ndarray:
        """Document index per token: increments after each EOS."""
        is_eos = ids == self.eos_token_id
        doc = np.cumsum(is_eos, dtype=np.int32)
        # The EOS itself belongs to the document it terminates.
        doc[is_eos] -= 1
        return doc

    def batches(self, micro_batch: int) -> Iterator[dict[str, np.ndarray]]:
        buf: list[dict[str, np.ndarray]] = []
        for item in self:
            buf.append(item)
            if len(buf) == micro_batch:
                yield {
                    k: np.stack([b[k] for b in buf]) for k in buf[0]
                }
                buf = []

    def state(self) -> dict:
        return self.cursor.to_dict()

    def tokens_remaining_in_epoch(self) -> int:
        consumed = sum(
            self.manifest.shards[i].n_tokens
            for i in range(min(self.cursor.shard, len(self.manifest.shards)))
        ) + self.cursor.offset
        return max(self.manifest.total_tokens - consumed, 0)


class TorchLoader:
    """Thin torch wrapper. Deliberately not a DataLoader: we need exact,
    single-threaded control of the cursor so a checkpoint is reproducible."""

    def __init__(self, dataset: PackedDataset, micro_batch: int, device: str = "cpu"):
        self.dataset = dataset
        self.micro_batch = micro_batch
        self.device = device
        self._it = dataset.batches(micro_batch)

    def __iter__(self):
        return self

    def __next__(self):
        import torch

        batch = next(self._it)
        return {
            k: torch.from_numpy(v).to(self.device, non_blocking=True)
            for k, v in batch.items()
        }


# ------------------------------------------------------------------ #
# Writing shards
# ------------------------------------------------------------------ #


class ShardWriter:
    """Accumulates token ids and flushes fixed-size uint16 shards."""

    def __init__(self, out_dir: str | Path, shard_tokens: int = 100_000_000,
                 prefix: str = "shard", stage: str = "S1", pool: str = "general") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.prefix = prefix
        self.stage = stage
        self.pool = pool
        self._buf: list[np.ndarray] = []
        self._n = 0
        self._idx = 0
        self.shards: list[ShardInfo] = []

    def add(self, token_ids: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(token_ids, dtype=TOKEN_DTYPE)
        if arr.size and int(arr.max()) > 65_535:
            raise ValueError("token id exceeds uint16 range; vocab must be <= 65536")
        self._buf.append(arr)
        self._n += arr.size
        while self._n >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, n_tokens: int) -> None:
        flat = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        head, tail = flat[:n_tokens], flat[n_tokens:]
        path = self.out_dir / f"{self.prefix}_{self._idx:05d}.bin"
        head.tofile(path)
        self.shards.append(
            ShardInfo(path=path.name, n_tokens=int(head.size),
                      stage=self.stage, pool=self.pool)
        )
        self._idx += 1
        self._buf = [tail] if tail.size else []
        self._n = int(tail.size)

    def close(self) -> list[ShardInfo]:
        if self._n > 0:
            self._flush(self._n)
        return self.shards
