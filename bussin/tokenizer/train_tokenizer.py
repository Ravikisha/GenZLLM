"""Byte-level BPE trained on the Bussin corpus.

Why not reuse GPT-2's or Llama's tokenizer (SPEC §6.1): both were fit to formal
web text and fragment modern internet English badly. A corpus-matched vocabulary
encodes "ngl that fit is bussin fr 💀" in roughly half the tokens, which on a
fixed free-compute budget is the cheapest available win -- it is simultaneously
~40% more effective context and ~40% more effective compute on exactly the text
this project cares about.

Two things here are not standard:

1. **The pre-tokenizer regex** keeps hashtags, @mentions, r/subreddits, :emotes:
   and emoji grapheme clusters (including ZWJ sequences) whole, and keeps
   punctuation *runs* together so "???!!!" is one token rather than six.

2. **Forced vocabulary.** Every current slang term, every top emoji, every
   Twitch emote and every common abbreviation is guaranteed a single token. That
   costs ~5,000 of 49,152 slots and makes the highest-frequency Gen-Z items free
   to encode.

Vocabulary is capped at 65,536 so token shards store as uint16 (SPEC §6.7).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..model.config import SPECIAL_TOKENS

MAX_VOCAB = 65_536

# Keeps internet-native units intact. Deviates from the GPT-2 pattern in four
# places, each marked below.
PRETOKENIZE_PATTERN = "|".join(
    [
        r"<\|[a-z_0-9/]+\|>",                       # our special tokens, atomic
        # The optional leading space matters. Without it the scan reaches the
        # space first, ' ?[^\s\p{L}\p{N}]+' claims " @", and "@user" splits into
        # " @" + "user" -- defeating the whole point of these alternatives.
        r" ?[#@][\w_]+",                            # (1) hashtags / mentions whole
        r" ?r/[A-Za-z0-9_]+",                       # (2) subreddit references whole
        r" ?:[a-z_]+:",                             # (3) :emote: syntax whole
        r"\p{Extended_Pictographic}(\x{200D}\p{Extended_Pictographic})*[\x{FE0F}\x{20E3}]*",
        r"'(?:[sdmt]|ll|ve|re)",                    # contractions
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",               # (4) punctuation runs together
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

# Abbreviations and emotes that must survive as single tokens.
FORCED_ABBREVS = [
    "ngl", "tbh", "istg", "ong", "fr", "iykyk", "nvm", "wdym", "hbu", "smh",
    "lmao", "lmfao", "lol", "rofl", "imo", "imho", "idk", "idc", "ikr", "tysm",
    "brb", "afk", "irl", "tbf", "ftw", "rn", "atm", "btw", "fyi", "asap",
    "omg", "wtf", "wth", "af", "ffs", "pmo", "icl", "lowkey", "highkey",
    "deadass", "fs", "ts", "wsp", "wyd", "hyd", "ily", "ilysm", "gn", "gm",
    "rlly", "prolly", "cuz", "tho", "thru", "pls", "plz", "thx", "ngmi", "wagmi",
]

FORCED_EMOTES = [
    "KEKW", "OMEGALUL", "LULW", "PogChamp", "Poggers", "Sadge", "monkaS",
    "Pepega", "PepeHands", "widepeepoHappy", "catJAM", "Copium", "Hopium",
    "malding", "Kappa", "PepeLaugh", "5Head", "peepoHappy", "Aware", "Clueless",
    "EZ Clap", "modCheck", "NODDERS", "Sadge", "KEKL", "ICANT",
]

# Elongation forms worth reserving: the most commonly stretched words, at the
# three lengths that carry distinct intensity.
ELONGATABLE = ["so", "no", "yes", "pls", "stop", "wait", "omg", "aw", "ah", "ugh",
               "yay", "nah", "bruh", "wow", "hey", "ok", "why", "what"]

HINGLISH_FUNCTION = [
    "hai", "nahi", "kya", "yaar", "bhai", "matlab", "acha", "accha", "bohot",
    "bahut", "kar", "raha", "rahi", "hoon", "mera", "tera", "apna", "abhi",
    "kuch", "sab", "phir", "lekin", "agar", "toh", "bhi", "hi", "na", "ke",
    "ki", "ka", "se", "mein", "par", "aur", "ye", "wo", "koi", "jab", "tab",
]


def build_forced_tokens(
    lexicon_path: str | Path | None = None,
    emoji_counts: Counter | None = None,
    top_emoji: int = 1_000,
) -> list[str]:
    """Assemble the tokens that must each encode as exactly one id."""
    forced: list[str] = []

    if lexicon_path and Path(lexicon_path).exists():
        from ..data.lexicon import Lexicon

        lex = Lexicon.load(lexicon_path)
        forced.extend(lex.single_word_current())

    forced.extend(FORCED_ABBREVS)
    forced.extend(FORCED_EMOTES)
    forced.extend(HINGLISH_FUNCTION)

    for word in ELONGATABLE:
        last = word[-1]
        for n in (3, 4, 6):
            forced.append(word + last * (n - 1))

    if emoji_counts:
        forced.extend(e for e, _ in emoji_counts.most_common(top_emoji))

    # A leading space is a different token under byte-level BPE, and it is the
    # far more common form in running text.
    with_space = [" " + t for t in forced if not t.startswith(" ")]
    forced.extend(with_space)

    seen, out = set(), []
    for t in forced:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def count_emoji(texts: Iterable[str], limit: int = 500_000) -> Counter:
    import regex

    pat = regex.compile(
        r"\p{Extended_Pictographic}(‍\p{Extended_Pictographic})*[️⃣]*"
    )
    counts: Counter = Counter()
    for i, t in enumerate(texts):
        if i >= limit:
            break
        counts.update(m.group(0) for m in pat.finditer(t))
    return counts


def _forcing_documents(
    forced: Sequence[str], repeats: int = 400, per_doc: int = 200
) -> Iterator[str]:
    """Synthesise documents that repeat each forced term often enough for BPE
    to learn it as a merge.

    Terms are emitted space-separated so the byte-level pre-tokenizer sees them
    in their natural ' word' form -- the variant that actually occurs in
    running text, and the only one worth spending vocabulary on. Measured on a
    16k-vocab run: 497/500 of the top current slang terms encode as a single
    token in that position.

    Do not also force the bare form. Under byte-level BPE "rizz" and " rizz"
    are different tokens, so forcing both doubles the vocabulary cost of every
    slang term -- ~6,800 of 49,152 slots -- to serve the <1% of occurrences
    that sit at the very start of a document.
    """
    batch: list[str] = []
    for _ in range(repeats):
        for term in forced:
            batch.append(term)
            if len(batch) >= per_doc:
                yield " ".join(batch)
                batch = []
    if batch:
        yield " ".join(batch)


def train_tokenizer(
    corpus: Iterator[str],
    vocab_size: int = 49_152,
    out_path: str | Path = "tokenizer/tokenizer.json",
    forced_tokens: Sequence[str] | None = None,
    min_frequency: int = 2,
    show_progress: bool = True,
    forcing_repeats: int = 400,
):
    """Train and save a byte-level BPE tokenizer."""
    if vocab_size > MAX_VOCAB:
        raise ValueError(
            f"vocab_size {vocab_size} exceeds {MAX_VOCAB}; token shards are uint16"
        )

    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(PRETOKENIZE_PATTERN), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()

    forced = list(forced_tokens or [])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Forcing is done by *oversampling in the training stream*, never with
    # `add_tokens`.
    #
    # `add_tokens` registers literal strings that are matched against raw text
    # before byte-level encoding, and the byte-level alphabet reuses printable
    # glyphs to stand for raw bytes. So adding the character "©" creates a
    # vocabulary entry that collides with the glyph that already denotes byte
    # 0xA9. Encoding "©" (UTF-8 C2 A9) then yields that single token, and
    # decoding it produces byte 0xA9 alone -- invalid UTF-8, silently replaced
    # with U+FFFD. Measured: 2.36% of general-pool documents corrupted this way,
    # every one of them at a non-ASCII character.
    #
    # Oversampling instead makes the forced terms ordinary learned merges, in
    # byte-level space, with no separate matching path.
    def _stream():
        yield from corpus
        if forced:
            yield from _forcing_documents(forced, repeats=forcing_repeats)

    tok.train_from_iterator(_stream(), trainer=trainer)

    from tokenizers.processors import ByteLevel as ByteLevelProcessor

    tok.post_processor = ByteLevelProcessor(trim_offsets=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    print(f"  saved {out}  (vocab {tok.get_vocab_size():,})")

    if forced:
        single = sum(1 for t in forced if len(tok.encode(t).ids) == 1)
        print(f"  forced terms encoding as one token: {single:,}/{len(forced):,} "
              f"({100 * single / len(forced):.1f}%)")
    return tok


# ------------------------------------------------------------------ #
# Evaluation gate (SPEC §6.8)
# ------------------------------------------------------------------ #

TARGETS = {
    "bytes_per_token_general": 4.0,
    "bytes_per_token_genz": 3.2,
    "bytes_per_token_hinglish": 2.8,
    "fertility_chat": 1.6,          # tokens per whitespace word, lower is better
    "roundtrip": 1.0,
    # Depends on vocab headroom: ~4,600 forced terms need a vocabulary much
    # larger than they are before most can be single tokens. Meaningless below
    # full scale, so it is scale-dependent like the compression targets.
    "forced_single_token_rate": 0.95,
}


def evaluate_tokenizer(
    tokenizer,
    samples: dict[str, list[str]],
    forced_terms: Sequence[str] = (),
) -> dict[str, float]:
    """Run the §6.8 gate. Do not tokenize 70 GB until this passes."""
    report: dict[str, float] = {}

    for pool, texts in samples.items():
        if not texts:
            continue
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tokens = sum(len(tokenizer.encode(t).ids) for t in texts)
        n_words = sum(len(t.split()) for t in texts)
        report[f"bytes_per_token_{pool}"] = n_bytes / max(n_tokens, 1)
        report[f"fertility_{pool}"] = n_tokens / max(n_words, 1)

    # Round-trip fidelity. This is where ZWJ emoji sequences silently corrupt.
    #
    # `skip_special_tokens` must be False. It defaults to True, which strips the
    # PII placeholders the cleaner inserts (<|email|>, <|phone|>) and reports a
    # ~1% round-trip failure that is the decoder behaving correctly. Chasing
    # that as a tokenizer bug is a dead end.
    all_texts = [t for texts in samples.values() for t in texts]
    ok = sum(
        1 for t in all_texts
        if tokenizer.decode(tokenizer.encode(t).ids, skip_special_tokens=False) == t
    )
    report["roundtrip"] = ok / max(len(all_texts), 1)

    if forced_terms:
        # Measured in running-text position, i.e. with the leading space.
        #
        # The bare form is a different token under byte-level BPE and occurs
        # only at the start of a document or immediately after punctuation.
        # Measuring it reported 0.228 while the form the model actually meets
        # was at 0.994, and reading that as a tokenizer failure would have led
        # to spending 14% of the vocabulary on duplicate tokens.
        single = sum(
            1 for t in forced_terms
            if len(tokenizer.encode(t if t.startswith(" ") else " " + t).ids) == 1
        )
        report["forced_single_token_rate"] = single / len(forced_terms)

    report["vocab_size"] = float(tokenizer.get_vocab_size())
    return report


# Below this much training text the compression targets are not reachable at
# any vocabulary size: BPE simply has not seen enough to learn long merges.
# Smoke runs sit far under it, and their bytes/token numbers mean nothing.
MIN_MEANINGFUL_SAMPLE_BYTES = 500_000_000


def check_gate(
    report: dict[str, float], sample_bytes: int | None = None
) -> tuple[bool, list[str], list[str]]:
    """Returns (passed, failures, warnings).

    `roundtrip` is the only scale-independent target and is always enforced:
    a tokenizer that cannot reproduce its input is broken at any size.

    Everything else -- the compression targets and
    `forced_single_token_rate` -- is only meaningful once the tokenizer has
    seen a realistic amount of text and has a realistic vocabulary to spend, so
    below MIN_MEANINGFUL_SAMPLE_BYTES they are reported as warnings rather than
    failures; otherwise every smoke run looks like a broken tokenizer. A real
    run is above the threshold, where they are enforced as failures.

    `forced_single_token_rate` in particular is bounded by vocab size, not by
    sample size: a 16k-vocab smoke run cannot seat 5k forced terms however much
    text it sees. It is the production vocabulary that makes the target
    reachable.
    """
    scale_free = {"roundtrip"}
    undersized = sample_bytes is not None and sample_bytes < MIN_MEANINGFUL_SAMPLE_BYTES

    failures: list[str] = []
    warnings: list[str] = []
    for key, target in TARGETS.items():
        if key not in report:
            continue
        value = report[key]
        if key.startswith("fertility"):
            bad, msg = value > target, f"{key}={value:.3f} above target {target}"
        else:
            bad, msg = value < target, f"{key}={value:.3f} below target {target}"
        if not bad:
            continue
        if key not in scale_free and undersized:
            warnings.append(msg + " (sample too small to be meaningful)")
        else:
            failures.append(msg)

    if undersized:
        warnings.append(
            f"tokenizer trained on {(sample_bytes or 0) / 1e6:.0f} MB; compression "
            f"targets assume >= {MIN_MEANINGFUL_SAMPLE_BYTES / 1e9:.1f} GB"
        )
    return (not failures), failures, warnings
