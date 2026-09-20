"""Register annotation -- the mechanism the whole project turns on.

Every pretraining document is scored on five measured dimensions and prefixed
with control tokens. The model then learns ``p(text | register)`` instead of a
marginal ``p(text)`` averaged over wildly different registers -- which is what
produces "Dear Sir, no cap".

Because the scores are *measured* rather than labelled, the same code that
annotates 70 GB of corpus also scores model outputs at evaluation time. That is
what makes "cringe" a number: Register Calibration Error is just
``mean(|requested - measured|)`` (SPEC §15.3).

No human labelling. No LLM labelling. Deterministic and cheap: ~30k docs/sec/core.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import regex  # unicode property support that `re` lacks

from .lexicon import AMBIGUOUS_SENSES, AMBIGUOUS_UNANCHORED, Lexicon

# ------------------------------------------------------------------ #
# Patterns
# ------------------------------------------------------------------ #

_EMOJI = regex.compile(r"\p{Extended_Pictographic}")
_WORD = regex.compile(r"[\p{L}\p{N}']+")
_TOKEN = regex.compile(r"[\p{L}\p{N}']+|[^\s\p{L}\p{N}]")
_REPEAT = regex.compile(r"(.)\1{2,}")
_SENT_END = regex.compile(r"[.!?]")
_HOME_ROW = set("asdfghjkl")
_VOWELS = set("aeiou")

# Common Twitch/Discord emotes behave as emoji, not words.
EMOTES = {
    "kekw", "omegalul", "lul", "lulw", "pogchamp", "pog", "poggers", "sadge",
    "monkas", "pepega", "pepehands", "widepeepohappy", "catjam", "ez", "gg",
    "copium", "hopium", "malding", "kappa", "pepelaugh", "5head", "peepohappy",
    "aware", "clueless", "icant", "modcheck", "nodders", "based", "ratio",
    # Added after calibration: genuinely high-register chat was scoring 0
    # because these were missing.
    "madge", "feelsgoodman", "feelsbadman", "sodaomega", "omegalul", "lulw",
    "kekl", "pepeds", "susge", "peepod", "widepeepo", "hypers", "pauseçhamp",
    "pausechamp", "monkaw", "monkahmm", "sadcat", "plink", "ratjam", "vibe",
    "gigachad", "chad", "cringe", "sheesh", "yep", "nopers", "okayeg",
    "docl", "lidl", "actually", "scam", "weirdchamp", "pogu", "poggies",
    "5head", "3head", "kok", "chatting", "bedge", "sleeper", "tuck",
}

# High-frequency internet abbreviations. Not exhaustive -- the unknown-short-token
# heuristic catches the tail.
ABBREVS = {
    "ngl", "tbh", "istg", "ong", "fr", "iykyk", "nvm", "wdym", "hbu", "smh",
    "lmao", "lmfao", "lol", "rofl", "imo", "imho", "idk", "idc", "ikr", "tysm",
    "ty", "np", "nt", "gg", "brb", "afk", "irl", "tbf", "ftw", "rn", "atm",
    "btw", "fyi", "asap", "omg", "wtf", "wth", "af", "ffs", "istfg", "nvmd",
    "pmo", "sybau", "icl", "lowkey", "highkey", "deadass", "fs", "ts", "gyat",
    "wsp", "wyd", "hyd", "ily", "ilysm", "gn", "gm", "sm", "rlly", "prolly",
    "bc", "cuz", "tho", "thru", "u", "ur", "yr", "pls", "plz", "thx", "k",
    "ppl", "ig", "dm", "rt", "tldr", "tl;dr", "xd", "bf", "gf", "bff", "dw",
    "ez", "ofc", "nsfw", "til", "eli5", "iirc", "afaik", "op", "mf",
    # Netspeak shorthand that the lexicon was scoring as slang: "Damia, Sage
    # of Stone (G) (SF) (txt) ^^^FAQ" -- a bot signature -- measured slang_2.
    "txt", "faq", "pm", "dl", "ul", "src", "img", "msg", "pls", "rq",
}

# Function words that appear at high rates in edited prose and low rates in chat.
FORMAL_MARKERS = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "additionally", "accordingly", "whereas", "thus",
    "regarding", "concerning", "approximately", "significant", "particular",
}

# Bucket boundaries. Calibrated so each bucket holds a meaningful slice of the
# corpus rather than everything collapsing into 0. Re-check in Phase 1 against
# the human-rated sample (SPEC §16 Phase 1 gate (a)).
THRESHOLDS: dict[str, Sequence[float]] = {
    "slang": (0.5, 2.0, 5.0, 10.0),      # weighted hits / 100 tokens -> 0..4
    "emoji": (0.5, 2.0, 6.0),            # emoji / 100 tokens        -> 0..3
    "abbrev": (2.0, 6.0, 14.0),          # abbrevs / 100 tokens      -> 0..3
    "elong": (1.0, 4.0),                 # elongation events / 100   -> 0..2
    "formal": (0.30, 0.55, 0.78),        # formality score 0..1      -> 0..3
}

DIM_ORDER = ("slang", "emoji", "abbrev", "elong", "formal")

# Rates are per 100 tokens, but chat messages are ~10 tokens long, so an
# unsmoothed rate gives a single hit a density of 10 and lands it in the top
# bucket. That is what made the corpus bimodal -- long documents at slang_0,
# any short message containing one lexicon term at slang_4 -- with nothing in
# between for the model to learn a gradient from.
#
# Dividing by at least MIN_RATE_TOKENS treats a short document as weak evidence
# rather than intense evidence: one hit in a ten-word message is "noticeable",
# and reaching "saturated" still requires several. Documents longer than the
# floor are unaffected.
MIN_RATE_TOKENS = 30


def _bucket(value: float, edges: Sequence[float]) -> int:
    b = 0
    for edge in edges:
        if value >= edge:
            b += 1
        else:
            break
    return b


@dataclass
class Register:
    slang: int = 0
    emoji: int = 0
    abbrev: int = 0
    elong: int = 0
    formal: int = 0

    def to_tokens(self) -> str:
        """Serialise as the control-token prefix used in training."""
        inner = "".join(f"<|{d}_{getattr(self, d)}|>" for d in DIM_ORDER)
        return f"<|reg|>{inner}<|/reg|>"

    def as_vector(self) -> list[int]:
        return [getattr(self, d) for d in DIM_ORDER]

    def distance(self, other: "Register") -> float:
        """Mean absolute difference -- the per-sample term of RCE (SPEC §15.3)."""
        a, b = self.as_vector(), other.as_vector()
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_tokens(cls, text: str) -> "Register | None":
        m = re.search(r"<\|reg\|>(.*?)<\|/reg\|>", text, re.S)
        if not m:
            return None
        vals = {}
        for d in DIM_ORDER:
            dm = re.search(rf"<\|{d}_(\d)\|>", m.group(1))
            if dm:
                vals[d] = int(dm.group(1))
        return cls(**vals)


@dataclass
class RegisterFeatures:
    """Raw continuous measurements, before bucketing. Kept for calibration."""

    n_tokens: int
    slang_density: float
    emoji_rate: float
    abbrev_rate: float
    elong_rate: float
    formality: float
    slang_terms: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class RegisterAnnotator:
    def __init__(self, lexicon: Lexicon | None = None,
                 thresholds: dict[str, Sequence[float]] | None = None) -> None:
        self.lexicon = lexicon
        self.thresholds = thresholds or THRESHOLDS

    # -------------------------------------------------------------- #

    @staticmethod
    def _is_keysmash(token: str) -> bool:
        if len(token) < 6 or not token.isalpha():
            return False
        low = token.lower()
        vowel_ratio = sum(c in _VOWELS for c in low) / len(low)
        home_ratio = sum(c in _HOME_ROW for c in low) / len(low)
        return vowel_ratio < 0.2 or home_ratio > 0.8

    def _formality(self, text: str, words: list[str]) -> float:
        """0 (pure chat) .. 1 (edited prose). Five equally weighted signals."""
        if not words:
            return 0.5
        n = len(words)

        # 1. terminal punctuation present at all
        has_terminal = 1.0 if _SENT_END.search(text) else 0.0

        # 2. capitalisation of sentence starts
        caps = sum(1 for w in words if w[:1].isupper())
        cap_ratio = min(caps / max(n / 12, 1), 1.0)

        # 3. mean word length (proxy for register formality)
        mean_len = sum(len(w) for w in words) / n
        len_score = min(max((mean_len - 3.0) / 3.0, 0.0), 1.0)

        # 4. formal function words
        formal_hits = sum(1 for w in words if w.lower() in FORMAL_MARKERS)
        formal_score = min(formal_hits / max(n / 60, 1), 1.0)

        # 5. absence of chat markers
        chat_hits = sum(1 for w in words if w.lower() in ABBREVS or w.lower() in EMOTES)
        chat_score = 1.0 - min(chat_hits / max(n / 15, 1), 1.0)

        return (has_terminal + cap_ratio + len_score + formal_score + chat_score) / 5.0

    # -------------------------------------------------------------- #

    def features(self, text: str) -> RegisterFeatures:
        words = _WORD.findall(text)
        tokens = _TOKEN.findall(text)
        n = max(len(tokens), MIN_RATE_TOKENS)
        per100 = 100.0 / n

        emoji_n = len(_EMOJI.findall(text))
        emote_n = sum(1 for w in words if w.lower() in EMOTES)

        abbrev_n = sum(1 for w in words if w.lower() in ABBREVS)

        elong_n = len(_REPEAT.findall(text))
        elong_n += sum(1 for w in words if self._is_keysmash(w))

        if self.lexicon is not None:
            # Emotes and abbreviations have their own dimensions. Urban
            # Dictionary also has entries for them, so without this they are
            # counted twice -- which is why "[Trivia] ... FeelsBadMan" scored
            # slang_3 despite containing no slang at all.
            hits = [(t, w) for t, w in self.lexicon.hits(words)
                    if t not in EMOTES and t not in ABBREVS]
            # A word whose literal sense dominates is weak evidence on its
            # own and full evidence alongside something unambiguous.
            if not any(t not in AMBIGUOUS_SENSES for t, _ in hits):
                hits = [(t, w * AMBIGUOUS_UNANCHORED) for t, w in hits]
            slang_density = 100.0 * sum(w for _, w in hits) / n
            slang_terms = [t for t, _ in hits][:32]
        else:
            slang_density, slang_terms = 0.0, []

        return RegisterFeatures(
            n_tokens=len(tokens),
            slang_density=slang_density,
            emoji_rate=(emoji_n + emote_n) * per100,
            abbrev_rate=abbrev_n * per100,
            elong_rate=elong_n * per100,
            formality=self._formality(text, words),
            slang_terms=slang_terms,
        )

    def annotate(self, text: str) -> Register:
        f = self.features(text)
        t = self.thresholds
        return Register(
            slang=_bucket(f.slang_density, t["slang"]),
            emoji=_bucket(f.emoji_rate, t["emoji"]),
            abbrev=_bucket(f.abbrev_rate, t["abbrev"]),
            elong=_bucket(f.elong_rate, t["elong"]),
            formal=_bucket(f.formality, t["formal"]),
        )

    def prefix(self, text: str) -> str:
        """The training-ready document: control tokens + text."""
        return self.annotate(text).to_tokens() + text

    # -------------------------------------------------------------- #

    def calibration_error(
        self, requested: Sequence[Register], produced_texts: Sequence[str]
    ) -> dict[str, float]:
        """Register Calibration Error -- the headline anti-cringe metric.

        Goes *up* both when the model under-uses and over-uses slang, so it
        cannot be gamed by suppressing slang. Target < 0.4 (SPEC §15.3).
        """
        if len(requested) != len(produced_texts):
            raise ValueError("requested and produced_texts must align")
        per_dim = {d: 0.0 for d in DIM_ORDER}
        total = 0.0
        for req, text in zip(requested, produced_texts):
            got = self.annotate(text)
            for d in DIM_ORDER:
                per_dim[d] += abs(getattr(req, d) - getattr(got, d))
            total += req.distance(got)
        n = max(len(requested), 1)
        out = {f"rce_{d}": per_dim[d] / n for d in DIM_ORDER}
        out["rce"] = total / n
        return out


def load_annotator(lexicon_path: str | Path | None) -> RegisterAnnotator:
    lex = Lexicon.load(lexicon_path) if lexicon_path and Path(lexicon_path).exists() else None
    return RegisterAnnotator(lex)
