"""Text cleaning that preserves the register instead of deleting it.

The prime directive (SPEC §5.0): standard LLM cleaning pipelines are designed to
remove exactly the things that make text Gen-Z. C4 drops any line without
terminal punctuation and any document with fewer than five sentences, which
applied to Twitch chat deletes essentially all of it.

So every filter here exists in two strengths -- `strict` for the general-English
pool, `preserving` for the internet and Gen-Z pools:

  no terminal punctuation   standard: drop      here: keep, it is the default
  repeated chars 'sooooo'   standard: normalise here: cap at 6, record the length
  all-lowercase             standard: penalise  here: keep, it is a register
  emoji                     standard: strip     here: keep, they are syntax
  short documents           standard: drop <50w here: keep >= 3 tokens
  non-dictionary words      standard: drop      here: keep, that is the slang
  profanity                 standard: drop      here: keep and tag
  keysmash 'asdfkjh'        standard: drop      here: keep, it is a lexical item

Slurs and PII are the exception: those are removed unconditionally (§5.11).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Iterable

import regex

# ------------------------------------------------------------------ #
# Patterns
# ------------------------------------------------------------------ #

URL = regex.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|io|gg|tv|co)/\S*")
MD_LINK = regex.compile(r"\[([^\]]+)\]\([^)]+\)")
MD_QUOTE = regex.compile(r"^\s*&?gt;.*$|^\s*>.*$", regex.M)
HTML_TAG = regex.compile(r"<[^>]{1,200}>")
MENTION = regex.compile(r"(?<![\w/])@[A-Za-z0-9_]{2,30}")
SUBREDDIT = regex.compile(r"\br/([A-Za-z0-9_]{2,25})\b")
USER_REF = regex.compile(r"\bu/[A-Za-z0-9_-]{2,25}\b")
EMAIL = regex.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
PHONE = regex.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
LONG_DIGITS = regex.compile(r"\b\d{9,}\b")
IPV4 = regex.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
REPEAT_CHAR = regex.compile(r"(.)\1{6,}")
REPEAT_EMOJI = regex.compile(r"(\p{Extended_Pictographic})\1{8,}")
WHITESPACE = regex.compile(r"[ \t]{2,}")
NEWLINES = regex.compile(r"\n{3,}")

# Word tokens that keep internal apostrophes, so "don't" does not become "don".
WORD = regex.compile(r"\p{L}+(?:'\p{L}+)*")

DELETED = {"[deleted]", "[removed]", "[deleted by user]", "", "."}

BOT_AUTHOR = regex.compile(r"(?i)(bot$|^auto|automoderator|moderator$|-bot$|_bot$)")
BOT_BODY = regex.compile(
    r"(?i)(i am a bot|this action was performed automatically|"
    r"beep boop|^!|contact the moderators|bot was created)"
)

# Twitch chat bots, which the Reddit-shaped patterns above miss entirely.
# Measured: 22.9% of the mined twitch_chat pool was bot output -- trivia
# rounds, sub notifications and point tallies, none of it human register.
TWITCH_BOT = regex.compile(
    r"(?i)(^\[(trivia|scramble|quiz|raffle|poll)\]|"
    r"answered the question (correctly|incorrectly)|"
    r"no one answered correctly|the (answer|word) was|"
    r"thank you for (gifting|subscribing|the follow)|"
    r"(has|have) gifted \d+|is now live|"
    r"\[similarity: \d+%\]|got \d+ points)"
)
PROMO = regex.compile(
    r"(?i)(discord\.gg/|onlyfans|check out my|sub to my|subscribe to my|"
    r"use code |promo code|click the link in)"
)

# Latin-script non-English function words. fastText mislabels short informal
# English, so a cheap high-precision filter runs first (SPEC §5.2).
NON_EN_MARKERS = {
    "de": {"ich", "und", "nicht", "das", "ist", "mit", "auch", "aber", "wenn", "auf"},
    "es": {"que", "para", "como", "pero", "esta", "este", "muy", "todo", "con", "por"},
    "fr": {"les", "des", "une", "est", "pas", "pour", "dans", "qui", "avec", "sur"},
    "pt": {"nao", "para", "com", "uma", "que", "mais", "seu", "por", "como"},
    "it": {"che", "non", "per", "sono", "questo", "come", "anche", "della"},
}

# Romanised Hindi function words -> route to the Hinglish pool rather than bin.
HINGLISH_MARKERS = {
    "hai", "nahi", "kya", "yaar", "bhai", "matlab", "acha", "accha", "bohot",
    "bahut", "kar", "raha", "rahi", "hoon", "mera", "tera", "apna", "abhi",
    "kuch", "phir", "lekin", "agar", "toh", "bhi", "koi", "jab", "tab", "mein",
    "karo", "kaise", "kyun", "thoda", "sab", "log", "baat", "ho", "hain",
}

# Markup and platform artefacts that leak in as apparent vocabulary. Found by
# running emerging-term discovery on raw text and reading the top of the list.
MARKUP_NOISE = {
    "https", "http", "www", "redd", "png", "jpg", "jpeg", "webp", "gif", "svg",
    "amp", "nbsp", "quot", "gt", "lt", "utm", "src", "href", "width", "height",
    "preview", "format", "auto", "blur", "crop", "thumbnail", "imgur", "gyazo",
    "pbs", "twimg", "cdn", "media", "redgifs", "giphy", "youtu", "youtube",
    "reddit", "wikipedia", "wiki", "php", "html", "json", "api", "url", "uri",
}


@dataclass
class CleanResult:
    text: str
    ok: bool
    reason: str = ""
    lang: str = "en"
    pool: str = "internet"
    elong_max_run: int = 0
    n_words: int = 0
    flags: dict[str, bool] = field(default_factory=dict)


@dataclass
class CleanConfig:
    mode: str = "preserving"          # preserving | strict
    min_words: int = 3
    max_words: int = 4096
    max_char_repeat: int = 6
    max_emoji_repeat: int = 8
    min_unique_ratio: float = 0.25
    max_non_alpha_ratio: float = 0.70
    max_token_len: int = 60
    drop_quotes: bool = True
    url_placeholder: str = "http"     # TimeLMs convention
    mention_placeholder: str = "@user"

    @classmethod
    def strict(cls) -> "CleanConfig":
        return cls(mode="strict", min_words=50, min_unique_ratio=0.35,
                   max_non_alpha_ratio=0.35)


# ------------------------------------------------------------------ #


# Scripts that are decisive on sight. Cheaper and more reliable than any
# classifier, and they catch the languages a Latin-only marker list misses.
NON_LATIN = regex.compile(
    r"[\p{Cyrillic}\p{Arabic}\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}"
    r"\p{Devanagari}\p{Thai}\p{Hebrew}\p{Bengali}\p{Tamil}\p{Telugu}]")

# Below this many words, statistical language ID is unreliable on chat: it calls
# "KEKW ez clap gg" Hungarian. Short text falls back to the marker heuristic,
# which defaults to English -- the safe direction, since dropping real Gen-Z is
# the expensive mistake.
LANGID_MIN_WORDS = 8

_LANGID = None


def _langid():
    global _LANGID
    if _LANGID is None:
        try:
            import py3langid

            _LANGID = py3langid
        except ImportError:
            _LANGID = False
    return _LANGID


def detect_language(words: list[str], text: str = "") -> str:
    """Route to 'en', 'hinglish', or a language code to be dropped.

    Order matters and is not arbitrary:

    1. **Hinglish markers first.** Romanised Hindi is Latin script and every
       statistical identifier mislabels it -- py3langid calls
       "bhai ye kya hai yaar" Shona. The marker list is the only thing that
       gets it right, so it runs before anything else.
    2. **Non-Latin script**, which is decisive without a classifier.
    3. **Statistical ID, but only on >= 8 words.** Short chat is where it
       fails, and short chat is exactly what this project must keep.

    This replaced a 5-language marker heuristic that let German, Portuguese,
    Serbian, Swedish, Lithuanian and Tagalog through. Those then scored as
    *high register* -- non-English text has no English terminal punctuation or
    formal markers, so `formality` reads low -- which put r/askserbia,
    r/programiranje and r/lietuva at the top of the community ranking.
    """
    if len(words) < 3:
        return "en"
    lowered = {w.lower() for w in words}

    hing = len(lowered & HINGLISH_MARKERS)
    if hing >= 2 or (hing >= 1 and len(words) <= 12):
        return "hinglish"

    body = text or " ".join(words)
    if body:
        non_latin = len(NON_LATIN.findall(body))
        if non_latin / max(len(body), 1) > 0.10:
            return "non_latin"

    lid = _langid()
    if lid and len(words) >= LANGID_MIN_WORDS:
        try:
            lang, _ = lid.classify(body)
            return "en" if lang == "en" else lang
        except Exception:
            pass

    # Fallback for short text and when py3langid is unavailable.
    best, best_hits = "en", 0
    for lang, markers in NON_EN_MARKERS.items():
        hits = len(lowered & markers)
        if hits > best_hits:
            best, best_hits = lang, hits
    if best_hits >= 3 and best_hits / max(len(lowered), 1) > 0.05:
        return best
    return "en"


# Unicode quotes and dashes. Without this, "don't" typed on a phone uses U+2019
# and the word regex splits it into "don" -- which is why an early discovery run
# reported `don`, `didn` and `doesn` as emerging vocabulary.
UNICODE_PUNCT = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "´": "'",
    "“": '"', "”": '"', "–": "-", "—": "-",
    " ": " ", "​": "", "﻿": "",
})


def strip_markup(text: str, cfg: CleanConfig) -> str:
    text = text.translate(UNICODE_PUNCT)
    text = html.unescape(html.unescape(text))      # doubly-escaped is common
    if cfg.drop_quotes:
        text = MD_QUOTE.sub("", text)
    text = MD_LINK.sub(r"\1", text)                # keep anchor text, drop target
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(cfg.url_placeholder, text)
    return text


def scrub_pii(text: str, cfg: CleanConfig) -> tuple[str, dict[str, bool]]:
    """Mandatory pass. Runs over the body, not just the schema -- Reddit and
    Urban Dictionary text carry names and contact details inline (SPEC §5.11)."""
    flags: dict[str, bool] = {}
    if EMAIL.search(text):
        text, flags["email"] = EMAIL.sub("<|email|>", text), True
    if IPV4.search(text):
        text, flags["ip"] = IPV4.sub("<|redacted|>", text), True
    if PHONE.search(text):
        text, flags["phone"] = PHONE.sub("<|phone|>", text), True
    if LONG_DIGITS.search(text):
        text, flags["long_digits"] = LONG_DIGITS.sub("<|number|>", text), True
    text = USER_REF.sub(cfg.mention_placeholder, text)
    text = MENTION.sub(cfg.mention_placeholder, text)
    return text, flags


def cap_repeats(text: str, cfg: CleanConfig) -> tuple[str, int]:
    """Cap elongation but record its original length -- 'sooo' and 'soooooo'
    are different intensities, and the pre-cap run feeds the `elong` register
    dimension, so nothing is lost by capping."""
    longest = 0
    for m in REPEAT_CHAR.finditer(text):
        longest = max(longest, len(m.group(0)))
    for m in REPEAT_EMOJI.finditer(text):
        longest = max(longest, len(m.group(0)))
    text = REPEAT_CHAR.sub(lambda m: m.group(1) * cfg.max_char_repeat, text)
    text = REPEAT_EMOJI.sub(lambda m: m.group(1) * cfg.max_emoji_repeat, text)
    return text, longest


def is_bot(author: str | None, body: str) -> bool:
    if author and BOT_AUTHOR.search(author):
        return True
    head = body[:300]
    return bool(BOT_BODY.search(head) or TWITCH_BOT.search(head))


def clean_text(
    raw: str,
    cfg: CleanConfig | None = None,
    author: str | None = None,
    pool: str = "internet",
) -> CleanResult:
    cfg = cfg or CleanConfig()

    if not raw or raw.strip() in DELETED:
        return CleanResult("", False, "deleted_or_empty")

    if is_bot(author, raw):
        return CleanResult("", False, "bot")
    if PROMO.search(raw):
        return CleanResult("", False, "promo")

    text = strip_markup(raw, cfg)
    text, pii_flags = scrub_pii(text, cfg)
    text, elong = cap_repeats(text, cfg)

    text = SUBREDDIT.sub(r"r/\1", text)            # subreddits are real lexical items
    text = WHITESPACE.sub(" ", text)
    text = NEWLINES.sub("\n\n", text).strip()

    if not text:
        return CleanResult("", False, "empty_after_clean")

    words = WORD.findall(text)
    n = len(words)
    if n < cfg.min_words:
        return CleanResult(text, False, f"too_short({n})")
    if n > cfg.max_words:
        return CleanResult(text, False, f"too_long({n})")

    lowered = [w.lower() for w in words]
    if len(set(lowered)) / n < cfg.min_unique_ratio:
        return CleanResult(text, False, "repetitive")

    non_alpha = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    if non_alpha / max(len(text), 1) > cfg.max_non_alpha_ratio:
        return CleanResult(text, False, "mostly_symbols")

    if any(len(w) > cfg.max_token_len for w in words):
        return CleanResult(text, False, "hash_or_base64")

    # URL placeholder dominating the text means it was a link post.
    if lowered.count(cfg.url_placeholder) / n > 0.25:
        return CleanResult(text, False, "link_spam")

    lang = detect_language(words, text)
    if lang not in ("en", "hinglish"):
        return CleanResult(text, False, f"lang({lang})", lang=lang)

    if cfg.mode == "strict":
        if not regex.search(r"[.!?]", text):
            return CleanResult(text, False, "no_terminal_punctuation")

    return CleanResult(
        text=text, ok=True, lang=lang,
        pool="hinglish" if lang == "hinglish" else pool,
        elong_max_run=elong, n_words=n, flags=pii_flags,
    )


def content_words(text: str) -> list[str]:
    """Lowercased word tokens with markup artefacts removed.

    Used for lexicon matching and emerging-term discovery. Excluding
    MARKUP_NOISE matters: running discovery without it returns `https`, `png`,
    `webp` and `redd` as the top 'emerging slang'.
    """
    out = []
    for w in WORD.findall(text):
        lw = w.lower()
        if lw in MARKUP_NOISE or len(lw) < 2:
            continue
        out.append(lw)
    return out


def clean_stream(
    rows: Iterable[dict],
    cfg: CleanConfig | None = None,
    text_keys: tuple[str, ...] = ("text", "body", "content", "Message", "message"),
    author_keys: tuple[str, ...] = ("author", "username", "user"),
    pool: str = "internet",
) -> Iterable[CleanResult]:
    cfg = cfg or CleanConfig()
    for row in rows:
        raw = next((row[k] for k in text_keys if isinstance(row.get(k), str) and row[k]), "")
        author = next((row[k] for k in author_keys if isinstance(row.get(k), str)), None)
        yield clean_text(raw, cfg, author=author, pool=pool)
