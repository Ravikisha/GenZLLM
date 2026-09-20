"""The dated slang lexicon -- the keystone artefact of the project.

Built from `georgiyozhegov/urbandictionary` (152,941 rows, verified), which is
the only slang source carrying a `time` and `score` per definition. Those two
columns are what let us distinguish *current* slang from *dead* slang
mechanically instead of by hand, which in turn powers:

  1. corpus mining -- score documents by recency-weighted slang density
  2. register annotation -- the `slang` dimension (SPEC §3.4)
  3. tokenizer forcing -- every current term must be a single token (SPEC §6.5)
  4. evaluation -- `status=dead` terms become the outdated-slang test (SPEC §14)

Licence note: Urban Dictionary data is CC-BY-SA-4.0. We use it as a *lexicon
for filtering* (which words exist, and when) rather than as training text,
which keeps the share-alike obligation off the model weights. See SPEC §1.12.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

# Half-life for recency weighting, in years. A term first attested 3 years ago
# weighs 0.5; 6 years ago, 0.25. Chosen so the 2024-2026 tail dominates.
RECENCY_HALF_LIFE_YEARS = 3.0

# Urban Dictionary is heavily polluted with joke entries about individuals.
# These drop roughly 40% of raw rows (SPEC §1.4).
MIN_SCORE = 1
MIN_TERM_LEN = 2
MAX_TERM_WORDS = 3

# Function words can never be slang, whatever Urban Dictionary says.
#
# Measured 2026-09-18: the unfiltered lexicon matched "that", "on", "the",
# "ever" and "2" as slang, so `slang_density` fired on every sentence and the
# register annotator scored at random -- Spearman rho -0.058 against a human
# rater, with 4/5 of a calibration sample mis-bucketed upward.
#
# Note the asymmetry this has to respect: "cap", "bet", "tea", "based", "ate",
# "valid", "woke", "drag" and "camp" are ordinary English words *and* real
# Gen-Z slang with drifted senses. So the filter is a closed stoplist of
# function words, not a frequency cutoff, and the curated set is exempt.
STOPWORDS = set("""
a an the this that these those there here and or but so if then than as
because while when where which who whom whose what why how all any both each
few more most other some such no nor not only own same too very can will just
don should now i me my myself we our ours ourselves you your yours yourself
he him his himself she her hers herself it its itself they them their theirs
themselves am is are was were be been being have has had having do does did
doing would could shall may might must ought of at by for with about against
between into through during before after above below to from up down in out
on off over under again further once ever never always also else ok okay yes
yeah get got go going went come came make made take took see saw know knew
think thought say said want need like really very much many lot new old good
bad big small time way day year thing things people man woman boy girl
oh ah eh uh um hm hmm huh aw aww ow ugh hey hi hello bye yay woo wow oops
""".split())

# Romanised-Hindi grammar words, which Urban Dictionary also has entries for.
# Measured 2026-09-18: "hai" matched 886 times across the Hinglish pool, "bas"
# 253, "hoon" 218 -- pushing 16.8% of Hinglish documents to slang_3/4 despite
# containing no English slang at all.
#
# The same rule as English applies: function words are grammar, not slang.
# Content-word Hinglish slang ("yaar", "bhai", "jugaad", "bakwas") is kept,
# because that genuinely is register.
# Filler and game-stat jargon that sits just under the frequency floor. These
# are domain vocabulary, not register markers: "I got 140 on tank, 100 on dps"
# is a build description, and scoring it as saturated slang is what a purely
# frequency-based filter cannot catch.
JARGON_STOPWORDS = set("""
bla blah blabla yada dps aoe hp mp exp xp gg ez cd dot hot buff nerf
meta tier elo mmr kd kda fps ping lag afk alt smurf
""".split())

HINDI_STOPWORDS = set("""
hai hain ho hoon hun tha thi the thing raha rahi rahe kar kiya karo karna
kya kyun kaise kahan kab jab tab ab abhi phir aur ya lekin agar toh bhi
na nahi nai mat bas sab kuch koi ek do teen mein me se ko ka ki ke par pe
apna apni mera meri tera teri uska uski hum tum tu aap woh yeh ye wo is us
jo so ne li le liya diya dena aaya gaya jata hota hoti hote karte karta
""".split())

# Single characters and bare numbers are not slang either -- except where the
# curated set says otherwise, which is how "W" and "L" survive.
_TRIVIAL = re.compile(r"^[a-z0-9]$|^\d+$")

# A closed stoplist is whack-a-mole: after removing "the" and "on", the
# annotator still fired on "was", "mom", "pound", "fon" and "wrt". The general
# rule is that a *common English word* is not evidence of slang unless a slang
# sense is independently documented.
#
# Zipf frequency: 7 is "the", 5 is "money", 4 is "monkey", 3 is "unicycle".
# Terms at or above COMMON_ZIPF are dropped unless the curated set vouches for
# them -- which is what keeps "cap", "bet", "tea", "based", "valid" and "woke".
COMMON_ZIPF = 3.6

# A slang term is a word, not a typographic gesture. The curated set carries a
# long tail of chat shorthand ("<3", "w/", "*s*", "b&", "^5", "t:)t") that is
# punctuation art rather than vocabulary; it matched nothing useful and inflated
# the slang density of any document containing stray symbols.
_WELL_FORMED = re.compile(r"^[a-z][a-z0-9'-]*$")


def is_well_formed(term: str) -> bool:
    return all(_WELL_FORMED.match(w) for w in term.split())


# Languages whose common words kept surfacing as "emerging Gen-Z slang" before
# document-level language ID was fixed: habe, auch, aber (de), muito, quando,
# sempre (pt), piu (it), sie (pl), kasi, naman (tl). Document filtering handles
# most of it now; this is a cheap second line for terms that slip through a
# mixed-language document.
OTHER_LANGS = ("de", "pt", "es", "fr", "it", "nl", "sv", "pl", "tr", "id", "tl")
FOREIGN_ZIPF = 3.0


def is_foreign_word(term: str) -> bool:
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return False
    if zipf_frequency(term, "en") >= 2.5:
        return False          # also common in English -> not evidence of foreign
    return any(zipf_frequency(term, lang) >= FOREIGN_ZIPF for lang in OTHER_LANGS)


def _zipf(term: str) -> float:
    try:
        from wordfreq import zipf_frequency

        return max(zipf_frequency(w, "en") for w in term.split())
    except Exception:
        return 0.0

_PROPER_NOUN = re.compile(r"^[A-Z][a-z]+$")
_NAME_DEF = re.compile(
    r"\b(the (sweetest|nicest|best|most amazing|kindest) (girl|boy|guy|person|man|woman)"
    r"|is a (girl|boy|guy) who|my (girlfriend|boyfriend|bestie|bff))\b",
    re.I,
)
_URL = re.compile(r"https?://\S+")


@dataclass
class LexEntry:
    term: str
    first_seen: str          # ISO date of earliest definition, or "" if unknown
    last_seen: str           # ISO date of latest definition, or "" if unknown
    n_defs: int
    median_score: float
    recency_weight: float    # 0..1, exponential decay on first_seen
    status: str              # current | aging | dead
    definition: str          # best-scoring definition, truncated
    example: str
    source: str = "urbandictionary"   # urbandictionary | curated | corpus
    corpus_rate: float = 0.0          # occurrences per million tokens, recent slice

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def recency_weight(first_seen: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    years = (now - first_seen).days / 365.25
    return float(2 ** (-max(years, 0.0) / RECENCY_HALF_LIFE_YEARS))


def classify_status(first_seen: datetime, now: datetime | None = None) -> str:
    """Provisional status from age alone.

    This is refined by `refine_status_from_corpus`, which re-attests terms
    against the recent corpus slice. Age alone over-calls `dead` for durable
    terms like "lowkey", so never ship the provisional value.
    """
    now = now or datetime.now(timezone.utc)
    years = (now - first_seen).days / 365.25
    if years <= 4:
        return "current"
    if years <= 8:
        return "aging"
    return "dead"


def is_initialism(term: str, definition: str) -> bool:
    """True when the definition is just the term's letters spelled out.

    This is what separates the two halves of the curated MLBtrio set. Its
    contemporary entries are real slang -- "cap" (a lie), "bet" (agreement),
    "tea" (gossip). Its AIM/SMS-era tail is initialisms that happen to be
    spelled like extremely common English words:

        was -> "Wait a second"     so  -> "Significant other"
        we  -> "Whatever"          wrt -> "With regard to"

    Merged unfiltered, those made the annotator fire on ordinary prose. Term
    shape cannot separate them (rizz, fam, stan and w are short and
    consonant-heavy but contemporary); what does is that an initialism's
    definition reproduces its own letters.
    """
    t = re.sub(r"[^a-z0-9]", "", term.lower())
    if not t or len(t) > 6 or " " in term.strip():
        return False
    words = re.findall(r"[a-z0-9]+", (definition or "").lower())
    if len(words) < 2:
        return False
    initials = "".join(w[0] for w in words[: len(t) + 1])
    normalised = t.replace("2", "t").replace("4", "f").replace("8", "a")
    return initials.startswith(t[:4]) or initials.startswith(normalised[:4])


def _is_junk(term: str, definition: str) -> bool:
    low = term.lower().strip()
    # Function words and bare tokens: never slang, however many Urban
    # Dictionary entries exist for them.
    if (low in STOPWORDS or low in HINDI_STOPWORDS
            or low in JARGON_STOPWORDS or _TRIVIAL.match(low)):
        return True
    if all(w in STOPWORDS or w in HINDI_STOPWORDS for w in low.split()):
        return True
    # Ordinary English is not slang on Urban Dictionary's say-so. The curated
    # set re-adds the genuine drifted senses afterwards (see merge_curated).
    if _zipf(low) >= COMMON_ZIPF:
        return True
    if len(term) < MIN_TERM_LEN or len(term.split()) > MAX_TERM_WORDS:
        return True
    if _PROPER_NOUN.match(term.strip()):      # "Ciro", "Lily" -- name entries
        return True
    if _NAME_DEF.search(definition):
        return True
    if not definition.strip():
        return True
    return False


def build_lexicon(
    rows: Iterable[dict],
    min_score: int = MIN_SCORE,
    now: datetime | None = None,
) -> list[LexEntry]:
    """Aggregate per-definition UD rows into one entry per term.

    Expects rows with keys: `word`, `definition`, `example`, `time`, `score`.
    """
    now = now or datetime.now(timezone.utc)
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        term = (row.get("word") or "").strip()
        definition = _URL.sub("", (row.get("definition") or "")).strip()
        if not term:
            continue
        try:
            score = int(row.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        if score < min_score:
            continue
        ts = _parse_time(row.get("time"))
        if ts is None:
            continue
        if _is_junk(term, definition):
            continue
        grouped[term.lower()].append(
            {"score": score, "time": ts, "definition": definition,
             "example": (row.get("example") or "").strip()}
        )

    entries: list[LexEntry] = []
    for term, defs in grouped.items():
        defs.sort(key=lambda d: d["score"], reverse=True)
        times = [d["time"] for d in defs]
        first, last = min(times), max(times)
        scores = sorted(d["score"] for d in defs)
        median = scores[len(scores) // 2]
        entries.append(
            LexEntry(
                term=term,
                first_seen=first.date().isoformat(),
                last_seen=last.date().isoformat(),
                n_defs=len(defs),
                median_score=float(median),
                recency_weight=round(recency_weight(first, now), 4),
                status=classify_status(first, now),
                definition=defs[0]["definition"][:400],
                example=defs[0]["example"][:300],
            )
        )
    entries.sort(key=lambda e: (-e.recency_weight, -e.median_score))
    return entries


def refine_status_from_corpus(
    entries: list[LexEntry],
    recent_counts: dict[str, int],
    recent_total_tokens: int,
    current_per_million: float = 1.0,
    aging_per_million: float = 0.05,
) -> list[LexEntry]:
    """Re-attest terms against a recent corpus slice.

    `recent_counts` maps term -> occurrences in text from the last ~12 months.
    A term still used frequently is `current` however old it is ("lowkey" is
    from 2010 and very much alive); a term nobody writes any more is `dead`
    however recently it was coined.
    """
    if recent_total_tokens <= 0:
        return entries
    per_m = 1e6 / recent_total_tokens
    for e in entries:
        rate = recent_counts.get(e.term, 0) * per_m
        e.corpus_rate = round(rate, 3)
        if rate >= current_per_million:
            e.status = "current"
        elif rate >= aging_per_million:
            e.status = "aging"
        else:
            e.status = "dead"
    return entries


# Contemporary slang whose surface form is an ordinary English word, so the
# frequency filter removes it and MLBtrio does not always cover it.
#
# Measured: without these, "bro is cooked lmaooo" and "she ate that, no crumbs"
# scored slang_0. The frequency filter is still right in general -- it is what
# stops "the" and "was" being slang -- but drifted senses need naming.
DRIFTED_SENSES = {
    "cooked": "ruined, defeated, or (of a person) doing extremely well",
    "mid": "mediocre, disappointing",
    "bro": "term of address; also dismissive third person ('bro thinks he...')",
    "ate": "performed excellently",
    "slay": "to do something impressively well",
    "fire": "excellent",
    "clean": "stylish, well-executed",
    "dunked": "publicly outdone or humiliated someone",
    "cracked": "extremely skilled",
    "washed": "past one's peak",
    "goated": "the greatest of all time",
    "salty": "bitter, resentful",
    "ratioed": "outnumbered by replies disagreeing with you",
    "cringe": "embarrassing to witness",
    "based": "admirably unconcerned with others' approval",
    "sending": "making one laugh uncontrollably",
    "living": "thoroughly enjoying something",
    "eating": "performing excellently",
    "cooking": "doing something impressively",
    "lock in": "to focus intensely",
    "crash out": "to lose emotional control",
    "hits different": "is notably better in a particular context",
    "no crumbs": "flawlessly, leaving nothing wanting",
    # Common English words whose contemporary sense is a genuine register
    # marker. Everything here is frequency-exempt, so the bar is a *documented*
    # drift, not merely a word Urban Dictionary happens to have an entry for.
    "cap": "a lie ('no cap' = no lie)",
    "bet": "agreement; 'all right, understood'",
    "tea": "gossip, the interesting details",
    "facts": "an expression of emphatic agreement",
    "valid": "legitimate, deserving of respect",
    "ratio": "a reply outperforming the post it answers",
    "woke": "alert to social injustice; often pejorative",
    "lit": "exciting, excellent",
    "shade": "a veiled insult ('throwing shade')",
    "karen": "an entitled, demanding person",
    "stan": "to be an intense fan of",
    "shook": "badly shaken, startled",
    "savage": "brutally direct, without regard for feelings",
    "clown": "a person behaving foolishly; to mock",
    "chad": "a stereotypically confident, dominant man",
    "snack": "an attractive person",
    "jelly": "jealous",
    "thirsty": "conspicuously desperate for attention",
    "flex": "to show off; an act of showing off",
    "wig": "an expression of astonishment ('wig snatched')",
    "sis": "term of address, regardless of gender",
    "basic": "conventional, unoriginal in taste",
    "extra": "excessive, doing far too much",
    "ship": "to want two people to be in a relationship",
    "smash": "to have sex with",
    "drag": "to criticise someone at length and in public",
    "arc": "a phase in someone's life ('villain arc')",
    "ion": "I do not ('ion even care')",
    "glow-up": "a dramatic improvement in appearance",
    "high-key": "openly, emphatically",
    "e-girl": "a woman with an online-subculture aesthetic",
    "e-boy": "a man with an online-subculture aesthetic",
    # Ordinary informal register. The lexicon is Urban-Dictionary-derived, so
    # it carried exotic coinages while missing the common markers that
    # actually distinguish chat from prose: "yo anyone wanna give me somethin
    # for free haha" measured slang_0 because not one of its words was known.
    "yo": "informal greeting or attention marker",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "have got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "lemme": "let me",
    "gimme": "give me",
    "ain't": "am not / is not / are not",
    "dude": "term of address, regardless of gender",
    "bruh": "term of address; also an expression of dismay",
    "sucks": "is bad",
    "crap": "rubbish, nonsense",
    "prick": "an unpleasant person",
    "nah": "no",
    "yeah": "yes",
    "yep": "yes",
    "nope": "no",
    "hella": "very",
    "wack": "bad, objectionable",
    "legit": "genuinely, genuine",
    "vibe": "an atmosphere or feeling",
    "sus": "suspicious",
    "bail": "to leave abruptly",
    "ghost": "to cut off contact without explanation",
    "salty": "bitter, resentful",
    "shady": "untrustworthy",
    "creep": "an unsettling person",
    "dupe": "a duplicate",
}


# Drifted senses whose literal reading is at least as common as the slang one.
# A bag-of-words matcher has no way to tell "bone moist dries out when out
# living thing" from "I'm living for this", so these count only when the
# document already contains an unambiguous slang term. That costs a little
# recall on documents whose ONLY slang is one ambiguous word, and removes a
# large class of false positives on ordinary prose.
# Current slang the dated sources cannot supply. The Urban Dictionary dump
# ends 2023-11-09 and the curated set predates the present wave, so 29 of 62
# core contemporary terms were absent -- including "bussin", which the project
# is named after. A dated dictionary can say what is *dead*; it cannot say
# what is *current*. These are unambiguous coinages, so they also serve as the
# anchors that let AMBIGUOUS_SENSES count at full weight.
CURRENT_SLANG = {
    "bussin": "excellent, especially of food",
    "gyatt": "exclamation at a large backside",
    "sigma": "self-reliant, admirably aloof (often ironic)",
    "skibidi": "nonsense intensifier from a viral series",
    "fanum tax": "taking a portion of a friend's food",
    "mewing": "tongue posture claimed to sharpen the jawline",
    "looksmaxxing": "systematically improving one's appearance",
    "cheugy": "out of date, trying too hard",
    "mogging": "visibly outclassing someone in looks",
    "glazing": "excessive, fawning praise",
    "yap": "to talk at length about nothing",
    "yapping": "talking at length about nothing",
    "beige flag": "a trait that is neither good nor bad, merely odd",
    "talking stage": "the period before a relationship is official",
    "situationship": "an undefined romantic arrangement",
    "periodt": "emphatic full stop to a statement",
    "fit": "an outfit",
    "drip": "stylish clothing",
    "opps": "enemies, rivals",
    "down bad": "desperate, humiliatingly infatuated",
    "pressed": "visibly upset, bothered",
    "chopped": "unattractive; badly done",
    "aura": "one's presence or charisma, scored in points",
    "locked in": "intensely focused",
    "crashout": "a loss of emotional control",
    "crashing out": "losing emotional control",
    "delulu": "deluded, wishfully unrealistic",
    "unserious": "absurd, not to be taken seriously",
    "clapped": "unattractive, worn out",
    "simp": "someone excessively deferential to a love interest",
    "bop": "an excellent song",
    "bops": "excellent songs",
    "slaps": "is excellent, especially of music",
    "ick": "a sudden turn-off",
    "finna": "fixing to, about to",
    "bouta": "about to",
    "boutta": "about to",
    "tryna": "trying to",
    "deadass": "seriously, genuinely",
}

AMBIGUOUS_SENSES = {
    # Literal sense dominates: these are ordinary words far more often than
    # they are register markers.
    "living", "clean", "hard", "arc", "ace", "drag", "ship", "extra",
    "basic", "valid", "burn", "shade", "camp", "peak", "creep", "smash",
    "eating", "cooking", "sending", "fire", "stone", "lit", "down",
}

# How much an ambiguous term counts when nothing unambiguous corroborates it.
# Not zero: "you ate and left no crumbs" is saturated slang whose every term
# is individually ambiguous, and gating them out scored it 0. Not one:
# "the living room was clean" is not slang at all.
AMBIGUOUS_UNANCHORED = 0.3


def merge_curated(entries: list[LexEntry], curated: Iterable[dict]) -> list[LexEntry]:
    """Fold in a hand-curated set (e.g. MLBtrio's 1,779 verified entries).

    Curated terms carry no attestation date. **Do not fabricate one** -- stamping
    them with today's date corrupts the recency signal that the whole lexicon
    exists to provide, and makes a 2019 term look freshly coined. They get an
    empty `first_seen` and are trusted as `current` on the strength of the
    curation instead, with their real date to be recovered by
    `discover_emerging_terms` if they appear in the recent corpus.
    """
    by_term = {e.term: e for e in entries}
    for row in curated:
        term = (row.get("Slang") or row.get("term") or "").strip().lower()
        if not term or len(term.split()) > MAX_TERM_WORDS:
            continue
        definition = (row.get("Description") or row.get("definition") or "").strip()
        example = (row.get("Example") or row.get("example") or "").strip()
        # The curated set is exempt from the common-word filter -- that is how
        # "cap", "bet" and "tea" survive -- but not from the initialism check,
        # which is what its legacy tail actually is.
        # The curated set is exempt from the frequency filter, not from the
        # structural ones: "9" and "oh" were reaching the lexicon this way.
        if is_initialism(term, definition):
            continue
        if (term in STOPWORDS or term in HINDI_STOPWORDS
                or term in JARGON_STOPWORDS or _TRIVIAL.match(term)):
            continue
        if not is_well_formed(term):
            continue
        # The curated set is exempt from the frequency filter only where the
        # drifted sense is documented. Without that condition the exemption
        # admitted "add", "mom", "tank", "camp" and "lab" as Gen-Z slang, and a
        # single one of them in a ten-word chat message scored slang_4.
        if _zipf(term) >= COMMON_ZIPF and term not in DRIFTED_SENSES:
            continue
        if term in by_term:
            existing = by_term[term]
            existing.status = "current"
            existing.recency_weight = max(existing.recency_weight, 0.8)
            if not existing.definition:
                existing.definition = definition[:400]
        else:
            by_term[term] = LexEntry(
                term=term, first_seen="", last_seen="", n_defs=1,
                median_score=50.0, recency_weight=0.85, status="current",
                definition=definition[:400], example=example[:300],
                source="curated",
            )
    # Drifted senses: common English words with documented contemporary slang
    # meanings. Added last so they survive every earlier filter.
    for term, definition in DRIFTED_SENSES.items():
        if term not in by_term:
            by_term[term] = LexEntry(
                term=term, first_seen="", last_seen="", n_defs=1,
                median_score=60.0, recency_weight=0.95, status="current",
                definition=definition, example="", source="curated",
            )
        else:
            by_term[term].status = "current"
            by_term[term].recency_weight = max(by_term[term].recency_weight, 0.9)

    out = list(by_term.values())
    out.sort(key=lambda e: (-e.recency_weight, -e.median_score))
    return out


# ------------------------------------------------------------------ #
# Emerging-term discovery
# ------------------------------------------------------------------ #

# Measured 2026-09-17: the largest public Urban Dictionary dump
# (georgiyozhegov/urbandictionary) has its most recent definition dated
# 2023-11-09. Probing 19 contemporary terms against the built lexicon, 13 were
# absent -- gyatt, delulu, mogging, looksmaxxing, skibidi, sigma, aura, cooked,
# bussin, yap, glazing, pookie, "crash out".
#
# So a dated dictionary can tell you what is *dead*, but it structurally cannot
# tell you what is *current*: lexicographers lag usage by years, and the dumps
# lag the lexicographers. Current slang has to be discovered from the corpus.
UD_DUMP_CUTOFF = "2023-11-09"


def discover_emerging_terms(
    recent_counts: dict[str, int],
    recent_total_tokens: int,
    known_terms: set[str],
    english_vocab: set[str],
    min_per_million: float = 2.0,
    min_len: int = 3,
    max_len: int = 20,
    baseline_counts: dict[str, int] | None = None,
    baseline_total_tokens: int | None = None,
    min_growth: float = 3.0,
) -> list[LexEntry]:
    """Find slang the dictionaries have not caught up with yet.

    A token is emerging slang when it is (a) frequent in recent social text,
    (b) absent from a standard English vocabulary, and (c) either unknown to the
    lexicon or growing sharply against an older baseline slice.

    This is how "gyatt" gets found: nobody wrote a dictionary entry for it
    before the dump was taken, but it is everywhere in 2024-2026 Reddit.

    Args:
        recent_counts: token -> count over the recent corpus slice.
        english_vocab: a standard-English word list (e.g. from `wordfreq` or
            the tokenizer's high-frequency prose tokens).
        baseline_counts: the same counts over an older slice. Supplying it turns
            the filter from "unknown word" into "unknown *or* rising word",
            which also catches semantic drift ("mid", "cooked").
    """
    if recent_total_tokens <= 0:
        return []
    per_m = 1e6 / recent_total_tokens
    base_per_m = (
        1e6 / baseline_total_tokens
        if baseline_counts and baseline_total_tokens
        else None
    )

    found: list[LexEntry] = []
    for term, count in recent_counts.items():
        if not (min_len <= len(term) <= max_len) or not term.isalpha():
            continue
        if is_foreign_word(term):
            continue
        rate = count * per_m
        if rate < min_per_million:
            continue
        if term in english_vocab:
            # A known English word is only interesting as *semantic drift* --
            # "mid" and "cooked" acquiring pejorative senses. Detecting that
            # from a frequency ratio alone does not work unless the two slices
            # are topic-matched, and in practice they never are: comparing an
            # all-of-Reddit scrape against three named subreddits makes
            # "anyone", "currently" and "team" look like emerging slang, because
            # what actually changed was the topic mix, not the language.
            #
            # So require independent evidence that a slang sense exists at all
            # (the term is in the lexicon) before trusting a growth signal.
            if base_per_m is None or term not in known_terms:
                continue
            old = (baseline_counts.get(term, 0) or 0) * base_per_m
            if old > 0 and rate / old < min_growth:
                continue
        elif term in known_terms:
            continue  # already in the lexicon with a real date

        found.append(
            LexEntry(
                term=term, first_seen="", last_seen="", n_defs=0,
                median_score=0.0,
                # Discovered in the recent slice, so by construction current.
                recency_weight=1.0, status="current",
                definition="", example="", source="corpus",
                corpus_rate=round(rate, 3),
            )
        )
    found.sort(key=lambda e: -e.corpus_rate)
    return found


# ------------------------------------------------------------------ #
# Runtime lookup
# ------------------------------------------------------------------ #


# Status floors for the matching weight.
#
# Weighting by recency alone inverted the signal it was meant to carry. A term
# Urban Dictionary first recorded in 2010 decays to ~0.005 however common it is
# today, so corpus-attested slang like "tripping" contributed nothing, while
# junk injected through the curated path carried 0.85. Corpus attestation is
# the stronger evidence: `refine_status_from_corpus` has already measured
# whether the term is in live use. Recency now only modulates within a status.
STATUS_BASE = {"current": 1.0, "aging": 0.5, "dead": 0.0}


def effective_weight(entry: "LexEntry") -> float:
    base = STATUS_BASE.get(entry.status, 0.3)
    if base == 0.0:
        return 0.0
    return base * (0.5 + 0.5 * entry.recency_weight)


class Lexicon:
    """Fast lookup used by the register annotator and the miner."""

    def __init__(self, entries: list[LexEntry]) -> None:
        self.entries = entries
        self.by_term: dict[str, LexEntry] = {e.term: e for e in entries}
        self.weights: dict[str, float] = {
            e.term: w for e in entries if (w := effective_weight(e)) > 0.0
        }
        # Multi-word terms need phrase matching, so keep them separate.
        self.phrases: dict[str, float] = {
            e.term: self.weights[e.term] for e in entries
            if " " in e.term and e.term in self.weights
        }
        self.max_phrase_len = max((len(p.split()) for p in self.phrases), default=1)

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        # Admissibility is enforced here as well as at build time. A lexicon
        # file outlives the code that produced it -- it is shipped to workers
        # and re-attested in place -- so the runtime must not trust that the
        # filters in force when it was written are the ones in force now.
        # Entries are also deduplicated: re-attestation appends, and duplicate
        # terms were reaching the file.
        seen: set[str] = set()
        entries: list[LexEntry] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = LexEntry(**json.loads(line))
            if e.term in seen or not cls.admissible(e):
                continue
            seen.add(e.term)
            entries.append(e)

        # DRIFTED_SENSES is defined in code, not in the file, so a lexicon
        # built before a term was curated would silently omit it -- which is
        # how "yo", "wanna" and "dude" stayed missing after being added.
        for term, definition in {**DRIFTED_SENSES, **CURRENT_SLANG}.items():
            if term in seen:
                continue
            entries.append(LexEntry(
                term=term, first_seen="", last_seen="", n_defs=1,
                median_score=60.0, recency_weight=0.95, status="current",
                definition=definition, example="", source="curated",
            ))
        return cls(entries)

    @staticmethod
    def admissible(e: "LexEntry") -> bool:
        t = e.term
        if not t or not is_well_formed(t) or _TRIVIAL.match(t):
            return False
        if t in STOPWORDS or t in HINDI_STOPWORDS or t in JARGON_STOPWORDS:
            return False
        if all(w in STOPWORDS or w in HINDI_STOPWORDS for w in t.split()):
            return False
        if _zipf(t) >= COMMON_ZIPF and t not in DRIFTED_SENSES:
            return False
        return True

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for e in self.entries:
                fh.write(e.to_json() + "\n")

    def terms(self, status: str | None = None) -> list[str]:
        if status is None:
            return [e.term for e in self.entries]
        return [e.term for e in self.entries if e.status == status]

    def single_word_current(self) -> list[str]:
        """Terms the tokenizer must encode as one token (SPEC §6.5)."""
        return [e.term for e in self.entries if e.status == "current" and " " not in e.term]

    def hits(self, tokens: list[str]) -> list[tuple[str, float]]:
        """Recency-weighted slang hits in a tokenised document."""
        found: list[tuple[str, float]] = []
        lowered = [t.lower() for t in tokens]

        for i, tok in enumerate(lowered):
            w = self.weights.get(tok)
            if w is not None and " " not in tok:
                found.append((tok, w))
            # greedy phrase match
            for n in range(2, self.max_phrase_len + 1):
                if i + n > len(lowered):
                    break
                phrase = " ".join(lowered[i : i + n])
                pw = self.phrases.get(phrase)
                if pw is not None:
                    found.append((phrase, pw))
        return found

    def density(self, tokens: list[str]) -> float:
        """Recency-weighted slang hits per 100 tokens."""
        if not tokens:
            return 0.0
        return 100.0 * sum(w for _, w in self.hits(tokens)) / len(tokens)

    def stats(self) -> dict[str, int]:
        by_status: dict[str, int] = defaultdict(int)
        for e in self.entries:
            by_status[e.status] += 1
        return {"total": len(self.entries), **dict(by_status)}
