"""Register and anti-cringe metrics.

The payoff of building the register annotator is that evaluation is nearly free:
the same code that annotates 70 GB of corpus also scores model outputs, so
"cringe" stops being a vibe and becomes a number.

The headline metric is **Register Calibration Error** (SPEC §15.3): mean
absolute difference between the register that was *requested* and the register
that was *produced*. It rises both when the model under-uses slang and when it
over-uses it, so unlike "average slang density" it cannot be gamed by
suppressing slang.

The optimisation target is **distribution matching**, not maximisation: at a
given register, the model's slang-density histogram should match the histogram
of real human text at that register. Not more slang. Not less. The same.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from statistics import median
from typing import Sequence

from ..data.lexicon import Lexicon
from ..data.register import DIM_ORDER, Register, RegisterAnnotator


# ------------------------------------------------------------------ #
# Register calibration
# ------------------------------------------------------------------ #


def register_calibration_error(
    annotator: RegisterAnnotator,
    requested: Sequence[Register],
    produced: Sequence[str],
) -> dict[str, float]:
    """RCE, overall and per dimension. Target < 0.4 on 0-4 scales.

    Report per-dimension as well as overall: failure is usually concentrated,
    and a model can nail `emoji` while completely ignoring `formal`.
    """
    return annotator.calibration_error(requested, produced)


def context_adaptation(
    annotator: RegisterAnnotator,
    prompts: Sequence[str],
    responses: Sequence[str],
) -> dict[str, float]:
    """The mirror test (SPEC §15.7).

    Feed pairs of prompts that differ *only* in register and measure whether the
    model's output register tracks the input's. Target Pearson r >= 0.75.

    A model scoring near zero is either always-formal or always-slangy, and both
    are failures -- this is the single most important behaviour in the project,
    because it is what stops slang appearing where it does not belong.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must align")

    xs, ys = [], []
    for prompt, response in zip(prompts, responses):
        pr = annotator.annotate(prompt)
        rr = annotator.annotate(response)
        # Collapse to a single informality axis for the correlation.
        xs.append(pr.slang + pr.emoji + pr.abbrev + (3 - pr.formal))
        ys.append(rr.slang + rr.emoji + rr.abbrev + (3 - rr.formal))

    return {
        "context_adaptation_r": _pearson(xs, ys),
        "n": float(len(xs)),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


# ------------------------------------------------------------------ #
# Distribution matching
# ------------------------------------------------------------------ #


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
    """First Wasserstein distance between two 1-D samples.

    Implemented directly rather than pulling in scipy, which is not guaranteed
    present on every free-tier image.
    """
    if not a or not b:
        return float("nan")
    sa, sb = sorted(a), sorted(b)
    n = max(len(sa), len(sb))
    total = 0.0
    for i in range(n):
        qa = sa[min(int(i * len(sa) / n), len(sa) - 1)]
        qb = sb[min(int(i * len(sb) / n), len(sb) - 1)]
        total += abs(qa - qb)
    return total / n


def slang_density_distance(
    annotator: RegisterAnnotator,
    model_texts: Sequence[str],
    human_texts: Sequence[str],
) -> dict[str, float]:
    """How far the model's slang-density distribution is from real humans'.

    Target < 0.15 at matched register. This is the metric that encodes
    "naturalness rather than maximum slang density": it penalises a model that
    is blander than humans *and* one that is denser.
    """
    m = [annotator.features(t).slang_density for t in model_texts]
    h = [annotator.features(t).slang_density for t in human_texts]
    return {
        "slang_density_wasserstein": wasserstein_1d(m, h),
        "model_mean_density": sum(m) / max(len(m), 1),
        "human_mean_density": sum(h) / max(len(h), 1),
    }


# ------------------------------------------------------------------ #
# Freshness, variety, leakage, repetition
# ------------------------------------------------------------------ #


def slang_freshness(
    lexicon: Lexicon, texts: Sequence[str], now: datetime | None = None
) -> dict[str, float]:
    """Median first-attestation year of the slang the model produces.

    A model reaching for terms first attested in 2013 is producing dated slang,
    which is failure mode 3 in SPEC §15.2. The corresponding lexicon status
    breakdown says the same thing more bluntly.
    """
    now = now or datetime.now(timezone.utc)
    years: list[float] = []
    status: Counter = Counter()

    for text in texts:
        words = re.findall(r"[A-Za-z']+", text.lower())
        for term, _ in lexicon.hits(words):
            entry = lexicon.by_term.get(term)
            if not entry:
                continue
            status[entry.status] += 1
            if entry.first_seen:
                try:
                    years.append(int(entry.first_seen[:4]))
                except ValueError:
                    pass

    total = sum(status.values()) or 1
    return {
        "median_slang_year": float(median(years)) if years else float("nan"),
        "n_slang_hits": float(sum(status.values())),
        "frac_current": status["current"] / total,
        "frac_aging": status["aging"] / total,
        "frac_dead": status["dead"] / total,
    }


def slang_type_token_ratio(lexicon: Lexicon, texts: Sequence[str]) -> float:
    """Unique slang terms / total slang tokens. Target >= 0.35.

    A low ratio is failure mode 2 and 6 in SPEC §15.2: the model has three
    catchphrases and deploys them constantly.
    """
    terms: list[str] = []
    for text in texts:
        words = re.findall(r"[A-Za-z']+", text.lower())
        terms.extend(t for t, _ in lexicon.hits(words))
    if not terms:
        return float("nan")
    return len(set(terms)) / len(terms)


def formal_mode_leakage(
    annotator: RegisterAnnotator, texts_at_slang_zero: Sequence[str]
) -> dict[str, float]:
    """Slang emitted when slang was explicitly set to zero. Target < 0.2/100 tok.

    This is the direct measurement of "does it randomly insert bro/fr/no cap",
    and it is only meaningful because 70% of pretraining is slang_0 -- restraint
    is a trained behaviour, not a hoped-for one.
    """
    if not texts_at_slang_zero:
        return {"formal_mode_slang_per_100": float("nan")}
    densities = [annotator.features(t).slang_density for t in texts_at_slang_zero]
    return {
        "formal_mode_slang_per_100": sum(densities) / len(densities),
        "formal_mode_clean_frac": sum(1 for d in densities if d == 0) / len(densities),
    }


def repetition_rate(texts: Sequence[str], n: int = 4) -> dict[str, float]:
    """Fraction of n-grams that repeat across generations.

    Measured across the whole sample, not within one text: the failure being
    caught is the model reusing the same phrase in every reply.
    """
    grams: Counter = Counter()
    total = 0
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            grams[tuple(words[i : i + n])] += 1
            total += 1
    if total == 0:
        return {"repetition_rate": float("nan")}
    repeated = sum(c for g, c in grams.items() if c > 1)
    return {
        "repetition_rate": repeated / total,
        "distinct_4gram_ratio": len(grams) / total,
    }


# ------------------------------------------------------------------ #
# Aggregate report
# ------------------------------------------------------------------ #


@dataclass
class CringeReport:
    rce: float
    rce_per_dim: dict[str, float]
    context_adaptation_r: float
    slang_density_wasserstein: float
    slang_ttr: float
    median_slang_year: float
    frac_dead_slang: float
    formal_mode_slang_per_100: float
    repetition_rate: float

    # Targets from SPEC §14.5 and §15.
    TARGETS = {
        "rce": ("<", 0.40),
        "context_adaptation_r": (">", 0.75),
        "slang_density_wasserstein": ("<", 0.15),
        "slang_ttr": (">", 0.35),
        "formal_mode_slang_per_100": ("<", 0.20),
        "frac_dead_slang": ("<", 0.15),
    }

    def verdict(self) -> tuple[bool, list[str]]:
        failures = []
        for key, (op, target) in self.TARGETS.items():
            value = getattr(self, key, None)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            bad = value >= target if op == "<" else value <= target
            if bad:
                failures.append(f"{key}={value:.3f} (want {op} {target})")
        return (not failures), failures

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()}
        ok, failures = self.verdict()
        d["passed"] = ok
        d["failures"] = failures
        return d


def full_report(
    annotator: RegisterAnnotator,
    lexicon: Lexicon,
    requested: Sequence[Register],
    produced: Sequence[str],
    human_reference: Sequence[str],
    formal_mode_texts: Sequence[str],
    mirror_prompts: Sequence[str] = (),
    mirror_responses: Sequence[str] = (),
) -> CringeReport:
    rce = register_calibration_error(annotator, requested, produced)
    dist = slang_density_distance(annotator, produced, human_reference)
    fresh = slang_freshness(lexicon, produced)
    leak = formal_mode_leakage(annotator, formal_mode_texts)
    rep = repetition_rate(produced)
    adapt = (
        context_adaptation(annotator, mirror_prompts, mirror_responses)
        if mirror_prompts else {"context_adaptation_r": float("nan")}
    )
    return CringeReport(
        rce=rce["rce"],
        rce_per_dim={d: rce[f"rce_{d}"] for d in DIM_ORDER},
        context_adaptation_r=adapt["context_adaptation_r"],
        slang_density_wasserstein=dist["slang_density_wasserstein"],
        slang_ttr=slang_type_token_ratio(lexicon, produced),
        median_slang_year=fresh["median_slang_year"],
        frac_dead_slang=fresh["frac_dead"],
        formal_mode_slang_per_100=leak["formal_mode_slang_per_100"],
        repetition_rate=rep["repetition_rate"],
    )
