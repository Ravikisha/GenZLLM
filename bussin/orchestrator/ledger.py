"""Quota ledger.

**Kaggle does not expose quota through its API.** The settings page shows
`Kaggle GPU 00:00 / 30 hrs` and `Kaggle TPU 00:00 / 20 hrs`, but there is no
documented endpoint behind it. So the orchestrator keeps its own book: it
records every dispatch and accrues the elapsed time against a reset boundary.

Two consequences that shape the design:

* **Estimate high, not low.** A session that overruns the real quota gets
  killed mid-step and wastes the checkpoint interval, so usage is accrued
  optimistically-for-safety (a dispatch counts from submission, and unknown
  outcomes count as full sessions) and the usable budget is discounted.
* **Reconcile occasionally.** `reconcile()` lets a human paste the real numbers
  from the settings page, which corrects drift without re-plumbing anything.

The ledger lives beside `RUN_STATE.json` in the checkpoint repo and uses the
same compare-and-swap, so two orchestrator ticks can never double-spend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any

SAFETY_MARGIN = 0.90  # treat 30h as 27h usable


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# ------------------------------------------------------------------ #
# Reset schedules
# ------------------------------------------------------------------ #


def last_weekly_reset(weekday: int = 5, hour: int = 0, now: datetime | None = None) -> datetime:
    """Most recent reset boundary. Kaggle resets Saturday 00:00 UTC (weekday 5)."""
    now = now or utcnow()
    candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    days_since = (now.weekday() - weekday) % 7
    candidate -= timedelta(days=days_since)
    if candidate > now:
        candidate -= timedelta(days=7)
    return candidate


def next_weekly_reset(weekday: int = 5, hour: int = 0, now: datetime | None = None) -> datetime:
    return last_weekly_reset(weekday, hour, now) + timedelta(days=7)


def last_daily_reset(hour: int = 0, now: datetime | None = None) -> datetime:
    now = now or utcnow()
    c = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return c if c <= now else c - timedelta(days=1)


def next_daily_reset(hour: int = 0, now: datetime | None = None) -> datetime:
    return last_daily_reset(hour, now) + timedelta(days=1)


# ------------------------------------------------------------------ #
# Platform budgets
# ------------------------------------------------------------------ #


@dataclass
class PlatformBudget:
    name: str
    quota_hours: float
    period: str                 # weekly | daily | monthly
    effective_tflops: float     # for ranking; measured in Phase 3
    session_hours: float        # max single session
    min_useful_hours: float     # below this, dispatching wastes startup cost
    automatable: bool = True
    reset_weekday: int = 5      # Saturday, for weekly
    notes: str = ""


# Throughput figures are the midpoints from SPEC §9.6 and are ESTIMATES until
# measured on real hardware in Phase 3.
BUDGETS: dict[str, PlatformBudget] = {
    "kaggle-tpu": PlatformBudget(
        "kaggle-tpu", 20.0, "weekly", 115.0, 9.0, 1.0,
        notes="TPU v3-8. Separate quota from GPU -- verified on the account.",
    ),
    "kaggle-gpu": PlatformBudget(
        "kaggle-gpu", 30.0, "weekly", 30.0, 12.0, 1.0,
        notes="T4 x2. Never select P100: no tensor cores, ~6x slower.",
    ),
    "lightning": PlatformBudget(
        "lightning", 15.0, "monthly", 36.0, 4.0, 0.75,
        notes="Free credits; L4 supports bf16.",
    ),
    "colab": PlatformBudget(
        "colab", 12.0, "daily", 15.0, 4.0, 0.5,
        notes="Pre-emptible. Official CLI supports headless `colab run`.",
    ),
}


# ------------------------------------------------------------------ #
# Ledger
# ------------------------------------------------------------------ #


@dataclass
class Dispatch:
    platform: str
    worker_id: str
    dispatched_at: str
    finished_at: str | None = None
    hours: float | None = None          # billed hours, once known
    status: str = "running"             # running | done | failed | unknown
    detail: str = ""

    def billed(self, budget: PlatformBudget, now: datetime | None = None) -> float:
        """Hours to charge against the quota.

        An unfinished dispatch is charged its elapsed time so far, and an
        abandoned one is charged a whole session -- overcharging is recoverable,
        undercharging gets a session killed mid-step.
        """
        if self.hours is not None:
            return self.hours
        start = parse(self.dispatched_at)
        if not start:
            return budget.session_hours
        elapsed = ((now or utcnow()) - start).total_seconds() / 3600
        if self.status == "running":
            # Cap at the session limit: a "running" record older than that is
            # a dispatch whose completion we never saw.
            return min(elapsed, budget.session_hours)
        return min(max(elapsed, 0.0), budget.session_hours)


@dataclass
class Ledger:
    dispatches: list[dict[str, Any]] = field(default_factory=list)
    reconciled: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated_at: str = ""

    # -------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Ledger":
        if not d:
            return cls()
        return cls(
            dispatches=d.get("dispatches", []),
            reconciled=d.get("reconciled", {}),
            updated_at=d.get("updated_at", ""),
        )

    # -------------------------------------------------------------- #

    def period_start(self, platform: str, now: datetime | None = None) -> datetime:
        b = BUDGETS[platform]
        if b.period == "weekly":
            return last_weekly_reset(b.reset_weekday, now=now)
        if b.period == "daily":
            return last_daily_reset(now=now)
        now = now or utcnow()
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def next_reset(self, platform: str, now: datetime | None = None) -> datetime:
        b = BUDGETS[platform]
        if b.period == "weekly":
            return next_weekly_reset(b.reset_weekday, now=now)
        if b.period == "daily":
            return next_daily_reset(now=now)
        start = self.period_start(platform, now)
        return (start + timedelta(days=32)).replace(day=1)

    def used_hours(self, platform: str, now: datetime | None = None) -> float:
        now = now or utcnow()
        b = BUDGETS[platform]
        start = self.period_start(platform, now)

        # A human-supplied reading from the settings page overrides accrual for
        # anything before the reading's timestamp.
        base = 0.0
        since = start
        rec = self.reconciled.get(platform)
        if rec and (t := parse(rec.get("at"))) and t >= start:
            base = float(rec.get("used_hours", 0.0))
            since = t

        for raw in self.dispatches:
            d = Dispatch(**raw)
            if d.platform != platform:
                continue
            at = parse(d.dispatched_at)
            if not at or at < since:
                continue
            base += d.billed(b, now)
        return base

    def remaining_hours(self, platform: str, now: datetime | None = None) -> float:
        b = BUDGETS[platform]
        return max(b.quota_hours * SAFETY_MARGIN - self.used_hours(platform, now), 0.0)

    def available(self, platform: str, now: datetime | None = None) -> bool:
        return self.remaining_hours(platform, now) >= BUDGETS[platform].min_useful_hours

    def has_running(self, now: datetime | None = None) -> Dispatch | None:
        now = now or utcnow()
        for raw in reversed(self.dispatches):
            d = Dispatch(**raw)
            if d.status != "running":
                continue
            start = parse(d.dispatched_at)
            budget = BUDGETS.get(d.platform)
            if not start or not budget:
                continue
            # Older than a session plus slack: treat as lost, not running.
            if (now - start).total_seconds() / 3600 < budget.session_hours + 1:
                return d
        return None

    # -------------------------------------------------------------- #

    def record_dispatch(self, platform: str, worker_id: str, detail: str = "") -> Dispatch:
        d = Dispatch(platform=platform, worker_id=worker_id,
                     dispatched_at=iso(utcnow()), detail=detail)
        self.dispatches.append(asdict(d))
        self.updated_at = iso(utcnow())
        return d

    def close_dispatch(self, worker_id: str, status: str = "done",
                       hours: float | None = None, detail: str = "") -> bool:
        for raw in reversed(self.dispatches):
            if raw.get("worker_id") == worker_id and raw.get("status") == "running":
                raw["status"] = status
                raw["finished_at"] = iso(utcnow())
                if hours is not None:
                    raw["hours"] = round(hours, 4)
                else:
                    start = parse(raw["dispatched_at"])
                    if start:
                        raw["hours"] = round(
                            (utcnow() - start).total_seconds() / 3600, 4)
                if detail:
                    raw["detail"] = detail
                self.updated_at = iso(utcnow())
                return True
        return False

    def reconcile(self, platform: str, used_hours: float, note: str = "") -> None:
        """Record a real reading from the platform's settings page."""
        self.reconciled[platform] = {
            "at": iso(utcnow()), "used_hours": used_hours, "note": note,
        }
        # Accrued records before this reading are now superseded.
        self.dispatches = [
            d for d in self.dispatches
            if d["platform"] != platform or (parse(d["dispatched_at"]) or utcnow())
            >= (parse(self.reconciled[platform]["at"]) or utcnow())
        ]
        self.updated_at = iso(utcnow())

    def prune(self, keep_days: int = 60) -> int:
        cutoff = utcnow() - timedelta(days=keep_days)
        before = len(self.dispatches)
        self.dispatches = [
            d for d in self.dispatches if (parse(d["dispatched_at"]) or utcnow()) >= cutoff
        ]
        return before - len(self.dispatches)

    # -------------------------------------------------------------- #

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        """Everything the dashboard needs about quota."""
        now = now or utcnow()
        out: dict[str, Any] = {"generated_at": iso(now), "platforms": {}}
        for name, b in BUDGETS.items():
            used = self.used_hours(name, now)
            remaining = self.remaining_hours(name, now)
            nxt = self.next_reset(name, now)
            out["platforms"][name] = {
                "quota_hours": b.quota_hours,
                "usable_hours": round(b.quota_hours * SAFETY_MARGIN, 2),
                "used_hours": round(used, 2),
                "remaining_hours": round(remaining, 2),
                "pct_used": round(100 * used / max(b.quota_hours, 1e-9), 1),
                "period": b.period,
                "next_reset": iso(nxt),
                "hours_to_reset": round((nxt - now).total_seconds() / 3600, 2),
                "available": self.available(name, now),
                "automatable": b.automatable,
                "effective_tflops": b.effective_tflops,
                "session_hours": b.session_hours,
                "notes": b.notes,
            }
        running = self.has_running(now)
        out["running"] = asdict(running) if running else None
        out["recent"] = self.dispatches[-25:]
        return out


def rank_platforms(ledger: Ledger, now: datetime | None = None,
                   only_automatable: bool = True) -> list[str]:
    """Best platform first: highest throughput among those with budget left."""
    candidates = [
        name for name, b in BUDGETS.items()
        if (b.automatable or not only_automatable) and ledger.available(name, now)
    ]
    return sorted(candidates, key=lambda n: -BUDGETS[n].effective_tflops)
