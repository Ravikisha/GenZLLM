"""Bussin training dashboard.

Runs as a free Hugging Face Space. Reads `orchestrator/dashboard.json` from the
checkpoint repo, which the orchestrator republishes on every tick, so there is
no server to keep alive and no build step.

Set `HF_TOKEN` and `BUSSIN_CKPT_REPO` as Space secrets.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

CKPT_REPO = os.environ.get("BUSSIN_CKPT_REPO", "GodAsap/bussin-ckpt")
DASHBOARD_FILE = "orchestrator/dashboard.json"
REFRESH_SECONDS = 120

st.set_page_config(page_title="Bussin", page_icon="📊", layout="wide")


@st.cache_data(ttl=REFRESH_SECONDS)
def load() -> dict | None:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            CKPT_REPO, DASHBOARD_FILE, repo_type="model",
            token=os.environ.get("HF_TOKEN"), force_download=True,
        )
        return json.loads(open(path, encoding="utf-8").read())
    except Exception as exc:
        st.error(f"could not read {DASHBOARD_FILE} from {CKPT_REPO}: {exc}")
        return None


def fmt_hours(h: float | None) -> str:
    if h is None:
        return "--"
    if h < 1:
        return f"{h * 60:.0f}m"
    return f"{h:.1f}h"


d = load()
if not d:
    st.stop()

model = d.get("model") or {}
live = d.get("live") or {}
prog = d.get("progress") or {}
status = d.get("status") or {}

# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #

st.title(f"🅱️ {model.get('name') or 'bussin'}")
gen = d.get("generated_at", "")
st.caption(
    f"{model.get('n_params', 0):,} parameters · "
    f"updated {gen} · auto-refreshes every {REFRESH_SECONDS}s"
)

action = status.get("action")
reason = status.get("reason", "")
if status.get("finished"):
    st.success(f"Run finished. {reason}")
elif action == "dispatched":
    st.success(f"**Training dispatched** — {reason}")
elif action == "sleep":
    st.info(f"**Quotas exhausted, waiting for reset** — {reason}")
elif action == "none" and "lease held" in reason:
    st.success(f"**Training in progress** — {reason}")
else:
    st.warning(f"**Idle** — {reason}")

# ------------------------------------------------------------------ #
# Progress
# ------------------------------------------------------------------ #

c = st.columns(5)
c[0].metric("Step", f"{live.get('step', 0):,}",
            help=f"of {live.get('total_steps') or '?'}")
c[1].metric("Tokens seen", f"{prog.get('tokens_seen', 0) / 1e9:.3f}B",
            f"{prog.get('pct_complete', 0):.1f}% of plan")
c[2].metric("Loss", f"{live.get('loss'):.4f}" if live.get("loss") else "--")
c[3].metric("Perplexity",
            f"{live.get('perplexity'):.1f}" if live.get("perplexity") else "--")
c[4].metric("Throughput",
            f"{(live.get('tok_per_s') or 0):,} tok/s",
            help="tokens per second, last logged step")

st.progress(min(prog.get("pct_complete", 0) / 100, 1.0))

# --- the panel that actually says whether the run is on track ------
st.subheader("Training adequacy")
a = st.columns(4)
tpp = prog.get("tokens_per_param", 0)
a[0].metric("Tokens / parameter", f"{tpp:.1f}",
            help="Chinchilla-optimal is 20; the over-trained regime small "
                 "models need is 70+")
a[1].metric("vs Chinchilla", f"{prog.get('pct_of_chinchilla', 0):.0f}%")
a[2].metric("ETA",
            f"{prog.get('eta_weeks')} wk" if prog.get("eta_weeks") else "--",
            help="at the observed throughput and remaining weekly quota")
a[3].metric("Weekly capacity",
            f"{(prog.get('weekly_token_capacity') or 0) / 1e9:.1f}B tok")

if tpp:
    if tpp < 10:
        st.error(f"**{tpp:.1f} tokens/parameter — below minimally trained.** "
                 "Output will be incoherent.")
    elif tpp < 20:
        st.warning(f"**{tpp:.1f} tokens/parameter — below Chinchilla-optimal (20).** "
                   "A smaller model trained longer would beat this.")
    elif tpp < 70:
        st.info(f"**{tpp:.1f} tokens/parameter — past Chinchilla, short of the "
                f"70+ over-trained regime.**")
    else:
        st.success(f"**{tpp:.1f} tokens/parameter — properly trained.**")

# ------------------------------------------------------------------ #
# Quota
# ------------------------------------------------------------------ #

st.subheader("Platform quota")
quota = d.get("quota") or {}
if quota:
    rows = []
    for name, p in quota.items():
        rows.append({
            "platform": name,
            "used": f"{p['used_hours']:.1f}h",
            "quota": f"{p['quota_hours']:.0f}h",
            "remaining": fmt_hours(p["remaining_hours"]),
            "used %": p["pct_used"],
            "resets in": fmt_hours(p["hours_to_reset"]),
            "available": "yes" if p["available"] else "no",
            "auto": "yes" if p.get("automatable") else "manual",
            "TFLOP/s": p.get("effective_tflops"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={
            "used %": st.column_config.ProgressColumn(
                "used %", min_value=0, max_value=100, format="%.0f%%")
        },
    )
    st.caption(
        "Quota is **estimated**: Kaggle does not expose it via API, so the "
        "orchestrator accrues it from dispatch durations against a 90% safety "
        "margin. Reconcile against the settings page occasionally."
    )

# ------------------------------------------------------------------ #
# Charts
# ------------------------------------------------------------------ #

charts = d.get("charts") or {}
if charts.get("loss"):
    st.subheader("Training")
    left, right = st.columns(2)
    loss = pd.DataFrame(charts["loss"], columns=["step", "loss"]).set_index("step")
    left.line_chart(loss, height=260)
    left.caption("loss")
    if charts.get("lr"):
        lr = pd.DataFrame(charts["lr"], columns=["step", "lr"]).set_index("step")
        right.line_chart(lr, height=260)
        right.caption("learning rate — warmup, stable, then the decay that is "
                      "also the Gen-Z anneal")
    l2, r2 = st.columns(2)
    if charts.get("grad_norm"):
        gn = pd.DataFrame(charts["grad_norm"],
                          columns=["step", "grad_norm"]).set_index("step")
        l2.line_chart(gn, height=200)
        l2.caption("gradient norm")
    if charts.get("tok_per_s"):
        tp = pd.DataFrame(charts["tok_per_s"],
                          columns=["step", "tok_per_s"]).set_index("step")
        r2.line_chart(tp, height=200)
        r2.caption("throughput")

# ------------------------------------------------------------------ #
# Benchmark
# ------------------------------------------------------------------ #

st.subheader("BUSSBENCH")
bb = d.get("bussbench")
if bb:
    st.dataframe(pd.DataFrame(bb), hide_index=True, use_container_width=True)
else:
    st.info(
        "No benchmark results yet. **Pretraining reports loss and perplexity, "
        "not accuracy** — accuracy exists only once BUSSBENCH runs, which costs "
        "GPU time and so happens at milestones."
    )

# ------------------------------------------------------------------ #
# Sessions
# ------------------------------------------------------------------ #

sess = d.get("sessions") or {}
st.subheader(f"Sessions ({sess.get('count', 0)})")
eff = sess.get("mean_efficiency")
if eff is not None:
    st.metric("Mean session efficiency", f"{eff:.1%}",
              help="fraction of wall-clock inside a session spent on training "
                   "steps; target is 95%+")
    if eff < 0.90:
        st.warning("Below 90% — quota is going to setup, downloads or "
                   "validation rather than matrix multiplies.")
if sess.get("recent"):
    st.dataframe(pd.DataFrame(sess["recent"]), hide_index=True,
                 use_container_width=True)

if d.get("recent_dispatches"):
    with st.expander("Recent dispatches"):
        st.dataframe(pd.DataFrame(d["recent_dispatches"]), hide_index=True,
                     use_container_width=True)

for note in d.get("notes", []):
    st.caption(f"· {note}")
