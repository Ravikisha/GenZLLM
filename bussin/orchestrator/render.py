"""Render the dashboard to a self-contained static page.

Hugging Face now bills Gradio and Docker Spaces on free CPU (`402 Payment
Required: hosting Gradio and Docker Spaces on free cpu-basic requires a PRO
subscription`); **static Spaces remain free for everyone**.

That constraint pushes towards a better design anyway. The orchestrator already
runs every 15 minutes and already holds the data, so it renders the page and
pushes it. Consequences:

* no runtime, no cold start, no PRO
* the checkpoint repo stays private -- the browser never needs a token,
  because nothing is fetched client-side
* the same file works on a static Space, GitHub Pages, or straight off disk

Charts are inline SVG. No CDN, no JavaScript framework, no build step.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

CHINCHILLA, WELL_TRAINED = 20.0, 70.0


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else "--"))


def _sparkline(points: Sequence[Sequence[float]], width: int = 520,
               height: int = 150, colour: str = "#7c5cff",
               logy: bool = False) -> str:
    pts = [(float(a), float(b)) for a, b in (points or []) if b is not None]
    if len(pts) < 2:
        return '<p class="muted">no data yet</p>'
    import math

    xs = [p[0] for p in pts]
    ys = [math.log10(max(p[1], 1e-9)) if logy else p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    pad = 6

    def sx(x: float) -> float:
        return pad + (x - x0) / dx * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - y0) / dy * (height - 2 * pad)

    d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                 for i, (x, y) in enumerate(zip(xs, ys)))
    area = (f"M{sx(xs[0]):.1f},{height - pad:.1f} "
            + " ".join(f"L{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
            + f" L{sx(xs[-1]):.1f},{height - pad:.1f} Z")
    lo, hi = (pts[0][1], pts[-1][1])
    return f"""<svg viewBox="0 0 {width} {height}" class="chart" role="img">
  <path d="{area}" fill="{colour}" opacity="0.12"/>
  <path d="{d}" fill="none" stroke="{colour}" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>
  <text x="{pad}" y="14" class="axis">{hi if logy else f'{max(p[1] for p in pts):.4g}'}</text>
  <text x="{pad}" y="{height - 2}" class="axis">{min(p[1] for p in pts):.4g}</text>
  <text x="{width - pad}" y="{height - 2}" class="axis" text-anchor="end">step {int(x1):,}</text>
</svg>"""


def _verdict(tpp: float) -> tuple[str, str]:
    if tpp <= 0:
        return "idle", "Not started."
    if tpp < 10:
        return "bad", (f"{tpp:.1f} tokens/parameter — below minimally trained. "
                       "Output will be incoherent.")
    if tpp < CHINCHILLA:
        return "warn", (f"{tpp:.1f} tokens/parameter — below Chinchilla-optimal "
                        f"({CHINCHILLA:.0f}). A smaller model trained longer "
                        f"would beat this.")
    if tpp < WELL_TRAINED:
        return "ok", (f"{tpp:.1f} tokens/parameter — past Chinchilla, short of "
                      f"the {WELL_TRAINED:.0f}+ regime small models need.")
    return "good", f"{tpp:.1f} tokens/parameter — properly trained."


CSS = """
:root{--bg:#f7f7fb;--fg:#14141c;--muted:#6b6b7b;--card:#fff;--line:#e6e6ef;
--accent:#7c5cff;--good:#17916b;--ok:#2b7fd4;--warn:#b7791f;--bad:#c1392b;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0f0f14;--fg:#e9e9f2;--muted:#9a9aae;--card:#17171f;--line:#262634;}}
:root[data-theme="dark"]{--bg:#0f0f14;--fg:#e9e9f2;--muted:#9a9aae;
--card:#17171f;--line:#262634;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,
system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:16px}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:26px;margin:.2em 0}h2{font-size:17px;margin:1.6em 0 .6em}
.muted{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px;margin-bottom:14px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.metric{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.metric .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.metric .v{font-size:24px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
.banner{border-radius:12px;padding:14px 16px;margin-bottom:14px;border:1px solid var(--line)}
.good{border-left:4px solid var(--good)}.ok{border-left:4px solid var(--ok)}
.warn{border-left:4px solid var(--warn)}.bad{border-left:4px solid var(--bad)}
.idle{border-left:4px solid var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.bar{height:7px;background:var(--line);border-radius:99px;overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--accent)}
.chart{width:100%;height:auto}
.axis{fill:var(--muted);font-size:10px}
.charts{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
code{background:var(--line);padding:1px 5px;border-radius:5px;font-size:13px}
@media(max-width:640px){body{padding:16px 12px}.v{font-size:20px}}
"""


def render(d: dict[str, Any]) -> str:
    model = d.get("model") or {}
    live = d.get("live") or {}
    prog = d.get("progress") or {}
    status = d.get("status") or {}
    quota = d.get("quota") or {}
    sessions = d.get("sessions") or {}
    charts = d.get("charts") or {}

    action = status.get("action")
    reason = status.get("reason", "")
    if status.get("finished"):
        klass, title = "good", "Run finished"
    elif action == "dispatched":
        klass, title = "good", "Training dispatched"
    elif action == "sleep":
        klass, title = "warn", "Quotas exhausted — waiting for reset"
    elif action == "none" and "lease held" in reason:
        klass, title = "good", "Training in progress"
    else:
        klass, title = "idle", "Idle"

    tpp = prog.get("tokens_per_param") or 0
    vclass, vtext = _verdict(tpp)

    metrics = [
        ("Step", f"{live.get('step', 0):,}"),
        ("Tokens seen", f"{prog.get('tokens_seen', 0) / 1e9:.3f}B"),
        ("Loss", f"{live['loss']:.4f}" if live.get("loss") else "--"),
        ("Perplexity", f"{live['perplexity']:,.0f}" if live.get("perplexity") else "--"),
        ("Throughput", f"{(live.get('tok_per_s') or 0):,}/s"),
        ("Complete", f"{prog.get('pct_complete', 0):.1f}%"),
    ]
    metric_html = "".join(
        f'<div class="metric"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div></div>' for k, v in metrics
    )

    qrows = ""
    for name, p in quota.items():
        pct = min(p.get("pct_used", 0), 100)
        qrows += (
            f"<tr><td><b>{_esc(name)}</b></td>"
            f"<td>{p['used_hours']:.1f} / {p['quota_hours']:.0f} h</td>"
            f"<td><div class='bar'><i style='width:{pct:.0f}%'></i></div></td>"
            f"<td>{p['remaining_hours']:.1f} h</td>"
            f"<td>{p['hours_to_reset']:.0f} h</td>"
            f"<td>{'yes' if p['available'] else 'no'}</td>"
            f"<td>{'auto' if p.get('automatable') else 'manual'}</td></tr>"
        )

    srows = ""
    for s in (sessions.get("recent") or [])[-10:]:
        eff = s.get("efficiency")
        srows += (
            f"<tr><td>{_esc(s.get('worker_id'))}</td>"
            f"<td>{_esc(s.get('platform'))}</td>"
            f"<td>{_esc(s.get('steps'))}</td>"
            f"<td>{_esc(s.get('end_step'))}</td>"
            f"<td>{f'{eff:.1%}' if eff else '--'}</td></tr>"
        )

    eff_mean = sessions.get("mean_efficiency")
    eff_note = ""
    if eff_mean is not None:
        warn = (" — below 90%, so quota is going to setup, downloads or "
                "validation rather than matrix multiplies"
                if eff_mean < 0.90 else "")
        eff_note = (f"<p class='muted'>Mean session efficiency "
                    f"<b>{eff_mean:.1%}</b> (target 95%+){warn}.</p>")

    bench = d.get("bussbench")
    bench_html = (
        "<p class='muted'>No results yet. <b>Pretraining reports loss and "
        "perplexity, not accuracy</b> — accuracy exists only once BUSSBENCH "
        "runs, which costs GPU time and so happens at milestones.</p>"
        if not bench else
        "<pre>" + _esc(json.dumps(bench, indent=1)) + "</pre>"
    )

    notes = "".join(f"<li>{_esc(n)}</li>" for n in d.get("notes", []))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bussin</title>
<meta http-equiv="refresh" content="180">
<style>{CSS}</style></head><body><div class="wrap">

<h1>Bussin</h1>
<p class="muted"><code>{_esc(model.get('name') or 'bussin')}</code> ·
{model.get('n_params', 0):,} parameters · updated {_esc(d.get('generated_at'))}
· page refreshes every 3 min</p>

<div class="banner {klass}"><b>{_esc(title)}</b><br>
<span class="muted">{_esc(reason)}</span></div>

<div class="grid">{metric_html}</div>

<div class="card {vclass}">
  <h2 style="margin-top:0">Training adequacy</h2>
  <p><b>{_esc(vtext)}</b></p>
  <p class="muted">
    vs Chinchilla ({CHINCHILLA:.0f} tok/param): {prog.get('pct_of_chinchilla', 0):.0f}% ·
    ETA {_esc(prog.get('eta_weeks'))} weeks ·
    weekly capacity {(prog.get('weekly_token_capacity') or 0) / 1e9:.1f}B tokens
  </p>
  <p class="muted">This is the panel that says whether the run is on track.
  Loss alone does not.</p>
</div>

<h2>Platform quota</h2>
<div class="card"><table>
<tr><th>platform</th><th>used</th><th></th><th>remaining</th>
<th>resets in</th><th>available</th><th>mode</th></tr>
{qrows or '<tr><td colspan="7" class="muted">no data</td></tr>'}
</table>
<p class="muted">Quota is <b>estimated</b>: Kaggle does not expose it via API,
so the orchestrator accrues it from dispatch durations against a 90% safety
margin. Reconcile against the settings page occasionally.</p></div>

<h2>Training curves</h2>
<div class="charts">
  <div class="card"><b>loss</b>{_sparkline(charts.get('loss'), colour='#7c5cff')}</div>
  <div class="card"><b>learning rate (WSD)</b>{_sparkline(charts.get('lr'), colour='#2b7fd4')}</div>
  <div class="card"><b>throughput tok/s</b>{_sparkline(charts.get('tok_per_s'), colour='#17916b')}</div>
  <div class="card"><b>gradient norm</b>{_sparkline(charts.get('grad_norm'), colour='#b7791f')}</div>
</div>

<h2>Sessions ({sessions.get('count', 0)})</h2>
<div class="card">{eff_note}<table>
<tr><th>worker</th><th>platform</th><th>steps</th><th>end step</th><th>efficiency</th></tr>
{srows or '<tr><td colspan="5" class="muted">none yet</td></tr>'}
</table></div>

<h2>BUSSBENCH</h2>
<div class="card">{bench_html}</div>

<div class="card"><ul class="muted">{notes}</ul></div>
</div></body></html>"""


def write(d: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(d), encoding="utf-8")
    return p
