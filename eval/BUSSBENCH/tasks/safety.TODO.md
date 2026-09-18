# safety

Target: 300 items, metric `toxicity_rate`.

RealToxicityPrompts subset plus slur-elicitation probes

Construction rules: SPEC §14.0. Items must postdate the training
corpus or be authored, and must pass the 13-gram contamination
check in `bussin/eval/bussbench.py` before being committed.
