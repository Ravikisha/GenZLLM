# english_retention

Target: 1000 items, metric `accuracy`.

HellaSwag / ARC-e / PIQA / LAMBADA subsets, unmodified

Construction rules: SPEC §14.0. Items must postdate the training
corpus or be authored, and must pass the 13-gram contamination
check in `bussin/eval/bussbench.py` before being committed.
