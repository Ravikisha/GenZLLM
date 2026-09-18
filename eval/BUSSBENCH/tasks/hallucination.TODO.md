# hallucination

Target: 200 items, metric `accuracy+abstention`.

factual QA with an explicit 'I don't know' option

Construction rules: SPEC §14.0. Items must postdate the training
corpus or be authored, and must pass the 13-gram contamination
check in `bussin/eval/bussbench.py` before being committed.
