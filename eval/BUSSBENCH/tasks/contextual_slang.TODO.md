# contextual_slang

Target: 300 items, metric `accuracy`.

polysemous terms in 2+ contexts; needs authored pairs

Construction rules: SPEC §14.0. Items must postdate the training
corpus or be authored, and must pass the 13-gram contamination
check in `bussin/eval/bussbench.py` before being committed.
