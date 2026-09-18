# conversation

Target: 150 items, metric `human_1_5`.

5-turn dialogues, human 1-5 plus LLM judge

Construction rules: SPEC §14.0. Items must postdate the training
corpus or be authored, and must pass the 13-gram contamination
check in `bussin/eval/bussbench.py` before being committed.
