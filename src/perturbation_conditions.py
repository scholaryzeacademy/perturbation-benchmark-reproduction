"""Shared helper for GEARS/PertData's perturbation 'condition' string format
(BUILD_PLAN.md Sec.6): 'ctrl' for control cells, 'GENE+ctrl' for a
single-gene perturbation, 'GENE1+GENE2' for a combinatorial one -- verified
directly against gears/pertdata.py's own condition-parsing logic. Used by
both the Stage 2 ridge baseline's gene encoding and the Stage 3a GEARS
wrapper's condition<->gene-list conversion, so the two never drift out of
sync.
"""


def perturbed_genes(condition: str) -> list[str]:
    return [g for g in condition.split("+") if g != "ctrl"]
