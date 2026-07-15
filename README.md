Implementations for Paper:
Interpretable and Fair Mechanisms for Abstaining Classifiers

## Note: rejection-rule count is dataset-size sensitive

`IFAC`'s rejection-rule filter (`SelectiveClassifier_IFAC.py`) requires either a near-total confidence
swing between a protected group and its complement, or `abs(slift) >= min_slift`, on top of a
significance check. On small datasets, narrow rule bases cover so few people that group confidence
collapses to ~0/1 by chance, so the swing-based condition fires easily whether or not the effect is real.
On larger, more stable datasets, real disparities tend to show up as smaller (but still significant) slift
values, so that condition alone can yield **zero** rejection rules even where a clear disparity exists.

Two parameters address this, both defaulting to the original hardcoded values (no behavior change unless
set explicitly):

- `IFAC(..., min_slift=<float>)` — lets a smaller-but-significant slift qualify a rule (the parameter was
  previously accepted but silently unused).
- `extract_class_association_rules_in_predictions(..., min_confidence=<float>)` — the apriori confidence
  needed to mine a candidate rule per subgroup in the first place. Defaults to 0.85; if the outcome you
  care about (e.g. a "denied"/negative label) stays well under that prevalence even in the most
  disadvantaged subgroup, apriori won't mine a candidate rule for it at all, and `min_slift` has nothing to
  work with downstream. Lowering it multiplies apriori's rule count and runtime per subgroup.

Before trusting either knob, sanity-check the raw outcome rate per protected group directly from the
model's own predictions (not ground truth) against the reference group, independent of apriori/IFAC
entirely — that tells you whether zero rejection rules means "the model is fair" or "the gate missed it."

Please cite as

`
@inproceedings{lenders2024interpretable,
  title={Interpretable and fair mechanisms for abstaining classifiers},
  author={Lenders, Daphne and Pugnana, Andrea and Pellungrini, Roberto and Calders, Toon and Pedreschi, Dino and Giannotti, Fosca},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={416--433},
  year={2024},
  organization={Springer}
}
`