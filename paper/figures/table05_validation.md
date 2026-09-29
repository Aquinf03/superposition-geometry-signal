# Table 5. Out-of-metric validation

Matched control vs baseline (seed 0). Out-of-metric validation suite.

| Claim | Result | Detail |
| --- | --- | --- |
| downstream loss | **PASS** | control final loss=0.0103 vs baseline=0.0289 (slack=0.05) |
| cross feature baseline | **PASS** | baseline: related=0.0156 < 0.5×unrelated=0.2726 (diff_features_step_00039) |
| cross feature control | **PASS** | control: related=0.0097 < 0.5×unrelated=0.2591 (diff_features_step_00039) |
| cross feature preserved under control | **PASS** | control still separates related vs unrelated neighborhoods |
| held out not gamed | **PASS** | /train−heldout/ control=0.2088 baseline=0.2399 (cap≈0.2899) |
| held out not regressed | **PASS** | held-out L8 interference_mean: control=0.2101 (err=0.1399) vs baseline=0.2160 (err=0.1340); target=0.35; no improvement (within slack) |
| control trajectory checks | **PASS** | validate_training_geometry on control: PASS (6/6 claims) |

**7/7 claims pass.**
