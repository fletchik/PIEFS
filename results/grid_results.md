# PIEFS — Preliminary Baseline Results

> **Status:** Pre-bugfix baseline (original ratio_gram bug, no GlobalLowRank).
> Full results (D0–D5 ablations) will be added after experiment runs complete.
> See `EXPERIMENT_PLAN.md` for the run schedule.

Accuracy % (mean ± std, 5 seeds, test split)

| Dataset   | off             | diag            | lambda_u_trotter |
|-----------|-----------------|-----------------|-----------------|
| two_moon  | 100.00 ± 0.00%  | 99.97 ± 0.03%   | 99.91 ± 0.07%   |
| circles   | 78.23 ± 13.33%  | 79.16 ± 4.31%   | 73.65 ± 6.65%   |
| htru2     | 97.52 ± 0.07%   | 97.48 ± 0.04%   | 97.56 ± 0.08%   |
| mnist_mc  | 94.63 ± 0.45%   | 93.85 ± 0.15%   | 94.00 ± 0.33%   |

**D0 result (post-fix, 3 seeds):** global_low_rank r=1 on htru2 → **97.77%**
vs off 97.36% (+0.41 pp), confirming the ratio_gram² fix works as intended.
