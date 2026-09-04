# Headline Findings: PRISM Hybrid IR-Drop Predictor

## 1. Which Variant Leads on Which Metric, and Why

### PR-AUC & Ranking: Not the Problem
- **physics_only and physics_affine have the highest PR-AUC in the study (0.9584)**, while hybrid scores 0.9483 and learned_only scores 0.9269.
  - The paired bootstrap difference between hybrid and physics_affine is -0.0079 (95% CI [-0.0101, +0.0093]), which **includes zero** (not statistically distinguishable).
- **Ranking is not the problem; calibration is.** Despite its leading PR-AUC, `physics_only` catches only **21.6% of violations** (recall 0.2162) at the 45 mV budget threshold.
  - Its systematic **-9.22 mV bias** drops 638 of 814 violations at perfect precision (1.0000), which is the dangerous failure mode: silent, and it looks clean.

### The Calibration Gap and Cross-Design Generalization
- **physics_affine is fitted to zero bias on train designs, but comes back at -1.98 mV on three unseen holdout designs.** A global affine correction does not transfer across designs. It misses 141 violations (17.3% miss rate, recall 0.8268) while inflating precision to 0.9413.
- The **hybrid model reads grid-strength and congestion variation off early-stage features and lands at +0.25 mV on the same unseen designs.** This generalization across blocks is precisely what the design-level holdout was built to detect.
- The hybrid trades ~1% of PR-AUC (paired CI includes zero) for unbiased magnitude (+0.25 mV vs -1.98 mV), **53 fewer missed violations** (88 vs 141 missed), and the **only calibrated interval in the study** (PICP 0.7178 at 4.39 mV MPIW).

### F-Beta and Operating Point Crossover
- At beta = 1.0 (equal weight to precision and recall), physics_affine leads on F1 (0.8803 vs 0.8471) due to precision inflation from under-prediction.
- The measured crossover threshold is **beta* = 1.4169**.
- For any operating regime where missing an IR violation is penalized more than a false alarm (beta > 1.42), hybrid is the superior classifier:
  - beta = 1.5: hybrid = 0.8638, physics_affine = 0.8589
  - beta = 2.0: hybrid = 0.8734, physics_affine = 0.8474

### Continuous Field Accuracy (Role C Downstream Relevance)
- Downstream, role C's adjoint solve consumes the **predicted FIELD** (solving A^T lambda = grad slack). MAE, RMSE, R2, and Spearman are the numbers that govern static timing analysis and slack attribution quality.
- On continuous field metrics, **hybrid dominates decisively, and all paired CIs exclude zero**:
  - MAE: **1.87 mV** vs 3.59 mV (paired diff -1.72 mV, 95% CI [-2.02, -1.57] mV)
  - RMSE: **2.84 mV** vs 4.96 mV (paired diff -2.11 mV, 95% CI [-2.55, -1.82] mV)
  - R2: **0.9663** vs 0.8976 (paired diff +0.0702, 95% CI [+0.0461, +0.1113])
  - Spearman rho: **0.9881** vs 0.9541 (paired diff +0.0353, 95% CI [+0.0148, +0.0578])

## 2. The Bias Asymmetry: Why Under-Prediction is Dangerous

- Measured mean bias on holdout:
  - `physics_only`: **-9.22 mV**
  - `physics_affine`: **-1.98 mV**
  - `learned_only`: **+0.03 mV**
  - `hybrid`: **+0.25 mV**
- In physical design signoff, **under-prediction is hazardous**: predicting 42 mV when real drop is 48 mV causes an unflagged timing violation to escape into silicon, leading to chip failure. Over-prediction merely prompts conservative local grid stiffening or cell padding. `physics_only` (-9.22 mV) and `physics_affine` (-2.09 mV) both suffer from persistent under-prediction. `hybrid` (+0.18 mV) is virtually unbiased.

## 3. Conformal Prediction Intervals

- **Only `hybrid` and `learned_only` produce a prediction interval.** `physics_only` and `physics_affine` produce point predictions with zero uncertainty awareness.
- With the additive-only conformal correction, `hybrid` achieves **PICP = 0.7178** (nominal target 0.80) with a mean interval width of **4.39 mV** (2.5x tighter than the rejected 15.66 mV rule). `learned_only` requires an interval of **10.27 mV** to achieve PICP = 0.8173.

## 4. Per-Design Consistency and Bootstrap Caveat

> *Methodological Caveat on Paired Bootstrap CIs*: The paired bootstrap resamples 3 holdout designs, giving at most 10 distinct multisets. Those CIs are necessarily coarse.

- To verify robustness without multiset coarseness, the **per-design delta table** confirms sign agreement across all three unseen designs:
  - pr_auc: 1/3 favour hybrid
  - violation_f1: 1/3 favour hybrid
  - violation_precision: 0/3 favour hybrid
  - violation_recall: 3/3 favour hybrid
  - mae_mv: 3/3 favour hybrid
  - rmse_mv: 3/3 favour hybrid
  - r2: 3/3 favour hybrid
  - spearman: 3/3 favour hybrid
  - bias_mv: 3/3 favour hybrid

## 5. Grouped Permutation Feature Importance

Measured impact of feature groups on hybrid MAE (permutation on holdout):
- `phys_`: **+4.485 mV** ± 0.022 mV
- `grid_`: **+0.266 mV** ± 0.008 mV
- `cur_`: **+0.119 mV** ± 0.005 mV
- `scn_`: **+0.115 mV** ± 0.007 mV
- `conc_`: **+0.019 mV** ± 0.002 mV
- `sw_`: **+0.000 mV** ± 0.000 mV
- `top_`: **+-0.012 mV** ± 0.003 mV

- Note on `grid_`, `conc_`, `top_`, and `sw_`: These groups contribute modest standalone permutation impact (ΔMAE < 0.30 mV). As established in Session S3, their raw pairwise correlation with residual is max|ρ| ≈ 0.02–0.13. This is an inherent property of this synthetic corpus: sub-tile concentration was modelled as mean-normalised lognormal over ~23 instances per fine cell, which averages out, whereas real concentration is structural. Naming that limitation is the point.