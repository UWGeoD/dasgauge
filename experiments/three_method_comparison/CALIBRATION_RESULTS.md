# C2ST calibration and power study

Rates labeled `type_i_calibration` estimate Type-I error for a method under its
own target null and assumptions. Other null rows are deliberately outside those
assumptions or, for the swap method in regimes D/E, outside `H0_swap`.

## Key findings

- Under the independent i.i.d. null, all three headline references were valid
  and rejection rates ranged from 0.022 to 0.044.
- Under strongly dependent but exchangeable pairs, valid McNemar and swap rates
  ranged from 0.020 to 0.041. The pair-split Binomial diagnostic rejected at
  rates 0.131/0.138, showing anti-conservative individual-correctness
  calibration.
- The cyclic equal-marginal construction violated the swap null but the chosen
  accuracy statistic rejected at rate 0.000; this is lack of sensitivity, not
  validation of swap invariance.
- With record-wide dependence, all headline references were outside their
  stated assumptions and rejection rates were 0.981--1.000.
- Under the mean-shift alternative, headline power ranged from 0.084 to 0.200
  across the two sample sizes.

## Complete results

| regime | n pairs | method | interpretation | reject rate | MCSE | 95% Wilson CI |
|---|---:|---|---|---:|---:|---:|
| D. Equal marginals, nonexchangeable i.i.d. pairs | 40 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 40 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 40 | `paired_mcnemar` | type_i_calibration | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 40 | `paired_swap_permutation` | sensitivity_to_swap_null_violation_under_H0_dist | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 80 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 80 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 80 | `paired_mcnemar` | type_i_calibration | 0.000 | 0.000 | [0.000, 0.004] |
| D. Equal marginals, nonexchangeable i.i.d. pairs | 80 | `paired_swap_permutation` | sensitivity_to_swap_null_violation_under_H0_dist | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 40 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 40 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 40 | `paired_mcnemar` | type_i_calibration | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 40 | `paired_swap_permutation` | type_i_calibration | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 80 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 80 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 80 | `paired_mcnemar` | type_i_calibration | 0.000 | 0.000 | [0.000, 0.004] |
| C. Exact-copy pairs | 80 | `paired_swap_permutation` | type_i_calibration | 0.000 | 0.000 | [0.000, 0.004] |
| A. Independent i.i.d. null | 40 | `original_binomial_c2st` | type_i_calibration | 0.038 | 0.006 | [0.028, 0.052] |
| A. Independent i.i.d. null | 40 | `pair_preserving_binomial` | type_i_calibration | 0.046 | 0.007 | [0.035, 0.061] |
| A. Independent i.i.d. null | 40 | `paired_mcnemar` | type_i_calibration | 0.027 | 0.005 | [0.019, 0.039] |
| A. Independent i.i.d. null | 40 | `paired_swap_permutation` | type_i_calibration | 0.039 | 0.006 | [0.029, 0.053] |
| A. Independent i.i.d. null | 80 | `original_binomial_c2st` | type_i_calibration | 0.044 | 0.006 | [0.033, 0.059] |
| A. Independent i.i.d. null | 80 | `pair_preserving_binomial` | type_i_calibration | 0.035 | 0.006 | [0.025, 0.048] |
| A. Independent i.i.d. null | 80 | `paired_mcnemar` | type_i_calibration | 0.022 | 0.005 | [0.015, 0.033] |
| A. Independent i.i.d. null | 80 | `paired_swap_permutation` | type_i_calibration | 0.027 | 0.005 | [0.019, 0.039] |
| F. Mean-shift alternative | 40 | `original_binomial_c2st` | power | 0.118 | 0.010 | [0.099, 0.139] |
| F. Mean-shift alternative | 40 | `pair_preserving_binomial` | power | 0.137 | 0.011 | [0.117, 0.160] |
| F. Mean-shift alternative | 40 | `paired_mcnemar` | power | 0.084 | 0.009 | [0.068, 0.103] |
| F. Mean-shift alternative | 40 | `paired_swap_permutation` | power | 0.130 | 0.011 | [0.111, 0.152] |
| F. Mean-shift alternative | 80 | `original_binomial_c2st` | power | 0.200 | 0.013 | [0.176, 0.226] |
| F. Mean-shift alternative | 80 | `pair_preserving_binomial` | power | 0.192 | 0.012 | [0.169, 0.218] |
| F. Mean-shift alternative | 80 | `paired_mcnemar` | power | 0.150 | 0.011 | [0.129, 0.173] |
| F. Mean-shift alternative | 80 | `paired_swap_permutation` | power | 0.172 | 0.012 | [0.150, 0.197] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 40 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.002 | 0.001 | [0.001, 0.007] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 40 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.131 | 0.011 | [0.111, 0.153] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 40 | `paired_mcnemar` | type_i_calibration | 0.020 | 0.004 | [0.013, 0.031] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 40 | `paired_swap_permutation` | type_i_calibration | 0.031 | 0.005 | [0.022, 0.044] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 80 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.000 | 0.000 | [0.000, 0.004] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 80 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.138 | 0.011 | [0.118, 0.161] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 80 | `paired_mcnemar` | type_i_calibration | 0.041 | 0.006 | [0.030, 0.055] |
| B. I.i.d. exchangeable pairs with strong negative dependence | 80 | `paired_swap_permutation` | type_i_calibration | 0.039 | 0.006 | [0.029, 0.053] |
| E. Across-pair dependence | 40 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 0.989 | 0.003 | [0.980, 0.994] |
| E. Across-pair dependence | 40 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 0.993 | 0.003 | [0.986, 0.997] |
| E. Across-pair dependence | 40 | `paired_mcnemar` | null_behavior_outside_method_assumptions | 0.981 | 0.004 | [0.971, 0.988] |
| E. Across-pair dependence | 40 | `paired_swap_permutation` | sensitivity_to_swap_null_violation_under_H0_dist | 0.989 | 0.003 | [0.980, 0.994] |
| E. Across-pair dependence | 80 | `original_binomial_c2st` | null_behavior_outside_method_assumptions | 1.000 | 0.000 | [0.996, 1.000] |
| E. Across-pair dependence | 80 | `pair_preserving_binomial` | null_behavior_outside_method_assumptions | 1.000 | 0.000 | [0.996, 1.000] |
| E. Across-pair dependence | 80 | `paired_mcnemar` | null_behavior_outside_method_assumptions | 1.000 | 0.000 | [0.996, 1.000] |
| E. Across-pair dependence | 80 | `paired_swap_permutation` | sensitivity_to_swap_null_violation_under_H0_dist | 1.000 | 0.000 | [0.996, 1.000] |
