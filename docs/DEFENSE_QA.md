# Defense Q&A

Questions about the choices we made while running the project, with short answers and where the evidence is. D-numbers refer to [DECISIONS_LOG.md](DECISIONS_LOG.md).

General questions about the data and design are answered in [LIMITATIONS_AND_DEFENSE.md](LIMITATIONS_AND_DEFENSE.md) and not repeated here: synthetic data, the random split, dropped dates, why R² is low, not merging real data. The evidence files for those answers: `02_date_gaps.csv`, `02_lag_autocorrelation.csv`, `02_city_means.csv`, `02_target_correlations.csv`.

## About the method

**Q: Did the test set influence any choice?**
No:
- No tuning or baseline script loads the test rows; each prints "test set not loaded".
- The Step 2 audit does describe all 88,489 rows, test rows included. That is descriptive only (distributions, correlations, dates); no threshold, feature or setting was chosen from it. If asked, we can rerun the audit on training rows only; with 70k rows the numbers barely change.
- The spike threshold (13) comes from training rows only (D4).
- Scalers and encoders are fitted inside pipelines on training folds.
- The test set is used once, in Step 9.

**Q: Did you tune until you got the result you wanted?**
No. The guards are all on record:
1. Selection metrics and the tie rule were fixed in advance (D12).
2. Every rule change was made before the FULL results existed. The first FULL SVM run was stopped before it wrote anything (D9).
3. Every range widening is recorded in the tables (`edge_note`).
4. Where a rule was a made-up number, it was replaced with a principled one: fold-to-fold SD instead of ad-hoc cutoffs (D8).
5. The result we report is a tie, i.e. no model improved. That is the opposite of a tuned-to-win result.

**Q: Why train the SVM on only 20,000 rows?**
- Kernel SVM training time grows faster than the square of the row count (measured n^1.9–2.1).
- The learning curve shows no gain from 5k to 40k rows: at most 0.004 RMSE, against a fold-to-fold SD of 0.017.
- More data wouldn't change the result, so the cap is justified by evidence, not just by time (D7, D10). Evidence: `07_svm_learning_curve.csv` and the two learning-curve figures.

**Q: The SVM's CV PR-AUC is 0.251, but you report 0.239. Why the lower number?**
- 0.251 is the best of about 60 candidates on the same folds, so it is inflated by selection (Cawley & Talbot 2010), and it was measured on a different sample.
- 0.239 is the same settings held fixed and scored on the full-training folds, which is comparable with the other models.
- **A mathematical check:** with PM2.5 as the only feature, every model whose score only rises with PM2.5 has the same PR-AUC. Logistic gets 0.240, so 0.251 can't be real (D10).
- Step 9 gives the unbiased test-set number.

**Q: The SVR's epsilon is 3.4. Isn't that suspiciously large?**
- It was chosen by CV on training rows, within a range fixed before the run. The top of that range is one noise SD, following Cherkassky & Ma 2004 (D8).
- CV RMSE is almost flat across epsilon (best 3.398, median 3.403 in Phase A, FS-PM25), so epsilon doesn't drive any result.
- A large epsilon means the SVR ignores errors inside a ±3.4 band, which is reasonable when the noise SD is about 3.4.
- The literature reference value (≈ 0.25) is logged for transparency.

**Q: Why random search for the SVM but a full grid for the tree?**
- An SVM fit takes seconds to minutes, so random search (Bergstra & Bengio 2012) gets the most coverage for the budget.
- A tree fit takes about a second, so a full grid is affordable and removes any sampling luck.

**Q: Why 5-fold CV and not repeated or nested CV?**
- 5-fold CV is standard for tuning.
- Nested CV would give unbiased tuning scores but costs 5× more. Our single, untouched test set (Step 9) serves the same purpose.
- Repeated CV (Step 8) was cut to keep the scope lean (D2).

**Q: Is the 0.05 RMSE tie rule arbitrary?**
It is a convention, fixed before any results. The observed differences between models are about 0.01. That is smaller than the fold-to-fold SD (about 0.02–0.04), so they'd be ties under any reasonable rule. Step 9 adds bootstrap CIs and paired differences for the formal test.

## About the results

**Q: What did the final test show?**
- Every regression model tied with the PM2.5-only line: test RMSE 3.409–3.425 against 3.419, R² about 0.15, no leakage alarm.
- PM2.5 alone matched all features for every model, so H1 is supported.
- Spike ranking reached PR-AUC 0.249 against 0.117 for prevalence.

The test set was used once, after every setting was fixed (D14). Tables: `09_test_*.csv`.

**Q: Do cities differ? Should each city get its own model?**
No, in practice:
- **Per-city models never beat the pooled model** on the test set, even though each was tuned on its own city's data (Table 4).
- **The slopes differ a little** (0.090–0.106; Wald χ²(7) = 15.7, p = 0.028, unadjusted). That's detectable with 70k rows but too small to matter, and partly an artefact of fitting a line to a staircase (D15, D17).

**Q: Is the spike early warning useful?**
- It is clearly better than chance: PR-AUC 0.25 vs 0.12, and F1 0.31 vs 0.21 for flagging every day.
- But it's weak: catching about half the spike days means flagging about 28% of all days, and only about 1 in 4.5 flagged days is a real spike.
- The limit is the data: one noisy signal (R² ≈ 0.15). No model can do better here.

**Q: All models tie. Did the complex models fail?**
No. When the signal is linear with r ≈ 0.39, the maximum reachable R² is about 0.154. A flexible model can only match the straight line; its extra flexibility can only fit noise, which cross-validation penalises. The SVM confirms this: CV chose a small gamma, which makes the kernel close to linear. Getting a tie is the correct outcome here. Beating the ceiling would suggest leakage.

**Q: So what's the take-away?**
1. Check the data first. The audit found the key facts (one signal, no time structure, synthetic) before any modelling.
2. Always compare against a simple baseline. Here nothing beats PM2.5 alone (H1).
3. Report fair, bias-free scores. Tuning scores overstate performance.
4. Model complexity should match the signal.

## Open weaknesses (state them before someone else does)
| Weakness | Status | Plan |
|---|---|---|
| alpha = 0 with the full one-hot city columns is collinear (dummy variable trap) | Checked: RMSE is flat across alpha from 0 to 1000, so predictions are unaffected | None needed; explained in D11 |
| The tree classifier's PR-AUC (0.227–0.232) is below Logistic's (0.240) | Explained by theory: steps vs a smooth ranking. The leaf-size range was widened properly. | Step 9 confirms it on the test set |
| MLP classifier can't use validation early stopping in sklearn (it watches accuracy) | Stops on training log-loss with CV-tuned L2 | State in the report (MODEL_JUSTIFICATION M4) |
| Split strata used a cutoff computed on all rows | Harmless (allocation only) | Mention in limitations |
| Tuning scores (all models, not only the SVM) carry selection bias | **Resolved by Step 9:** the test results (one pass, paired bootstrap CIs) confirm the CV picture: every regression model ties | Report test numbers, not tuning scores |
| The tree is detectably better on test RMSE (−0.011, CI excludes 0) | Real (the same gap appeared in CV) but 5× below the pre-set 0.05 margin | Call it a tie by the pre-registered rule, and say openly that it's statistically detectable but practically negligible |
| PR-AUC has no pre-set tie margin, so tiny gaps (e.g. MLP −0.0018) are labelled "worse" | No margin was added after seeing results, since that would be result-driven | Report effect sizes next to verdicts |
| The SVC flags 46% of test days, against about 28% for the other models | Its threshold is on the raw decision-function scale, set on 16k-row CV models and applied to a 20k-row model; no Platt scaling (compute rule) | State as a limitation. Its PR-AUC (threshold-free) equals Logistic's |
| The city-slope test is p = 0.028, unadjusted | Holm dropped; small effect (0.090–0.106); may partly come from fitting a line to the staircase | Report as weak, small heterogeneity that doesn't help prediction (per-city models don't beat pooled) |
| Slides | Deferred by the user | Make later from REPORT.md |
| Findings apply to this synthetic benchmark only | By design | First-page disclosure (METHODOLOGY Step 14) |
