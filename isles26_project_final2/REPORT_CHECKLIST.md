# Final report checklist (approximately eight pages)

Do not fill result placeholders until the corresponding generated artifact has
been checked. The report must be the authors' own writing and must cite every
external code or scientific source used.

## Front matter and abstract

- Title, both authors, IDs, and email addresses.
- Abstract of at most 300 words: problem, method, dataset, exact primary result,
  uncertainty, and conclusion.

## 1. Introduction (10%)

- Clinical/technical importance of chronic stroke-lesion segmentation.
- Why small lesions are difficult and why case sampling may help.
- Research question, hypothesis, contributions, and one-sentence results
  summary.

## 2. Related work (10%)

- ATLAS dataset/challenge and published stroke-lesion segmentation methods.
- nnU-Net and imbalance approaches: Focal, Tversky, patch sampling, and
  case-level sampling.
- State precisely how fixed and curriculum sampling differ from prior work.

## 3. Data (5%)

- ATLAS release/version, access conditions, MRI modality, labels, centers, and
  usable sample count.
- Exclusion/discovery rules and preprocessing.
- Lesion-volume distribution and definitions of small/medium/large bins.
- Five-fold subject grouping and center/size balancing. Include a fold table.

## 4. Methods (25%)

- nnU-Net configuration and all settings held constant.
- Equations for Dice, Focal loss, Tversky loss, fixed sampling
  `p_i ∝ max(V_i, 1 mm³)^-1`, and the three-phase curriculum.
- Explain that ground-truth volume is used only to sample training cases, not at
  inference time.
- Report sampling minimum/maximum probabilities and effective sample size from
  training logs.
- Define raw and penalized HD95 and justify the empty-surface policy.
- Cite all external implementations and identify all code written for this
  project.

## 5. Experiments and results (20%)

- Hardware, software versions, run time, folds, seed, checkpoints, and any
  deviations from `EXPERIMENT_PROTOCOL.md`.
- Overall table: mean, standard deviation, median, count, failure rate.
- Paired baseline comparisons with 95% confidence intervals and adjusted
  p-values.
- Lesion-size and center plots with sample counts.
- Training curves, qualitative examples, and common failure modes.
- Discuss effect sizes and uncertainty—not only which number is largest.

## 6. Conclusion (5%)

- Answer the research question in proportion to the evidence.
- State limitations: dataset scope, subgroup sizes, compute, and external
  generalization.
- Give concrete future work, such as capped/tempered sampling or external-test
  validation.

## Writing and formatting (5%)

- Approximately eight pages, English, consistent terminology, numbered figures
  and tables, readable captions, and no unsupported claims.
- Define every abbreviation and use “ATLAS” consistently; `isles26.py` is only
  the historical runner filename.

## Appendix/code (20%)

- Link the complete code archive/repository and include usage instructions.
- Include run manifests, split definition, trainer code, metric definitions,
  and all citations listed in `README.md`.
- Do not include patient data, raw dataset files, credentials, or generated
  environment files.

