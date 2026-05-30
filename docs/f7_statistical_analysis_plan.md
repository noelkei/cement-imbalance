# F7 Statistical Analysis Plan

## Purpose

This document fixes the statistical analysis scope for the final F7 comparison stage.

The goal is not only to rank variants by a single average metric, but to extract maximum signal about:

- predictive quality under temporal shift;
- robustness across seeds;
- class-wise behavior and Simpson-style failure modes;
- marginal effects and interactions of methodological axes;
- interpretability stability and driver consistency;
- practical production relevance under the real distribution we care about.

This document is intentionally broader than a leaderboard. It defines what we want to learn from the final F7 panel and what data structures, metrics, uncertainty summaries, and plots we need to support that analysis rigorously.

## Statistical Scope and Limits

### 1. The panel must not be treated as i.i.d.

The final F7 panel is not an i.i.d. sample of unrelated runs.

Runs share:

- the same official temporal split contract;
- common upstream datasets and preprocessing logic;
- common factor structure;
- repeated seeds across structurally comparable groups;
- repeated methodological components across families and policies.

Therefore:

- we should not interpret the per-run panel as a set of fully independent observations;
- paired and grouped analyses are preferred whenever design allows them;
- structural grouping must be explicit in both estimation and uncertainty quantification.

### 2. Seed uncertainty is not full population uncertainty

The variability induced by multiple seeds is valuable, but it measures only part of the uncertainty relevant to the project.

Seed variation captures mainly:

- optimization randomness;
- initialization sensitivity;
- training stochasticity;
- sensitivity to synthetic augmentation randomness when applicable.

It does **not** by itself fully capture:

- uncertainty over alternative temporal split realizations;
- uncertainty over alternative data collection periods;
- population-level uncertainty in a broader industrial deployment sense.

So:

- seed-based intervals should be interpreted primarily as **algorithmic robustness / stability** summaries;
- they should not be overclaimed as complete uncertainty statements about future deployment performance.

### 3. Temporal robustness and algorithmic robustness are distinct

This project has one canonical official temporal split.

Therefore:

- temporal robustness is assessed through behavior across `train`, `val`, and `test` under that official split;
- seed robustness is assessed through repeated runs within that split;
- the analysis does not estimate uncertainty over multiple alternative temporal partitions unless such a layer is explicitly added later.

This distinction should be explicit in the final report.

## Core Position

### 1. The analysis is not only about improving macro performance

Synthetic data were not introduced only to improve a macro average.

One of the key methodological motivations is the Simpson-type failure mode observed in this project:

- a model can appear strong in aggregate;
- `R²` or related global metrics can look inflated;
- but class-wise performance can still be poor;
- the model can effectively interpolate across the three cement classes instead of learning class-specific behavior properly.

Therefore, the final analysis must always operate on **two parallel axes**:

- aggregate behavior;
- class-wise behavior.

Neither axis is sufficient on its own.

### 2. Final reporting should treat `val` and `test` as co-primary descriptive endpoints after freeze

For methodological hygiene:

- variant design and iterative decisions happen before the final frozen analysis;
- we do not keep tuning on `test`.

However, once the final analysis panel is frozen, we want the final statistical readout to emphasize both:

- `val`, because it remains the canonical model-selection surface;
- `test`, because in this project it is operationally closer to the distribution we ultimately care about, and closer to `train` than `val` is.

So the final statistical analysis should not present `val` as the only primary endpoint and `test` as a footnote.

Instead:

- `val` and `test` are treated as **co-primary final descriptive endpoints** in the final frozen statistical report;
- `train` remains diagnostic, not a selection target.

This is not a license to optimize on `test`. It is a reporting stance applied **after freeze**.

### 3. Selection and final characterization are separate roles

The final methodology should maintain a clean distinction between:

- **selection**:
  - shortlist construction, methodological narrowing, and any pre-freeze decision logic;
- **final statistical characterization**:
  - the frozen final descriptive and inferential readout across the locked panel.

The intended rule is:

- `val` remains the canonical surface for shortlist construction and pre-freeze methodological choices;
- once the panel is frozen, both `val` and `test` are emphasized in the final statistical characterization;
- `test` is not reintroduced as a decision loop after freeze.

This distinction should remain explicit throughout the final analysis artifacts.

## Main Questions

The final analysis should answer these questions.

### Q1. Which methodological axes improve predictive performance overall?

We want the marginal contribution of:

- model family;
- `X` transform;
- `Y` transform;
- synthetic policy;
- training/run policy;
- FlowPre usage;
- FlowGen usage.

### Q2. Which axes improve class-wise performance rather than only aggregate performance?

This is central because of the Simpson-type risk.

We want to know:

- whether a variant improves all classes or only aggregate mixing;
- whether it reduces cross-class averaging behavior;
- whether it improves the minority or harder classes rather than only the easiest class.

### Q3. Which combinations show true synergy or antagonism?

We do not only want marginal effects.

We want to estimate:

- whether certain synthetic policies work better only under certain `X` transforms;
- whether FlowPre helps only with certain run policies;
- whether `MLP` and `XGBoost` respond differently to the same synthetic strategy.

### Q4. Which variants are robust across seeds?

We want:

- low variance between seeds;
- stable ranking;
- limited deterioration in the worst cases.

### Q5. Which variants generalize well across splits?

We want split-gap analysis:

- `val - train`
- `test - train`
- `test - val`

This should be done for both macro and class-wise metrics.

### Q6. What are the main drivers, and how stable are they?

We want interpretability that answers:

- which features consistently drive the prediction;
- whether those drivers are stable across seeds;
- whether synthetic policies or FlowPre change the driver structure;
- whether the same semantic drivers appear across model families.

## Estimands

To keep the final analysis disciplined, the document should not stop at “what we want to inspect”. We should state the quantities we want to estimate.

The main estimands are:

- average marginal effect of each methodological factor on:
  - `val_raw_real_macro_rrmse`
  - `test_raw_real_macro_rrmse`
- average marginal effect of each methodological factor on:
  - per-class `rrmse`
  - worst-class `rrmse`
  - Simpson-risk indicators
- average paired contrast between selected methodological alternatives under matched structural contexts
- average split-gap quantities:
  - `val_minus_train`
  - `test_minus_train`
  - `test_minus_val`
- average driver importance and driver stability across seeds on the canonical semantic interpretability surface

The purpose of stating estimands explicitly is:

- to prevent drifting into a large collection of loosely connected summaries;
- to keep the inference aligned with the questions that matter for the thesis close;
- to make later implementation decisions reproducible and auditable.

## Endpoints

## Predictive Endpoints

### Co-primary endpoints

The final analysis should treat these as co-primary descriptive endpoints:

- `val_raw_real_macro_rrmse`
- `test_raw_real_macro_rrmse`

These are the top-level endpoints for the final report.

These co-primary endpoints should be treated as:

- central summary endpoints for the frozen final report;
- not as permission to optimize on `test`.

### Parallel class-wise co-primary endpoints

To address Simpson-style behavior, the final analysis must also elevate class-wise endpoints, not only include them as appendix material.

At minimum:

- `val_raw_real_per_class_rrmse`
- `test_raw_real_per_class_rrmse`
- class macro-vs-per-class discrepancy summaries
- worst-class summaries

### Secondary predictive endpoints

These should also be retained in both `val` and `test`, at least at macro and per-class level where available:

- `rmse`
- `mse`
- `r2`
- `mape`

`R²` should remain an auxiliary endpoint, not the main carrier of the Simpson-related narrative.

In this project, the Simpson-style failure mode is better interrogated using error metrics in raw space, especially:

- `rrmse`
- `rmse`
- class-wise errors
- worst-class errors
- class-dispersion indicators

### Diagnostic endpoints

These are not ranking targets but are analytically important:

- `train_raw_real_macro_rrmse`
- `train_raw_real_per_class_rrmse`
- `val - train` gap
- `test - train` gap
- `test - val` gap

## Performance Views Required

Every main analysis slice should be available in all of the following views where possible:

- overall
- macro
- per_class
- worst_class
- overall quantiles
- per_class quantiles

This is especially important because a variant can:

- improve macro;
- worsen the worst class;
- or improve the center of the distribution while worsening the tails.

## Simpson-Focused Analysis Layer

This needs to be explicit and not buried inside the general performance chapter.

We want a dedicated analysis layer for class mixing / Simpson-style failure.

### Hypothesis

Some models appear good in aggregate because they predict a global trend across all three classes, rather than learning the conditional behavior of each class well.

This hypothesis should be evaluated not only with pooled vs class-wise metrics, but also by separating:

- between-class structure capture;
- within-class predictive quality.

### What to measure

At minimum:

- difference between overall and macro metrics;
- difference between overall and per-class summaries;
- worst-class error;
- class spread:
  - `max(class metric) - min(class metric)` for error metrics;
- class bias patterns:
  - whether one class is systematically underfit or overfit;
- per-class residual distribution summaries.

### Useful derived indicators

We should define explicit Simpson-risk indicators such as:

- `simpson_gap_rrmse = overall_rrmse - macro_rrmse`
- `class_dispersion_rrmse = max(per_class_rrmse) - min(per_class_rrmse)`
- `worst_class_penalty_rrmse = worst_class_rrmse - macro_rrmse`

These do not replace the raw metrics, but they make the phenomenon easy to compare across variants.

### Within-class / between-class decomposition

To strengthen the Simpson-focused layer, the final analysis should include a decomposition-oriented view whenever feasible.

The idea is to separate:

- performance that comes from fitting class offsets or between-class structure;
- performance that comes from modeling within-class behavior well.

Useful ways to operationalize this include:

- class-centered residual analysis;
- within-class centered target analysis;
- separate summaries of within-class error vs pooled error;
- comparisons between pooled and class-conditioned performance views.

This can add substantial signal because it directly addresses whether a model is genuinely learning conditional structure or merely exploiting aggregate class separation.

### Main question

Does a variant:

- genuinely improve each class;
- or only improve the pooled view?

Synthetic policies should be evaluated directly against this question.

## Statistical Summaries Per Group

For each structural group across seeds, compute:

- `n_seeds`
- mean
- std
- stderr
- median
- `p25`
- `p75`
- min
- max
- `95%` confidence interval for the mean

This should be done for:

- macro metrics;
- per-class metrics;
- split gaps;
- Simpson-risk derived indicators.

In addition to descriptive summaries, it is useful to define a notion of **practical relevance**:

- very small deltas should not automatically be treated as meaningful;
- statistical significance should not be allowed to dominate practical interpretation.

Where appropriate, the final report should introduce minimum practically relevant effect thresholds for the main endpoints.

## Confidence Intervals and Uncertainty

### Default summaries

For every aggregated performance endpoint:

- mean with `95%` CI;
- median with IQR;
- seed-level spread.

### Recommended interval strategy

Prefer bootstrap-based intervals over naive normal approximations when possible.

Recommended:

- bootstrap over seeds for aggregated structural groups;
- paired bootstrap for head-to-head comparisons between variants.

Bootstrap configuration should later be frozen explicitly, including:

- number of resamples;
- bootstrap seed;
- interval type:
  - percentile or BCa;
- resampling unit:
  - seed
  - or structurally paired unit when appropriate.

### What uncertainty should answer

Not only “what is the mean?” but:

- how stable is the mean across seeds?
- how uncertain is the difference between A and B?
- how often does A beat B?

Where model-based inference is used, we should also prefer:

- heteroscedasticity-robust standard errors;
- or cluster/group-aware uncertainty strategies when structurally appropriate.

## Method Hierarchy and Preferred Use

The statistical plan should not treat all methods as equally central.

For this project, the strongest and cleanest analysis is obtained by giving priority to methods that respect the repeated, structured, and partially paired nature of the panel.

### Primary inferential backbone

The primary inferential backbone should be:

- paired or blocked comparisons whenever the design allows them;
- seed-aware aggregation;
- structural-group-aware uncertainty;
- explicit split-aware and class-aware reporting.

In practice, this means that the first inferential question should usually be:

- within matched structural contexts, what changes when one methodological axis changes?

rather than:

- across the entire pooled panel, is there a global average difference?

This is the most faithful way to exploit the design actually produced by the campaign system.

### Preferred method stack

The recommended method stack, in order of importance, is:

1. descriptive multi-seed summaries with interval estimates;
2. paired / blocked comparison layer for head-to-head questions;
3. mixed-effects or repeated-structure regression layer for factor effects and interactions;
4. ANOVA-style decomposition as a support layer on fitted models;
5. selective post-hoc contrasts for justified questions;
6. sensitivity checks using non-parametric or permutation-style alternatives where useful.

This ordering is intentional. It reflects the fact that the design has strong structural pairing information that should not be discarded in favor of a single omnibus model.

### Why paired and blocked analysis comes first

Many of the most important scientific questions in this project are naturally paired:

- same structural group, different synthetic policy;
- same structural group, different `x_transform`;
- same family and data candidate, different run policy.

When those comparisons are available, paired or blocked analysis should be preferred because it:

- reduces nuisance variation;
- respects the design more directly;
- produces clearer deltas;
- yields stronger evidence for practical methodological questions.

This should be considered the default for high-priority contrasts.

### Why mixed-effects models should be preferred over standalone ANOVA

Where a model-based global layer is needed, mixed-effects style models are generally preferable to treating the panel as a classical independent-samples ANOVA table.

The reasons are:

- runs are not i.i.d.;
- seeds provide repeated algorithmic replications;
- structural groups create natural clustering;
- some effects and comparisons are better interpreted conditionally than marginally.

Accordingly, the recommended role of model-based analysis is:

- estimate factor effects and interactions on the per-run panel;
- account for blocking or random-effect structure where appropriate;
- provide a compact global synthesis of the design.

ANOVA remains useful, but primarily as a decomposition and support layer on top of that fitted structure.

### Recommended uncertainty strategy by task

Different questions justify different uncertainty tools.

Preferred defaults:

- aggregated group summaries:
  - bootstrap over seeds or over the smallest defensible repeated unit;
- paired contrasts:
  - paired bootstrap as default;
  - paired permutation test as high-value support when a randomization-style check is useful;
- regression-style factor models:
  - robust standard errors by default;
  - cluster-aware uncertainty when dependence structure is strong.

The analysis should avoid presenting a single uncertainty mechanism as if it were universally optimal for every layer.

### Simpson-focused methods should be treated as first-class, not supplemental

Because Simpson-type failure is a core methodological concern of the project, methods that isolate within-class behavior should not be treated as optional appendices.

The primary quality layer should therefore include:

- macro endpoints;
- per-class endpoints;
- worst-class summaries;
- class-dispersion indicators;
- within-class / between-class decomposition whenever feasible.

This is important because a method can look competitive on aggregate while remaining poor in the sense that motivated the synthetic-policy investigation in the first place.

### Non-parametric and permutation methods as sensitivity tools

Non-parametric and permutation-style methods are useful here, but mainly as sensitivity or robustness checks rather than the sole analysis backbone.

They are particularly useful when:

- the number of seeds is modest;
- tail behavior is emphasized;
- a paired contrast is important and we want a low-assumption corroboration;
- model-based assumptions are doubtful.

Good candidates include:

- paired permutation tests;
- paired Wilcoxon tests as secondary support;
- rank-based win-rate summaries.

### Dominance and Pareto views should complement, not replace, inference

Because the project is applied and not purely theoretical, the analysis should also include operational comparisons such as:

- quality vs runtime;
- robustness vs runtime;
- dominance or Pareto-style summaries.

These should not replace inferential analysis, but they add important decision signal and should be treated as a formal complement to the main statistical layer.

### Methods that should remain secondary

The following approaches may still appear, but they should remain secondary unless a very specific need arises:

- standalone omnibus ANOVA as the main argument;
- large-scale all-vs-all post-hoc tables;
- purely p-value-driven ranking;
- highly complex Bayesian hierarchical formulations introduced only for sophistication;
- multivariate omnibus methods that obscure the split-wise and class-wise story.

The goal is not to maximize methodological novelty. The goal is to maximize clarity, robustness, reproducibility, and defensibility under the actual design of the study.

## Effect Estimation

## Main factors

We want explicit factor-level effect estimation for:

- `model_family`
- `x_transform`
- `y_transform`
- `synthetic_policy`
- `run_policy`
- `flowpre_usage`
- `flowgen_usage`

### Important interactions

At minimum:

- `model_family × synthetic_policy`
- `x_transform × y_transform`
- `x_transform × synthetic_policy`
- `x_transform × run_policy`
- `flowpre_usage × synthetic_policy`
- `model_family × run_policy`

### Statistical framework

Recommended main framework:

- linear model or mixed-effects model on the per-run panel;
- seed treated as blocking / random effect where appropriate;
- structural group encoded explicitly.

When using regression-style effect estimation, standard errors should be chosen with care:

- robust standard errors by default where model assumptions are doubtful;
- grouped/clustered or bootstrap-based alternatives where dependency structure requires them.

We want this not only on the macro endpoint, but also on:

- class-wise endpoints;
- Simpson-risk derived indicators;
- split-gap indicators.

## Role of ANOVA and Post-hoc Analysis

ANOVA and post-hoc analysis are part of the intended statistical toolkit, but they are **not** the main analytical backbone and they must not be applied indiscriminately.

### Position

We do want to use:

- ANOVA or ANOVA-style effect decomposition on fitted factorial models;
- post-hoc contrasts when they answer a real structural question.

But we do **not** want:

- exhaustive `N x N` all-vs-all comparison across every possible variant;
- large tables of weakly motivated pairwise p-values;
- the entire final argument to depend only on omnibus significance tests.

### When ANOVA is appropriate

ANOVA is appropriate when the goal is to understand:

- whether a factor contributes meaningful variation to an endpoint;
- whether a selected interaction matters;
- how much structured variance is associated with a methodological axis.

Typical examples:

- effect of `synthetic_policy`;
- effect of `x_transform`;
- interaction `x_transform × synthetic_policy`;
- interaction `model_family × run_policy`.

In this project, ANOVA should be read as a **structured support layer** for factor interpretation, not as the sole decision rule.

### When post-hoc analysis is appropriate

Post-hoc comparisons are appropriate only when:

- there is a clear hypothesis or structural reason for the contrast;
- the compared variants are meaningfully comparable;
- the contrast helps answer a real methodological question.

Examples of justified contrasts:

- `flowgen_official` vs `none`;
- `flowgen_official` vs `kmeans_smote`;
- `x-candidate1` vs `x-standard`;
- one training policy vs another within the same family and structural context.

### How post-hoc should be used

Post-hoc analysis should be:

- selective;
- pre-justified when possible;
- preferably paired when the design allows it;
- always accompanied by effect size and uncertainty.

It should not be reduced to “p-value tables”.

For each relevant contrast, we want:

- mean delta;
- median delta;
- confidence interval;
- win rate;
- and, where useful, multiplicity-aware significance support.

### Recommended stance

The intended stance is:

- descriptive analysis first;
- factorial effect modeling second;
- ANOVA as a support layer for selected global questions;
- post-hoc only for justified contrasts;
- no blind all-vs-all multiple testing exercise.

This should be treated as a methodological guardrail for the final F7 analysis.

### Multiplicity control

Whenever multiple formal contrasts are tested, multiplicity control should be explicit.

The intended stance is:

- no blind exhaustive all-vs-all testing;
- a small confirmatory contrast family when justified;
- multiplicity-aware adjustment for those families when formal significance is reported.

Reasonable options include:

- Holm adjustment for small confirmatory families;
- FDR control for larger exploratory contrast collections.

The final analysis should distinguish clearly between:

- confirmatory contrasts;
- secondary targeted contrasts;
- exploratory contrasts.

## Contrast Registry

Before running the final inferential layer, we should maintain a contrast registry.

This registry should record:

- confirmatory contrasts;
- secondary contrasts;
- exploratory contrasts.

Examples of likely confirmatory or near-confirmatory contrasts:

- `flowgen_official` vs `none`
- `flowgen_official` vs `kmeans_smote`
- `x-candidate1` vs `x-standard`
- selected run-policy contrasts within a fixed family / structural context

The registry helps:

- prevent opportunistic contrast selection after seeing results;
- make multiplicity handling explicit;
- keep the final inferential layer disciplined and reproducible.

## Pairwise Comparison Layer

We also need direct head-to-head comparisons.

These should be paired whenever two variants differ only in one factor while sharing the rest of the structural context.

### Outputs per comparison

For each important pair:

- mean delta
- median delta
- `95%` CI for delta
- win rate
- fraction of seeds where A beats B
- fraction of structural groups where A beats B

Where relevant, the comparison layer should also allow a practical-decision view such as:

- superiority;
- non-inferiority;
- or “difference too small to matter operationally”.

### Suggested methods

- paired bootstrap
- paired permutation test
- paired Wilcoxon as non-parametric support if needed

P-values are acceptable as secondary information, but not as the main basis for conclusions.

## Risk / Tail Analysis

We want a robust view of production risk, not only central tendency.

### Required analyses

- worst-class behavior
- worst quantile behavior
- high-error tails:
  - `p90`
  - `p95`
- per-class high quantiles
- seed-wise worst-case summaries

### Questions

- Does a variant improve mean performance at the cost of worse tails?
- Does it improve one class but destabilize another?
- Does it reduce Simpson-risk while increasing variance?

## Interpretability Analysis

Interpretability should be analyzed as both:

- magnitude of drivers;
- stability of driver structure.

### Primary cross-family surface

Primary transversal surface:

- `semantic_bridge_perturbation`

Meaning:

- `MLP` direct: semantic input-level perturbation summaries
- `MLP FlowPre`: semantic projected input-level perturbation summaries
- `XGBoost`: perturbation surface

### Auxiliary surfaces

Keep as secondary:

- `xgb_native_shap`
- `mlp_flowpre_native_latent_perturbation`

These are analytically useful but should not replace the common semantic surface for cross-family comparison.

### Interpretability caution

Interpretability results must be reported carefully.

In particular:

- high importance does not imply causality;
- importance can be redistributed across correlated or compositional variables;
- small rank changes between nearby features may not imply a real conceptual change in the model.

This caution is especially important in this project because:

- features can be correlated;
- compositional representations induce dependence structure;
- multiple semantically related variables may substitute for each other in importance rankings.

### Required interpretability statistics

Per feature, per structural group:

- mean importance
- std
- stderr
- mean rank
- top-k frequency

### Stability statistics

Across seeds:

- Spearman rank correlation
- mean top-k intersection
- feature recurrence frequency
- stability by class where available

We should also include a **driver-set stability** view, not only an exact feature-rank view.

That means checking whether the same semantic driver family remains important even if the exact top-ranked feature changes among correlated neighbors.

This can be especially useful when:

- several nearby variables encode similar physical information;
- importance redistributes locally but conceptual interpretation remains stable.

### Null / baseline stability reference

Where possible, stability summaries should be contextualized against a minimal null or weak-reference baseline.

The goal is not to build a large extra framework, but to avoid interpreting a raw correlation or top-k overlap without context.

For example:

- whether observed seed-wise rank stability is clearly above a weak/randomized reference;
- whether a stability coefficient should be interpreted as low, moderate, or strong in this setting.

### Main interpretability questions

- Which drivers are globally stable?
- Which drivers change under synthetic augmentation?
- Which drivers differ between direct and FlowPre variants?
- Which drivers are shared by `MLP` and `XGBoost`?
- Are some “improvements” associated with qualitatively suspicious driver shifts?

## Required Plots

The final statistical report should at least support the following plots.

### Predictive performance

- grouped forest plots with mean + CI
- rank plots with uncertainty bars
- split-gap plots
- macro vs worst-class scatter
- macro vs per-class dispersion scatter
- cost-quality frontier

### Factor effects

- coefficient plots for main effects
- coefficient plots for selected interactions
- heatmaps for key interactions

### Simpson-focused plots

- overall vs macro
- macro vs worst-class
- per-class error profiles
- class dispersion distributions

### Tail / robustness plots

- `p90/p95` error comparison
- seed spread boxplots
- worst-case seed plots

### Interpretability

- top-feature stability bars
- rank-correlation distributions
- cross-family driver overlap views
- feature importance heatmaps by variant group

## Reproducibility Artifacts for the Analysis Layer

The final statistical analysis should be driven by explicit manifests, not only by scripts with embedded defaults.

### Analysis manifest

We should define an analysis manifest that freezes:

- campaigns included;
- lineage root used;
- panel version / aggregate version;
- endpoints analyzed;
- derived variables;
- confirmatory contrasts;
- bootstrap settings;
- model formulas;
- plot set to produce.

This is a high-ROI reproducibility layer and should be treated as part of the final analysis contract.

## Data Structures Needed

To support all of the above, we need the following canonical layers.

### 1. Per-run panel

One row per completed valid run, including:

- campaign metadata
- lineage metadata
- seed
- model family
- parsed factors:
  - `x_transform`
  - `y_transform`
  - `synthetic_policy`
  - `run_policy`
  - `flowpre_usage`
  - `flowgen_usage`
- macro endpoints
- per-class endpoints
- split gaps
- cost/runtime

### 2. Structural-group aggregate panel

One row per structural group with aggregated seed summaries.

### 3. Pairwise contrast panel

One row per paired comparison.

### 4. Interpretability aggregate panel

One row per:

- structural group
- feature
- split
- class if applicable

with seed-level aggregation and stability statistics.

### 5. Analysis manifest and contrast registry

The analysis layer should also include lightweight tracked artifacts that declare:

- which data products enter the final report;
- which contrasts belong to the confirmatory layer;
- which bootstrap and modeling defaults are frozen.

## Campaign-Provided Metadata Requirements

The final statistical layer depends critically on what is already persisted by the campaign system.

Before implementing the full analysis stack, we should explicitly require that the campaign runner / ledger / lineage layer provide enough canonical metadata so that the analysis does not need to reopen arbitrary historical artifacts in an ad hoc way.

This section only covers requirements that depend directly on the campaign-run layer.

## Upstream Statistical Analysis Prerequisites

The final statistical analysis does not start at the campaign layer. It depends on a chain of upstream contracts that must already be stable before any inferential or descriptive statistical layer can be treated as valid.

This section is intended to make explicit all blocking and near-blocking dependencies that must be provided by the pipeline before the statistical analysis begins. It includes campaign outputs, but is deliberately broader than the campaign system alone.

The goal is to avoid a situation where the statistical layer looks rigorous while silently depending on unstable, ambiguous, or partially undocumented upstream surfaces.

### 1. Official split contract

The statistical layer depends on a single official split contract already being frozen and traceable.

This means the upstream pipeline must provide, directly or through manifests:

- official split identifier
- split timestamps or split boundary specification
- split role assignment for each row or sample
- confirmation that the analysis panel uses the official temporal split rather than legacy shuffled variants

Without this, no final comparison is methodologically interpretable, regardless of how polished the statistical analysis is.

### 2. Dataset and transform provenance

The statistical layer depends on knowing exactly what dataset candidate each run used and how that candidate was produced.

The upstream pipeline therefore needs to provide stable provenance for:

- raw dataset source identifier
- processed dataset identifier
- dataset candidate identifier
- applied `X` transform
- applied `Y` transform
- any learned transform manifests or scaler manifests
- fit-on-train provenance for learned preprocessing

The analysis layer should not need to reverse-engineer transform lineage from filenames or historical conventions.

### 3. Leakage-safe training/evaluation contract

The statistical layer assumes that the upstream training/evaluation pipeline has already enforced the anti-leakage policy.

This means the upstream surface must make it possible to verify, at minimum:

- learned preprocessing fit only on `train`
- selection/tuning not performed on `test`
- final reported metrics computed on the intended split surfaces
- any raw-space inversion needed for comparison was valid and contract-compliant

The statistical layer is not the place to rediscover leakage; it must be able to inherit a validated contract from upstream.

### 4. Canonical metric grammar

The upstream evaluation surface must provide a metric grammar that is stable enough for later aggregation, contrasts, and reporting.

Before the statistical layer can operate, the pipeline should already provide:

- stable metric identifiers
- explicit split labels
- explicit aggregation level labels such as `overall`, `macro`, `per_class`, and `worst_class` where supported
- explicit value-space labels such as `raw_real` and `native`
- clear distinction between primary comparable metrics and auxiliary diagnostics

The statistical layer should never need to infer metric meaning from ad hoc column naming.

### 5. Class-resolved evaluation surface

Because the project explicitly targets the Simpson-style failure mode, the upstream evaluation layer must preserve class-level information as a first-class surface.

The pipeline therefore needs to provide:

- stable class identifiers
- per-class metric rows
- macro rows
- overall rows
- worst-class summaries where supported
- enough information to compute class-dispersion and within-class vs between-class diagnostics

If the class-resolved surface is missing or partial, the core statistical question of the project becomes underidentified.

### 6. Interpretability surface contract

The upstream pipeline must expose interpretability outputs in a way that is aggregation-ready and family-aware.

This means providing stable, documented surfaces for:

- `MLP` direct semantic feature influence
- `MLP FlowPre` semantic bridge feature influence
- `MLP FlowPre` latent auxiliary influence
- `XGBoost` perturbation influence
- `XGBoost` SHAP auxiliary influence

The statistical layer depends on knowing which of these surfaces is:

- primary for cross-family comparison
- auxiliary for within-family diagnosis

This must be decided upstream of the final analysis, not improvised inside notebooks.

### 7. Run-level validity and closure status

The statistical layer depends on a set of runs that are not just present, but properly closed and auditable.

Therefore the upstream pipeline must provide:

- completed vs failed vs blocked status
- run validity flags
- interpretability validity flags
- campaign closeout status
- lineage pooling readiness where applicable

This is required so that the statistical layer can operate on a clean, declared surface rather than one assembled ad hoc from partially completed artifacts.

### 8. Lineage and replication structure

The final analysis depends on knowing how repeated runs relate structurally across seeds and extensions.

The upstream pipeline must therefore provide:

- stable seed identifiers
- structural grouping across seeds
- lineage identifiers across primary and extensions
- parent-child campaign lineage
- enough metadata to reconstruct expected vs observed replication counts

Without this, any multi-seed aggregation or uncertainty quantification becomes ambiguous.

### 9. Runtime and operational surface

The final report is expected to discuss not only predictive quality but also cost and operational trade-offs.

Therefore the upstream pipeline should already provide:

- training runtime
- interpretability runtime
- total runtime
- any run-level operational anomalies that affect interpretability of runtime summaries

This allows the statistical layer to support cost-quality and robustness-cost analyses without rebuilding runtime summaries manually from logs.

### 10. Warning and anomaly trace surface

The statistical layer is not primarily a logging layer, but it still depends on an upstream reliability surface.

The pipeline should therefore provide:

- warning counts
- warning classification policy
- warning signatures
- surfaced vs silenced-known-noise distinction
- any run-level anomaly marker that could explain suspicious statistical behavior

This matters because some anomalies should exclude or contextualize runs before interpretation, even if the raw metrics exist.

### 11. Artifact discoverability contract

All core upstream artifacts needed by the statistical layer should be discoverable through explicit paths or manifests, not guessed from folder conventions.

This includes, at minimum:

- results artifacts
- metrics artifacts
- prediction sidecars
- interpretability summaries
- run manifests
- lineage aggregate inputs where they already exist

If discoverability is weak, reproducibility of the final analysis becomes fragile even when the underlying results are correct.

### 12. Versioned parser and build provenance

The statistical layer depends not only on the existence of upstream artifacts but also on stable interpretation of those artifacts.

The upstream-to-analysis contract should therefore provide or permit stable provenance for:

- factor parser version
- metric grammar version
- lineage aggregation build version
- panel build timestamp
- campaign/reporting code version where available

This is necessary so that two builds of the analysis panel can be meaningfully compared and audited.

### 13. Blocking vs non-blocking upstream dependencies

Not every upstream dependency has the same severity.

For the purposes of the final statistical layer, the following should be considered effectively blocking:

- missing or ambiguous official split contract
- missing comparability contract for raw-space metrics
- missing class-resolved metric surface
- missing lineage/seed structure for pooled analysis
- missing canonical artifact discoverability
- missing validity/closure state for runs entering the panel

The following are high-value but not always fully blocking if clearly marked:

- runtime surface
- warning surface
- auxiliary interpretability surfaces
- parser/build provenance beyond the minimum required contract

This distinction matters because it clarifies which upstream gaps invalidate the statistical layer entirely and which ones constrain only some sections of the final report.

### 14. Additional upstream contracts that should exist if they do not yet exist

Beyond the surfaces already discussed, there are a few compact upstream contracts that would materially strengthen the final statistical layer and should be treated as required additions if they are not yet formalized.

#### 14.1 Canonical class ontology manifest

Because the Simpson-style analysis depends on class-resolved behavior, the pipeline should expose a single canonical class ontology manifest.

At minimum, that manifest should define:

- stable class identifiers
- canonical class ordering
- class display labels if they differ from internal identifiers
- any class grouping or hierarchy if used later in reporting

This is highly relevant because without a canonical class ontology, per-class comparisons and worst-class summaries can drift across analysis code paths.

#### 14.2 Metric availability manifest

The pipeline should expose an explicit machine-readable manifest describing which metric surfaces are expected and available per run family.

At minimum, this should declare availability for:

- splits
- aggregation levels
- value spaces
- per-class support
- worst-class support
- quantile support if present
- interpretability surfaces

This is relevant because the analysis layer should know whether a surface is absent by design or absent because something broke.

#### 14.3 Structural factor parser contract

If factor fields are not persisted directly by the campaign layer, then a single canonical parser contract should exist upstream and be versioned.

That contract should define how to derive, at minimum:

- `x_transform`
- `y_transform`
- `synthetic_policy`
- `run_policy`
- `flowpre_usage`
- `flowgen_usage`

This is effectively blocking for reproducible factorial analysis if those fields are not already stored in the panel.

#### 14.4 Expected replication manifest

For seed-aggregated analysis, the pipeline should expose an explicit statement of expected replication structure.

At minimum, it should define:

- expected seeds per lineage
- expected seeds per structural group
- whether a panel is intended to be complete or partial

This is relevant because uncertainty summaries, completeness checks, and pooled contrasts all depend on knowing whether missing replications are true absences or expected absences.

#### 14.5 Analysis-ready split comparability marker

The pipeline should expose a compact marker that states whether a run is eligible for the final comparable statistical panel.

This can be implemented as a single derived flag if needed, but it should stand on top of the existing contracts and validity fields.

It should effectively summarize whether:

- the official split contract is satisfied
- raw-space comparability is satisfied
- the run is valid for F7 use
- the required result surfaces exist

This is not strictly a substitute for the underlying metadata, but it would provide a high-value guardrail for downstream statistical tooling.

#### 14.6 Split/class support surface

The pipeline should expose support counts and denominators for the key statistical slices used downstream.

At minimum, this should make available:

- sample counts per split
- sample counts per split and class
- confirmation of which rows contribute to each macro or per-class summary

This is highly relevant because macro, per-class, worst-class, and Simpson-style analyses are hard to interpret cleanly if the statistical layer cannot recover the support behind each reported metric.

#### 14.7 Raw target definition and unit contract

Because the final comparison surface is `raw_real`, the pipeline should expose a compact contract describing the target definition in that space.

At minimum, this should define:

- target identifier
- target space identifier
- unit or scale convention if applicable
- whether any inverse transform was required to reach the comparable surface

This is relevant because the statistical layer should not rely on implicit assumptions about what `raw_real` means when building final tables, effect sizes, or plots.

#### 14.8 Prediction-row join contract

If the final statistical layer is expected to support residual-level diagnostics, calibration views, or any analysis that goes below the run-level metric table, then the pipeline should expose a stable prediction-row join contract.

At minimum, this should make it possible to recover:

- a stable sample identifier or row identifier
- split membership for each prediction row
- class membership for each prediction row where applicable
- target and prediction values on the intended comparison surface

This is relevant because many high-value diagnostics become fragile if prediction sidecars cannot be joined back to the canonical evaluation population in a stable way.

#### 14.9 Feature-schema contract for interpretability

Because interpretability is a major analysis layer, the pipeline should expose a stable feature-schema contract for every semantic interpretability surface.

At minimum, that contract should preserve:

- stable feature identifiers
- stable feature ordering where ordering matters
- surface-specific feature namespaces when direct, projected, and latent spaces differ
- enough provenance to know whether two interpretability tables refer to the same semantic feature space

This is relevant because cross-seed and cross-method aggregation of feature importance becomes ambiguous if feature identity is only implicit in ad hoc CSV conventions.

#### 14.10 Metric aggregation and weighting contract

The pipeline should expose an explicit contract for how aggregate metrics are formed from lower-level quantities.

At minimum, this should make clear:

- whether `macro` means unweighted class average
- whether `overall` means sample-weighted aggregation
- whether any class weighting, masking, or normalization is applied
- the optimization direction of each key metric

This is relevant because the statistical layer cannot interpret differences between `overall`, `macro`, `per_class`, and `worst_class` correctly if the aggregation semantics are only implicit.

#### 14.11 Evaluation-population and exclusion contract

The pipeline should expose a compact contract describing which rows are eligible for evaluation and which rows, if any, were excluded from a given evaluation surface.

At minimum, this should make recoverable:

- the evaluation population definition per split
- whether any rows were excluded post-split from metric computation
- exclusion reasons or exclusion flags when they exist

This is relevant because final statistical summaries can otherwise mix nominally comparable runs that were actually evaluated on slightly different populations.

### 1. Canonical run-level identifiers

Each analyzed run should remain traceable through stable identifiers coming from the campaign system.

Required identifiers include:

- `campaign_id`
- `campaign_lineage_id`
- `root_campaign_id`
- `parent_campaign_id`
- `trial_id`
- `run_id`
- `lineage_trial_group_id`
- `comparison_group_id`
- `seed`
- `replication_index`

These identifiers are essential because they define:

- pairing structure;
- lineage aggregation;
- seed-level replication;
- structural comparability across primary campaign and extensions.

### 2. Canonical factor metadata

The analysis needs a clear factor ontology, but the source of truth for that ontology should come from campaign-provided metadata whenever possible.

At minimum, the analysis layer should be able to recover or derive from the campaign panel:

- `model_family`
- `dataset_candidate_id`
- `run_spec_id`
- `base_config_id`
- `objective_metric_id`
- `run_mode`
- `allow_test_holdout`
- `test_enabled`

In addition, the final analysis panel should expose parsed factors derived from campaign identifiers in a canonical and versioned way:

- `x_transform`
- `y_transform`
- `synthetic_policy`
- `run_policy`
- `flowpre_usage`
- `flowgen_usage`

This means the campaign layer should either:

- persist those parsed fields directly;
- or guarantee a single official parser from `dataset_candidate_id` and `run_spec_id`.

The key point is that factor parsing must not be left as a fragile notebook-side convention.

### 3. Contract and comparability metadata

The analysis depends on knowing that runs are comparable on the intended surface.

Therefore the campaign layer should continue exposing, at minimum:

- `contract_id`
- `raw_metric_contract_id`
- `raw_metric_contract_validation_status`
- `raw_real_available`
- `requires_raw_inversion`
- `raw_inversion_status`
- `value_space_default`
- `variant_fingerprint`

These fields are important because the statistical layer must be able to verify:

- that all runs used in a pooled table are on the same metric contract;
- that the comparison surface is truly `raw_real`;
- that raw-space inversion was available or not required;
- that no run enters a pooled analysis with an invalid contract state.

### 4. Artifact path contract

The final analysis should not depend on guessing artifact locations.

The campaign layer should provide explicit paths for the canonical artifacts needed by the analysis:

- `run_manifest_path`
- `results_path`
- `metrics_long_path`
- `prediction_sidecar_path`
- `interpretability_summary_path`

And, where applicable, the interpretability artifact paths needed for cross-seed aggregation:

- `feature_influence_global_path`
- `feature_influence_per_class_path`
- `input_feature_influence_global_path`
- `input_feature_influence_per_class_path`
- `latent_feature_influence_global_path`
- `latent_feature_influence_per_class_path`
- `xgb_perturbation_feature_influence_global_path`
- `xgb_perturbation_feature_influence_per_class_path`
- `xgb_shap_feature_influence_global_path`
- `xgb_shap_feature_influence_per_class_path`

This ensures the statistical layer can:

- aggregate results without re-deriving paths from folder conventions;
- aggregate interpretability across seeds and lineage;
- validate whether a requested analysis surface is actually available.

### 5. Runtime and operational metadata

Runtime is not just engineering detail; it is part of the operational analysis.

The campaign layer should provide:

- `training_runtime_s`
- `interpretability_runtime_s`
- `total_runtime_s`

This allows the final analysis to support:

- cost-quality frontiers;
- robustness vs runtime trade-offs;
- production-oriented interpretation of whether a gain is worth its computational cost.

### 6. Warning and execution metadata

Warnings and execution status are not the main statistical targets, but they are part of the reliability contract of the panel.

The campaign layer should provide:

- `execution_status`
- `campaign_valid`
- `campaign_valid_interpretability`
- `campaign_valid_f7`
- `warning_count_total`
- `warning_count_silenced_known_noise`
- `warning_count_surfaced`
- `warning_policy_counts`
- `warning_signature_counts`

This matters because:

- the final analysis should know whether a run is statistically usable;
- silent mixing of partially valid runs would weaken the final report;
- warning traces can help explain suspicious subpanels or artifacts without reopening raw logs manually.

### 7. Split-aware metric availability

The campaign system should continue to expose enough information for the analysis layer to know that the full split structure exists for a run.

At minimum, campaign-derived results should make it possible to recover:

- `train`
- `val`
- `test`

for:

- macro metrics
- per-class metrics
- worst-class metrics where supported
- quantile views where supported

The important part here is that the analysis layer should not need to infer split availability indirectly. It should be able to rely on campaign-provided artifacts and contract validation.

### 8. Lineage-pooling metadata

Because the final analysis aggregates primary campaign plus seed extensions, the campaign layer should be treated as responsible for exposing the metadata needed for valid lineage pooling.

Required lineage-pooling fields include:

- `lineage_pool_ready`
- `lineage_pool_blockers`
- included/excluded campaign ids at lineage level
- `expected_seed_count`
- `observed_seed_count`
- `seed_completeness_ratio`

These fields make it possible to distinguish between:

- a panel that is nominally present;
- and a panel that is actually complete and methodologically poolable.

### 9. Class metadata surface

Because class-wise behavior is central to the final analysis, the campaign-provided results surface should preserve class-resolved metric support.

The statistical layer therefore depends on campaign outputs preserving:

- per-class metric rows;
- stable class identifiers;
- worst-class summaries where supported.

This does not yet force a policy about how to interpret missing classes or incomplete groups, but it does require the campaign artifacts to preserve class-level granularity.

### 10. Versionable parser and panel provenance

The analysis depends not only on raw campaign fields, but on how those fields are interpreted into factors and pooled panels.

Therefore, the campaign-adjacent analysis layer should carry versionable provenance such as:

- factor parser version
- lineage aggregate build version
- panel build timestamp

Even if this is implemented later outside the runner itself, it still belongs to the contract of the campaign-derived statistical surface.

## Derived Variables We Should Explicitly Build

The analysis should explicitly derive and store:

- `test_minus_val`
- `val_minus_train`
- `test_minus_train`
- `simpson_gap_rrmse`
- `class_dispersion_rrmse`
- `worst_class_penalty_rrmse`
- equivalent derived indicators for selected secondary metrics where useful

Where feasible, the derived layer should also support:

- within-class centered performance summaries;
- class-specific split-gap summaries;
- practical-decision labels for selected contrasts.

## Practical Decision Logic

The final report should not rely on a single scalar.

We want at least three decision views:

### 1. Aggregate quality

Driven by:

- `val_raw_real_macro_rrmse`
- `test_raw_real_macro_rrmse`

### 2. Simpson-safe quality

Driven by:

- per-class `rrmse`
- worst-class `rrmse`
- class dispersion
- Simpson-risk indicators

### 3. Operational robustness

Driven by:

- seed stability
- split gaps
- tail metrics
- runtime cost

This allows us to separate:

- best average performer;
- best class-balanced performer;
- best robust / production-friendly performer.

## Statistical Guardrails

- Do not iterate on `test` after freeze.
- Do not collapse class-wise behavior into only macro summaries.
- Do not use only p-values.
- Always report effect size and uncertainty.
- Prefer paired comparisons where design allows.
- Make clear when an analysis is:
  - descriptive;
  - inferential;
  - exploratory.
- Do not interpret seed-based uncertainty as full deployment uncertainty.
- Do not treat the panel as i.i.d. when it is structurally grouped.
- Do not let `R²` dominate the Simpson-risk narrative.

## What This Plan Requires From the Implementation Layer

The reporting and aggregation layer should support:

- macro and per-class extraction for `train`, `val`, `test`
- split-gap derivation
- factor parsing from candidate / spec ids
- paired comparison scaffolding
- bootstrap-ready panels
- interpretability aggregation across seeds
- stability summaries for drivers

## Final Outcome We Want

After this analysis layer is implemented, we want to be able to answer, with statistical backing:

- which axes help overall;
- which axes reduce Simpson-type failure;
- which combinations are synergistic;
- which variants remain strong in both `val` and `test`;
- which variants are stable across seeds;
- which drivers are truly important and stable;
- which candidate is most defensible for the final thesis conclusion.

## Status

This is the first written version of the final F7 statistical analysis plan.

It is intentionally ambitious and should be refined, but from now on it acts as the target scope for the final analysis layer.
