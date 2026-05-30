# F7 Statistical Analysis Plan: Structured Version

## Purpose

This document is the structured companion to [docs/f7_statistical_analysis_plan.md](f7_statistical_analysis_plan.md).

Its role is to reorganize the same statistical intent into a cleaner, more normative, and more implementation-oriented structure without replacing the original planning document.

The original plan remains the full narrative source.
This document is the human-readable operational structure for downstream implementation.

## Scope

The final F7 statistical layer is intended to support:

- final comparison of methodological variants under the official temporal split;
- seed-aggregated analysis across a root campaign and seed extensions;
- class-wise analysis motivated by Simpson-type failure risk;
- interpretability aggregation and stability analysis;
- thesis-ready descriptive, inferential, and operational reporting.

This layer is broader than a leaderboard.
It is meant to produce defensible evidence about:

- predictive quality;
- class-wise quality;
- robustness across seeds;
- methodological effect sizes;
- interactions and synergies;
- operational runtime trade-offs;
- driver stability in interpretability.

## Statistical Framing

### Non-i.i.d. panel

The final panel must not be treated as i.i.d.

Runs share:

- the same official temporal split contract;
- common upstream preprocessing logic;
- repeated structural groups;
- repeated seeds;
- repeated methodological components.

Therefore:

- paired and grouped analyses are preferred whenever the design allows;
- structural grouping must be explicit in estimation and uncertainty quantification;
- global pooled summaries must not silently replace paired evidence.

### Two kinds of robustness

The final analysis must distinguish:

- algorithmic robustness:
  variation across seeds within the official split;
- temporal robustness:
  behavior across `train`, `val`, and `test` under the official temporal split.

Seed uncertainty is not full population uncertainty.
It measures mainly algorithmic stability, not uncertainty over alternative temporal worlds.

### Freeze discipline

The final methodology must distinguish:

- selection before freeze:
  shortlist logic and narrowing performed on `val`;
- final characterization after freeze:
  frozen statistical readout emphasizing both `val` and `test`.

`test` is not part of the iterative design loop.
After freeze, it is part of the final descriptive and inferential readout.

Operational freeze for block `14`:

- `val` governs shortlist logic before freeze;
- `val` and `test` are co-primary reporting surfaces after freeze;
- `test` must not be used to reopen design decisions.

## Research Questions

The final analysis should answer:

1. Which methodological axes improve predictive performance overall?
2. Which axes improve class-wise performance rather than only aggregate performance?
3. Which combinations show synergy or antagonism?
4. Which variants are robust across seeds?
5. Which variants generalize well across `train`, `val`, and `test`?
6. Which drivers are most important, and how stable are they?

## Estimands

The main estimands are:

- average marginal effect of each methodological factor on:
  - `val_raw_real_macro_rrmse`
  - `test_raw_real_macro_rrmse`
- average marginal effect of each methodological factor on:
  - per-class `rrmse`
  - worst-class `rrmse`
  - Simpson-risk indicators
- average paired contrast between selected alternatives under matched structural contexts
- average split-gap quantities:
  - `val_minus_train`
  - `test_minus_train`
  - `test_minus_val`
- average driver importance and driver stability on the canonical semantic interpretability surface

## Endpoints

### Co-primary predictive endpoints

The co-primary predictive endpoints after freeze are:

- `val_raw_real_macro_rrmse`
- `test_raw_real_macro_rrmse`

### Parallel class-wise co-primary endpoints

The class-wise co-primary layer must include:

- per-class `rrmse`
- worst-class `rrmse`
- class-dispersion indicators
- Simpson-risk indicators

### Secondary predictive endpoints

The final layer should also support:

- `rmse`
- `mse`
- `r2`
- `mape`
- additional `raw_real` summaries where useful

`R²` remains auxiliary in the Simpson-focused narrative.

### Diagnostic endpoints

The final diagnostic layer should preserve:

- `val_minus_train`
- `test_minus_train`
- `test_minus_val`
- runtime endpoints
- warnings and execution anomalies as contextual metadata

## Performance Views Required

The final analysis must support:

- aggregate performance;
- class-wise performance;
- worst-class performance;
- split-gap behavior;
- tail behavior;
- robustness across seeds;
- operational runtime trade-offs.

## Simpson-Focused Analysis Layer

### Core hypothesis

The synthetic-data line was not introduced only to improve a macro average.

The key concern is Simpson-type failure:

- a model may look good in aggregate;
- `R²` may appear inflated;
- but class-wise performance can still be poor;
- the model may effectively interpolate across classes instead of learning class-specific structure properly.

### Required measurements

The Simpson-focused layer should include:

- macro metrics;
- per-class metrics;
- worst-class metrics;
- class-dispersion summaries;
- within-class / between-class decomposition whenever feasible.

### Derived indicators

The derived indicator layer should explicitly support:

- `simpson_gap_rrmse`
- `class_dispersion_rrmse`
- `worst_class_penalty_rrmse`
- class-specific split-gap summaries
- within-class centered summaries where feasible

## Statistical Summaries Per Group

For every structural group and major endpoint, the descriptive layer should report:

- mean
- `95%` confidence interval
- median
- IQR
- seed-level spread
- min / max
- `p25 / p75`
- `p90 / p95` where relevant for risk

## Uncertainty Strategy

### Default uncertainty stance

Prefer bootstrap-based uncertainty summaries over naive normal approximations when possible.

### Recommended defaults

- structural-group summaries:
  bootstrap over seeds or the smallest defensible repeated unit;
- paired contrasts:
  paired bootstrap by default;
- model-based factor analyses:
  robust standard errors by default;
- strong dependence structures:
  cluster-aware uncertainty where appropriate.

### Uncertainty questions

Uncertainty summaries should answer:

- how stable is a variant across seeds?
- how uncertain is the difference between A and B?
- how often does A beat B?
- how unstable are tail or worst-class behaviors?

## Method Hierarchy

### Primary inferential backbone

The preferred inferential backbone is:

1. descriptive multi-seed summaries with intervals;
2. paired / blocked comparisons whenever the design allows;
3. mixed-effects or repeated-structure regression for factor effects and interactions.

### Support methods

The following are support layers, not the main backbone:

- ANOVA-style decomposition on fitted models;
- selective post-hoc contrasts;
- non-parametric and permutation sensitivity checks;
- Pareto or dominance views for applied interpretation.

### Methods that should remain secondary

The following should remain secondary unless a very specific need appears:

- standalone omnibus ANOVA as the main argument;
- all-vs-all post-hoc tables;
- purely p-value-driven ranking;
- unnecessarily complex Bayesian formulations introduced only for sophistication;
- multivariate omnibus methods that obscure the split-wise and class-wise story.

## Effect Estimation

### Main factors

The final factor layer should estimate effects for:

- `model_family`
- `x_transform`
- `y_transform`
- `synthetic_policy`
- `run_policy`
- `flowpre_usage`
- `flowgen_usage`

### Important interactions

The minimum interaction shortlist should include:

- `model_family × synthetic_policy`
- `x_transform × y_transform`
- `x_transform × synthetic_policy`
- `x_transform × run_policy`
- `flowpre_usage × synthetic_policy`
- `model_family × run_policy`

### Preferred statistical framework

Preferred framework:

- mixed-effects or repeated-structure regression on the per-run panel;
- seed treated as blocking or random effect where appropriate;
- structural grouping encoded explicitly.

This should be run not only on the macro endpoint but also on:

- class-wise endpoints;
- Simpson-risk indicators;
- split-gap indicators.

## ANOVA and Post-hoc Role

### Position

ANOVA and post-hoc analysis are part of the toolkit, but they are not the main backbone.

They should be used:

- when they answer a real structural question;
- when contrasts are clearly motivated;
- when they support factor interpretation rather than replace it.

### ANOVA role

ANOVA is appropriate to understand:

- whether a factor contributes meaningful variation;
- whether a selected interaction matters;
- how much structured variance is associated with a methodological axis.

### Post-hoc role

Post-hoc comparisons are appropriate only when:

- the contrast is methodologically motivated;
- the compared variants are meaningfully comparable;
- the result helps answer a real question.

Every important contrast should report:

- mean delta;
- median delta;
- confidence interval;
- win rate;
- multiplicity-aware significance support when formal testing is reported.

### Multiplicity control

The final inferential layer should distinguish:

- confirmatory contrasts;
- secondary targeted contrasts;
- exploratory contrasts.

Reasonable defaults:

- Holm adjustment for small confirmatory families;
- FDR control for larger exploratory collections.

### Confirmatory contrast families

The minimum confirmatory registry should include structurally motivated contrast families over:

- `synthetic_policy`
- `x_transform`
- `y_transform`
- `run_policy`
- selected interactions already frozen in the structured spec

`model_family` should remain a structured comparison layer, but not become an all-vs-all contrast explosion.

## Pairwise Comparison Layer

The pairwise layer should be built whenever two variants differ only in one factor while sharing the rest of the structural context.

For each important pair, the layer should support:

- mean delta;
- median delta;
- `95%` CI for delta;
- win rate;
- fraction of seeds where A beats B;
- fraction of structural groups where A beats B;
- superiority / non-inferiority / practical-relevance readouts where useful.

Suggested methods:

- paired bootstrap;
- paired permutation test;
- paired Wilcoxon as non-parametric support if needed.

## Risk and Tail Analysis

The final report should not only describe central tendency.

It should explicitly analyze:

- worst-class behavior;
- worst quantile behavior;
- high-error tails such as `p90` and `p95`;
- per-class high quantiles;
- seed-wise worst-case summaries.

The main questions are:

- does a variant improve the mean at the cost of worse tails?
- does it improve one class while destabilizing another?
- does it reduce Simpson-risk while increasing variance?

## Interpretability Analysis

Interpretability must be analyzed as both:

- magnitude of drivers;
- stability of driver structure.

### Primary cross-family surface

Primary transversal surface:

- `semantic_bridge_perturbation`

### Auxiliary surfaces

Auxiliary surfaces:

- `xgb_native_shap`
- `mlp_flowpre_native_latent_perturbation`

### Required interpretability statistics

The interpretability aggregation layer should support:

- mean importance;
- standard deviation;
- standard error;
- mean rank;
- top-k frequency;
- cross-seed stability summaries.

Default primary stability depth:

- `top-k = 10`

### Stability layer

The stability layer should support:

- rank-correlation summaries;
- top-k intersection summaries;
- semantic driver-family stability where correlated features are involved;
- null or weak-reference contextualization where possible.

## Required Plots

The final report should support at least:

- grouped forest plots with mean + CI;
- rank plots with uncertainty bars;
- split-gap plots;
- macro vs worst-class scatter;
- macro vs class-dispersion scatter;
- cost-quality frontier;
- coefficient plots for main effects;
- coefficient plots for selected interactions;
- interaction heatmaps;
- overall vs macro;
- macro vs worst-class;
- per-class error profiles;
- class-dispersion distributions;
- `p90/p95` comparison plots;
- seed-spread boxplots;
- worst-case seed plots;
- top-feature stability bars;
- rank-correlation distributions;
- cross-family driver overlap views;
- feature-importance heatmaps by variant group.

## Reproducibility Artifacts

The final analysis layer should be driven by explicit reproducibility artifacts.

### Analysis manifest

The analysis manifest should freeze:

- campaigns included;
- lineage root used;
- panel version or aggregate version;
- endpoints analyzed;
- derived variables;
- confirmatory contrasts;
- bootstrap settings;
- model formulas;
- plot set to produce.

### Contrast registry

The contrast registry should record:

- confirmatory contrasts;
- secondary contrasts;
- exploratory contrasts.

At minimum, the registry should also record:

- endpoint family;
- structural pairing rule;
- multiplicity rule;
- whether the contrast is selection-facing or final-reporting-only.

## Canonical Data Products

The final statistical layer should operate on canonical data products:

1. per-run panel
2. structural-group aggregate panel
3. pairwise contrast panel
4. interpretability aggregate panel
5. analysis manifest and contrast registry

## Upstream Prerequisites

The final statistical analysis depends on upstream contracts that must already be stable.

### Blocking prerequisites

The following should be treated as effectively blocking:

- official split contract;
- comparability contract for raw-space metrics;
- class-resolved metric surface;
- lineage and seed structure for pooled analysis;
- canonical artifact discoverability;
- validity and closure state for runs entering the panel.

### High-value but not always fully blocking

These are high-value and should exist if possible:

- runtime surface;
- warning surface;
- auxiliary interpretability surfaces;
- parser and build provenance beyond the minimum required contract.

### Additional upstream contracts that should exist if absent

The upstream layer should also expose or formalize:

- canonical class ontology manifest;
- metric availability manifest;
- structural factor parser contract;
- expected replication manifest;
- analysis-ready split comparability marker;
- split/class support surface;
- raw target definition and unit contract;
- prediction-row join contract;
- feature-schema contract for interpretability;
- metric aggregation and weighting contract;
- evaluation-population and exclusion contract.

## Campaign-Derived Requirements

The campaign layer should expose enough canonical metadata so the statistical layer does not have to reopen arbitrary historical artifacts ad hoc.

This includes:

- canonical run-level identifiers;
- canonical factor metadata;
- comparability and contract metadata;
- artifact paths;
- runtime metadata;
- warning and execution metadata;
- split-aware metric availability;
- lineage-pooling metadata;
- class metadata surface;
- parser and panel provenance.

## Derived Variables

The analysis implementation should explicitly build and preserve:

- `test_minus_val`
- `val_minus_train`
- `test_minus_train`
- `simpson_gap_rrmse`
- `class_dispersion_rrmse`
- `worst_class_penalty_rrmse`
- selected equivalents for secondary metrics where useful

## Practical Decision Logic

The final report should support practical interpretation along three axes:

### Aggregate quality

- overall predictive quality
- stability across seeds
- uncertainty around ranking

### Simpson-safe quality

- class-wise safety
- worst-class protection
- reduction of Simpson-risk indicators

### Operational robustness

- runtime cost
- cost-quality trade-offs
- warning or anomaly burden when relevant

The practical-significance layer should explicitly reserve a versioned threshold field for the main metric:

- `minimum_practical_effect_rrmse = TBD_before_final_reporting`

## Statistical Guardrails

The final analysis should respect the following guardrails:

- do not treat the panel as i.i.d.;
- prefer paired comparisons where the design allows;
- do not let `R²` dominate the Simpson narrative;
- do not rely on omnibus ANOVA alone;
- do not reduce the inferential layer to p-value tables;
- do not mix incomparable runs silently;
- do not conflate algorithmic robustness with temporal robustness.

## Implementation Requirements

The downstream implementation should provide:

- paired comparison scaffolding;
- bootstrap-ready panels;
- lineage-aware aggregate panels;
- class-wise metric availability;
- interpretability surfaces with stable feature identities;
- analysis manifests and registries;
- reproducible plot generation from canonical panels.

## Final Outcome

The desired final outcome is a statistical layer that can answer, in a clean and reproducible way:

- which axes improve predictive quality;
- which axes reduce Simpson-type failure;
- which combinations are robust across seeds;
- which effects survive on `test`;
- which drivers are stable and semantically defensible;
- which choices remain attractive once runtime and robustness are considered together.

## Status

This document is the structured operational companion to the original plan.
It is not a replacement for the original narrative document.

For block `14`, its methodological content should now be read as frozen operational guidance rather than open planning.
