# Project Context

## 1. Overview

This project is a thesis (`TFG`) in Python focused on data science / machine learning for a real industrial problem: predicting the `initial setting time` (`init`) of cement from chemical composition and material phase information.

The dataset contains roughly 6k batches/lots and includes:

- cement type information, with an approximate class proportion of `50/30/20`
- date information
- anonymized transformed variables due to NDA / publication constraints

The project should not be understood as a strictly i.i.d. prediction problem. Even if there are no clear temporal cycles, there is distribution shift over time because the supply / origin of the material changes, which affects both chemistry and downstream behavior.

## 2. Problem Framing

### Confirmed facts

- The main final target is `init` (`initial setting time`).
- `6h` and `end` are not currently part of the main final thesis focus.
- The project is centered on predictive performance under realistic distribution shift, not only on maximizing a metric under a random split.
- Synthetic data generation exists in the project, but as a secondary methodological line that supports or complements the predictive workflow.

### Methodological position

The thesis is intended to defend the following narrative:

- this is a real industrial problem
- the data exhibit distribution shift
- leakage must be avoided
- comparisons between methodological variants must be fair
- the thesis should close with defendable evidence about which pipeline would be most reasonable for near-term production use

## 3. Canonical Conceptual Pipeline

The implementation has multiple iterations and the codebase is not yet cleanly organized, but the current canonical conceptual pipeline is:

1. Start from a primary industrial dataset.
2. Remove impossible / physically invalid values.
3. Keep relevant rows and columns only.
4. Clean and prepare the data.
5. Create splits.
6. Apply different scaling / preparation variants.
7. Train `FlowPre` as an invertible multivariate transformation toward a space closer to a multivariate normal distribution while preserving structure between variables as much as possible.
8. Use that transformation to create derived dataset variants.
9. Use `FlowGen` on the best transformed space to generate realistic synthetic data.
10. Train an `MLP` on original and derived datasets to compare predictive performance.

### Important distinction

- This is the canonical conceptual workflow.
- It does **not** mean the current repository already implements this as a single clean, reproducible, centralized pipeline.

## 4. Current Core Components

### Active / central lines

#### `MLP`

- Main predictive model.
- Serves as the primary prediction engine / baseline across original and derived datasets.
- Intended to be the main vehicle for comparing methodological improvements.

#### `FlowPre`

- Methodologically important, not a side experiment.
- Acts as an invertible multivariate transformation / scaler for `X`.
- Motivation: go beyond per-feature scaling such as `StandardScaler` or `RobustScaler`, and instead approximate a richer multivariate normalization that preserves relationships between variables.
- The current best hyperparameters should be treated as operational priors inherited from optimization under the historical shuffled split.
- They are useful starting points, but they should not be interpreted as final canonical settings for the temporal thesis close.
- Official temporal FlowPre runs already exist materially under `outputs/models/official/flow_pre/`, and that phase is now closed.
- The current observed state is `162` complete official runs under `outputs/models/official/flow_pre/`:
  - `22` from `revalidate_v1`
  - `20` from `explore_v2`
  - `30` from `explore_v3`
  - `11` from `explore_v4`
  - `79` from the final `reseed`
- The reseed closure reached `79/79` planned runs complete, `0` incomplete, `0` pending.
- The expanded FlowPre evaluation layer under `outputs/reports/f6_explore_v2_results/` remains valuable as historical analysis surface, including the family views (`rrmse`, `mvn`, `fair`) and the additional lenses `flowgencandidate` and `rrmse_primary`.
- That evaluator introduced the global reconstruction sanity layer `global_reconstruction_status`, and it was the operational basis used to close FlowPre, but it should no longer be read as a still-open provisional surface.
- The official closeout artifacts now live in `outputs/models/official/flowpre_finalists/`, with one materialized finalist for each descriptive lens plus two FlowGen-oriented candidates.
- The descriptive finalists `rrmse`, `mvn`, and `fair` should be read as specialized FlowPre scalers / upstreams used to scale the dataset and derive dataset variants. They are not the main bases for training FlowGen.
- `candidate_1` and `candidate_2` are the two actual FlowPre work bases selected to start and develop the FlowGen phase.
- The final picks by lens are frozen for this phase:
  - `rrmse_primary` -> `hf256|l3|rq1x3|frq3|lr1e-4|mson|skoff`, seed `5678`
  - `mvn` -> `hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff`, seed `5678`
  - `fair` -> `hf192|l3|rq1x6|frq6|lr1e-3|mson|skoff`, seed `5678`
  - `flowgencandidate_priorfit` -> `hf192|l3|rq1x6|frq6|lr1e-3|mson|skoff`, seed `9101`
  - `flowgencandidate_robust` + `flowgencandidate_hybrid` -> `hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff`, seed `2468`
- A promoted `rrmse` winner manifest still exists under the winning run directory in `outputs/models/official/flow_pre/`, but it should be read only as an inherited technical compatibility artifact for legacy/provisional FlowGen entrypoints.
- It must not be read as the conceptual or operational main input of the FlowGen phase. The semantic role remains: `rrmse`, `mvn`, and `fair` are specialized scalers/upstreams, while `candidate_1` and `candidate_2` are the two FlowGen work bases for this phase.

#### `FlowGen`

- Main generative line.
- Used to generate realistic synthetic data from the best transformed feature space.
- Its role is subordinate to the predictive thesis objective: improving or supporting the predictive workflow rather than becoming the final goal by itself.
- The current best hyperparameters should likewise be treated as operational priors inherited from the historical shuffled split.
- They require revalidation under the temporal split before being promoted to closing configuration.
- `FlowGen` temperature tuning exists in the repository, but for now it belongs to the experimental line and remains outside the canonical closing path.
- Operationally it is no longer just unblocked: the official temporal exploration has already been materialized from the selected FlowPre work bases `candidate_1` and `candidate_2`, while any remaining `rrmse`-manifest consumption should be understood only as inherited technical compatibility.
- The current observed state is `50` complete official temporal FlowGen runs under `outputs/models/official/flowgen/`, excluding bases and campaign summaries:
  - `30` exploratory official runs from the pre-reseed frontier
  - `20` runs from the final reseed
- The historical v6-style ranking remains as traceability under `outputs/models/official/flowgen/campaign_summaries/rankings/`.
- The historical closeout layer for FlowGen exploration remains the final ranking implemented in `scripts/f6_flowgen_rank_official_v2.py`, with artifacts under `outputs/models/official/flowgen/campaign_summaries/final_rankings/`.
- The post-reseed aggregation closeout lives under `outputs/models/official/flowgen/campaign_summaries/post_reseed/`.
- The final winner family after post-reseed aggregation is `E03`.
- The unique materialized official finalist / winner of `FlowGen` is `flowgen_tpv1_c2_train_e03_seed2468_v1`, copied under `outputs/models/official/flowgen_finalist/`.
- The next active phase is no longer FlowGen reseed or open local FlowGen exploration, but downstream final comparison / closeout with `MLP`.
- `candidate_1` and `candidate_2` under `outputs/models/official/flowpre_finalists/` are the two selected FlowPre work bases for the FlowGen phase.
- Once FlowGen is closed and its promoted outputs exist, `candidate_1` and `candidate_2` should be read as historical traceability artifacts of how the generative phase started, not as indefinitely active canonical artifacts.

### Secondary / experimental line

#### Experimental `train_only` branch

- There is now a materially closed local experimental branch under `outputs/models/experimental/train_only/`.
- This branch does **not** reopen the official temporal closeout of `FlowPre` or `FlowGen`.
- Its purpose is downstream-oriented: test whether a generator fitted specifically to the real `train` domain can be more useful to rebalance `train` before `MLP`, even if it is not the best generator under temporal validation.
- The branch already has:
  - promoted `FlowPre train_only` work bases under `outputs/models/experimental/train_only/flowpre_finalists/`
  - a unique local `FlowGen train_only` finalist under `outputs/models/experimental/train_only/flowgen_finalist/`
- The promoted local `FlowGen train_only` finalist is `flowgen_trainonly_tpv1_ct1_reseedfinal_r3a2_t06_clip125_seed15427_v1`.
- In this branch, `monitoring_policy="train_only"` means the artifact key `"val"` is a train-derived pseudo-validation surface, not the canonical temporal validation split.
- Therefore, the `train_only` line is useful as a local downstream candidate, but it must not be confused with the official promoted `FlowGen` winner of the project.

#### `CVAE-CNF`

- Experimental line explored during the project.
- Code remains in the repository.
- It is not considered the main line and is not especially viable today as the preferred project direction.

## 5. Meaning of Key Variants

These names are used mainly around `FlowPre` candidate selection:

- `rrmse` = `relative root mean squared error`
- `mvn` = `multivariate normal`
- `fair` = aggregated / weighted criterion used mainly to select more balanced `FlowPre` candidates across metrics

### Operational note

For the closed FlowPre phase, the operative `fair` criterion is the one implemented in the current evaluator and used during the final selection. Historical alternatives may still exist in notebooks or exploratory cells, but they no longer govern this phase.

## 6. What Is Already Built

The current project already includes:

- a dataset of approximately 6k batches with date and cement type
- monotonic transformations used for anonymization
- cleaning of impossible values using physical rules
- a global `sum_chem` / `sum_phase` quality filter treated as accepted domain preprocessing
- an outlier workflow based on:
  - univariate density-based filtering
  - `Isolation Forest`
- a single class-conditioned `MLP` as the main predictive model
- advanced work with `nflows`:
  - as an invertible scaler / transformation (`FlowPre`)
  - as a basis for conditional synthetic data generation (`FlowGen`)
- realism metrics for synthetic generation such as `MMD`, `Wasserstein`, and `KS`
- a post-generation filtering step to discard impossible synthetic samples

## 7. Data and Artifact Status

### Confirmed source-of-truth elements

- `data/raw` contains the primary source data.
- `config/` contains valuable canonical base configuration plus public-safe tracked copies for the mixed sensitive mappings.
- local operational overlays for mixed sensitive config now belong under `config/local/` and must remain local-only.
- `data/splits/official/init_temporal_processed_v1/` now contains the canonical official temporal split contract created in F2, including manifest, row assignments, split summaries, class counts, and minimal drift artifacts.
- the canonical data path now reaches safely up to:
  - `data/sets/official/init_temporal_processed_v1/raw/df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1/`
  - `data/sets/official/init_temporal_processed_v1/scaled/` for the `16` classical deterministic `X × Y` scaled datasets materialized in F5a

### Provisional stance on derived data

- Many artifacts under `data/sets/` should currently be treated as potentially regenerable derived artifacts.
- They should not yet be assumed to be the definitive source of truth.
- A later review will be needed to determine which derived datasets are official reproducible products and which are historical / intermediate accumulation.
- `data/sets/df_input/`, `data/sets/scaled_sets/`, `data/sets/augmented_scaled_sets/`, and persisted FlowPre-derived variants remain legacy until regenerated from the current official raw bundle with canonical manifests.
- after the F5a boundary cleanup, `data/sets/scaled_sets/` should be read as a historical namespace; canonical classical scaled datasets now live under `data/sets/official/init_temporal_processed_v1/scaled/`.

### Outputs

- Most of `outputs/` should be treated as generated experiment artifacts, not as canonical source material.
- For the temporal closing line, the repo now reserves a separate official model namespace under `outputs/models/official/` so newly revalidated `FlowPre` / `FlowGen` runs do not mix again with historical outputs.
- In the current real state, `outputs/models/official/flow_pre/` contains `162` complete official FlowPre runs across `revalidate_v1`, `explore_v2`, `explore_v3`, `explore_v4`, and the final `reseed`.
- `outputs/models/official/flowpre_finalists/` materializes the final selected `FlowPre` winners and their rationale.
- Inside that namespace, `rrmse`, `mvn`, and `fair` are specialized scaling/upstream artifacts, while `candidate_1` and `candidate_2` are the two work bases selected for the FlowGen phase.
- `outputs/models/official/flowgen_finalist/` now materializes the final unique promoted winner of `FlowGen`.
- `outputs/models/experimental/train_only/flowpre_finalists/` now materializes the two local `FlowPre train_only` work bases.
- `outputs/models/experimental/train_only/flowgen_finalist/` now materializes the unique local `FlowGen train_only` finalist.
- A new tracked lightweight closeout layer now exists outside `outputs/` under:
  - `config/finalists_registry.yaml`
  - `config/finalists/`
  - `docs/finalists_registry.md`
  - `docs/finalists/`
- That layer is intentionally public-safe and small: it stores lightweight winner/finalist metadata, sanitized config snapshots, rationale summaries, and `local_only` references back to the heavy local artifacts.
- `outputs/reports/f6_reseed_topcfgs_v4_continue/` contains the current closure summary for the reseed.
- `outputs/reports/f6/`, `outputs/reports/f6_explore_v2/`, `outputs/reports/f6_explore_v2_results/`, `outputs/reports/f6_explore_v3/`, and `outputs/reports/f6_explore_v4/` should now be read as historical snapshots unless a file explicitly says it is the current reference.
- Within the historical evaluation surface, `flowgencandidate_*` inherits `reject` from the global reconstruction sanity layer whenever `global_reconstruction_status=fail`.

### Visibility boundary for the repo surface

- `docs/repo_visibility_matrix.md` is the practical registry of what is:
  - showable/publicable;
  - local/private/NDA;
  - mixed and therefore split into a public-safe tracked copy plus a local overlay.
- the tracked lightweight finalist layer under `config/finalists*` and `docs/finalists*` is part of the showable/publicable repo surface and acts as the canonical light backup of the final selections.
- for the current hardening stage, `data/raw/`, `data/processed/`, `data/cleaned/`, `data/splits/`, `data/sets/`, and generated artifacts under `outputs/` should be treated as local-only for git/publication purposes even if they remain valuable locally.
- the tracked copies of `config/type_mapping.yaml` and `config/column_groups.yaml` should be read as public-safe repo copies; the operational private versions belong under `config/local/`.
- that lightweight finalist layer does not replace the need to back up `data/raw/`, `config/local/`, and promoted local `outputs/`; the operational restore policy is documented in `docs/backup_and_restore.md`.

## 8. Repository Reality vs Project Reality

### Confirmed facts

- The repository is not yet well organized.
- Logic is spread across scripts, notebooks, versioned training scripts, and root-level modules.
- Notebooks are exploratory / operational rather than final polished deliverables.
- Some notebooks were used to run training or operational tasks that ideally should have been run from scripts or the terminal.

### Important interpretation rule

Poor placement in the repository should **not** be taken as evidence that a piece of logic is useless.

Many scripts and notebooks may still contain:

- reusable functions
- important methodological decisions
- useful plots, tables, or sanity checks
- reference material for reconstructing final notebooks and reporting

Anything that is currently misplaced but valuable should be treated as **logic to preserve / re-home later**, not automatically as discardable legacy.

## 9. Notebooks and Versioned Scripts

### Notebooks

- Current notebooks are mainly exploratory and operational.
- There are no final polished notebooks yet.
- None should be automatically discarded at this stage, because they may still contain useful reference knowledge that is not fully captured elsewhere.

### Versioned scripts in `training_scripts/`

- The latest `1-2` versions in each family are usually the most operational candidates.
- Earlier versions are usually historical.
- However, older scripts should not be discarded automatically if they contain unique or reusable logic.

## 10. Role of `training/eda.py`

Current understanding:

- It is not considered a central pillar of the final product.
- It likely contains mostly auxiliary / exploratory utilities, especially around visualization of `FlowPre` evolution and related inspections.

### Provisional stance

- Treat it as auxiliary / exploratory for now.
- It may still contain reusable logic worth preserving or extracting later.

## 11. Important Methodological Constraints

### Confirmed decisions / constraints

- Column anonymization and some transformations are real NDA-related publication constraints, not just convenience.
- The current project state includes decisions based on shuffled splits.
- Methodologically, shuffled evaluation is no longer considered ideal for the thesis close.

### Desired methodological direction

The intended direction is to:

- treat `init_temporal_processed_v1` as the canonical official temporal split; F3 has already moved statistical cleaning and canonical raw dataset building onto it
- avoid leakage
- use the minimal drift package already generated for the official split, and extend it later only when it improves the thesis close materially
- fit statistical cleaning only on TRAIN
- keep `val/test` as `flag-only` for learned statistical cleaning in the canonical path
- compare variants under a fair and defendable evaluation framework
- treat the current official `FlowPre` runs observed under the temporal path as the closed evidence base of the FlowPre phase
- use `outputs/models/official/flowpre_finalists/README.md` as the formal closeout reference for the semantic handoff to FlowGen; the winning `rrmse` promotion manifest remains only as inherited technical compatibility for legacy/provisional entrypoints
- compare final dataset combinations with `MLP` under the same seeds and a single frozen base config

## 12. Known Limitations of the Current State

- The repository structure does not yet clearly reflect the canonical story of the project.
- There are multiple versions / iterations of pipelines and experiments.
- It is not yet fully clear which derived datasets should be elevated to official reproducible artifacts.
- Valuable logic is spread across modules, notebooks, and historical scripts.
- Some parts of the methodology that are considered preferable for the thesis close are not yet implemented as the project's definitive workflow.

## 13. Experimental Comparison Space Under Consideration

There is a candidate experimental comparison space for the thesis close, but it is **not closed yet**.

This should be understood as:

- a pending experimental design to be narrowed down
- not a fully implemented comparison matrix
- not a fixed final plan

The final subset may end up being relatively small or larger (for example around 10, 30, 50 or more combinations), depending on time availability and methodological relevance.

### Candidate comparison axes

#### 13.1 Scaling / transformation

`F5` formalizes scaling as a `dataset-level` axis.

For `X`, the supported canonical space now includes:

- `RobustScaler`
- `StandardScaler`
- `QuantileTransformer`
- `MinMaxScaler`
- `FlowPre`-based variants

For `Y`, the supported canonical space now includes:

- `RobustScaler`
- `StandardScaler`
- `QuantileTransformer`
- `MinMaxScaler`

Important note:

- `X` and `Y` may be scaled differently depending on the combination.
- The canonical contract supports a broader `dataset-level` `X × Y` space, but the currently materialized official classical base is the `16` deterministic `X × Y` datasets under `data/sets/official/init_temporal_processed_v1/scaled/`.

#### 13.2 Synthetic data / dataset-level resampling

`F5` models synthetic augmentation as a `dataset-level` axis called `synthetic_policy`.

Current contractual values are:

- `none`
- `flowgen`
- `kmeans_smote`

Status today:

- `none` is usable now
- `flowgen` is supported by contract and the semantic work bases for that phase are now `candidate_1` and `candidate_2`; any residual dependency on a `FlowPre rrmse` manifest belongs only to inherited technical compatibility
- `flowgen` also has a closed local experimental `train_only` branch whose promoted finalist can be evaluated downstream as a local candidate, but it does not replace the official temporal winner
- `kmeans_smote` is implemented as a canonical non-trained `dataset-level` augmenter over canonical non-synthetic bundles already expressed in a chosen `(x_transform, y_transform)` variant

Important rules:

- `synthetic_policy` can only alter `train`
- `val/test` must remain untouched
- synthetic datasets must preserve `synthetic_policy_id`, `is_synth`, source manifests, and counts by split/class
- `KMeans-SMOTE` is modeled under `synthetic_policy`, not as a separate dataset-level balancing axis

#### 13.3 Batch construction / training-time sampling

These are `run-level` axes, not dataset identity.

Candidate options include:

- keep original dataset proportions
- force `1:1:1` class-balanced batches even if epoch endings remain globally imbalanced
- cycle / randomly oversample minority classes to maintain more balanced batches

#### 13.4 Loss weighting

This remains a `run-level` axis.

Candidate options include:

- weighting proportional to the original dataset distribution
- balanced weighting by class proportions
- inverse-frequency weighting to prioritize minority classes

#### 13.5 Selection metric / optimization objective

This remains a `run-level` axis.

Candidate options include:

- maximize `R2`
- minimize `RRMSE`
- evaluate globally
- evaluate using class-averaged criteria

#### 13.6 Additional baseline

- `XGBoost` is being considered as an optional external baseline.
- It is not a central established part of the current project.
- It should be treated as a pending optional reference for the thesis close, depending on time.

## 14. Current State vs Desired Thesis Close

### Current implemented state

- data cleaning and preprocessing exist
- outlier handling exists
- shuffled splits exist in the current project state
- the canonical official split already exists as `data/splits/official/init_temporal_processed_v1/`
- the official split is temporal, contiguous, deterministic, keeps full dates intact, and uses `test` as the most recent block with `val` immediately before it
- the official split already has a manifest, row-level assignments, split summaries, class counts, and a minimal drift package
- the legacy statistical cleaning audit exists as `data/cleaned/official/init_temporal_processed_v1/`
- the current canonical statistical cleaning audit is versioned under `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/`
- the current canonical raw dataset bundle is versioned under `data/sets/official/init_temporal_processed_v1/raw/df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1/`
- `data.sets.load_or_create_raw_splits()` now consumes the official raw bundle by default while preserving a legacy mode
- the current top-level artifacts under `data/splits/` remain legacy input bundles rather than the canonical split contract
- `FlowPre`, `FlowGen`, and `MLP` are already developed significantly
- synthetic realism metrics are already present
- the current best hyperparameters for `FlowPre` and `FlowGen` come from evaluation under the historical shuffled split
- `FlowPre` already has `162` official temporal runs materially present under `outputs/models/official/flow_pre/`
- `22` of those runs belong to `revalidate_v1`, `20` to `explore_v2`, `30` to `explore_v3`, `11` to `explore_v4`, and `79` to the final `reseed`
- the FlowPre reporting layer under `outputs/reports/f6/` can still be reconstructed from artifacts as a snapshot of the first official pass, but it is no longer the current reference for phase status
- the current closeout reference for `FlowPre` is `outputs/models/official/flowpre_finalists/README.md`
- there is now a formal `FlowPre` promotion manifest under the official namespace for the winning `rrmse`, but it should be read only as an inherited technical compatibility artifact rather than as the conceptual or operational main basis of `FlowGen`
- the final fair-comparison protocol for `MLP` with shared seeds and a frozen base config is not closed yet
- `FlowGen` temperature tuning is coded, but it is still experimental
- `FlowGen` is no longer blocked and no longer pending start; the official temporal exploration plus final reseed have already been materialized under `outputs/models/official/flowgen/`
- the historical closeout references for FlowGen exploration are `outputs/models/official/flowgen/campaign_summaries/final_rankings/`, `outputs/models/official/flowgen/FINAL_SELECTION.md`, and `outputs/reports/f6/flowgen_selected.csv`
- the current canonical closeout references for FlowGen are `outputs/models/official/flowgen/campaign_summaries/post_reseed/`, `outputs/models/official/flowgen_finalist/README.md`, and `outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json`
- the final winner family of FlowGen is `E03`
- the unique promoted FlowGen finalist / winner is `flowgen_tpv1_c2_train_e03_seed2468_v1`
- the local `train_only` branch has also been materially closed, with promoted `FlowPre` work bases and a promoted local `FlowGen` finalist under `outputs/models/experimental/train_only/`
- the next active recommendation after the official and local FlowGen closeouts is downstream final comparison with `MLP`, not more FlowGen reseed or open-ended local FlowGen exploration
- `evaluation/` already exists as a canonical lightweight layer for metrics, result normalization, drift loading, seed aggregation, and family-specific comparison
- the current operational FlowPre evaluation surface is already richer than the original family-only layer: it includes `flowgencandidate` and `rrmse_primary`, which were the basis used to close the phase
- that same operational surface now also includes a global reconstruction sanity layer (`global_reconstruction_status`) so descriptive rankings do not keep clearly unhealthy reconstruction cases in the same pool
- the canonical comparison helpers default to `val` and keep `test` blocked unless explicitly enabled
- the training loops no longer compute or persist `test` metrics by default; holdout evaluation requires explicit opt-in
- `MLP` already supports canonical `raw/real space` evaluation when the correct variant scaler is provided

### Desired close / pending work

The project still needs, at minimum:

- retraining under a more solid methodology
- extraction of defendable conclusions
- preparation of final reporting notebooks / tables / figures
- thesis writing

More specifically, the desired close includes:

- deciding which final variants will actually be compared within time constraints
- deciding whether to include additional baselines such as `XGBoost`
- standardizing what is optimized on validation and what remains blocked for test
- extending the official split-aware path from raw bundles toward scaled, evaluation and final-comparison artifacts without leakage
- keeping the already materialized classical scaled datasets under the official namespace and extending the canonical path toward any additional derived datasets that become promotable later
- keeping blocked dataset families modeled by contract until their upstream is revalidated
- using the closed official FlowGen winner as the promoted synthetic source for the downstream final comparison instead of reopening local FlowGen exploration
- deciding explicitly whether the closed local `train_only` FlowGen finalist joins the downstream `MLP` shortlist as an experimental candidate
- freezing a `closure seed set` and an `MLP` base config shared by all final dataset combinations
- running configurations with fixed seeds
- collecting comparable result tables
- performing `ANOVA + post-hoc`, or a robust equivalent, on validation results
- selecting 2-3 finalist configurations
- confirming finalists once on a blocked test set
- closing with a short interpretability / error-analysis section

## 15. Interpretation Rules for Future Work

This context document should distinguish clearly between:

- **current reality**
- **methodological decisions already accepted**
- **candidate experimental space**
- **desired direction / pending work**

It should **not** treat the desired close as if it were already implemented, and it should **not** be read as the final target architecture.

## 16. Open Items Still Not Fully Fixed

- Exact final formula for the `fair` criterion used to select `FlowPre` candidates.
- Final shortlist of experiment combinations for the thesis close.
- Exact balancing techniques to include in the final comparison.
- Whether `SMOTE` / `KMeans-SMOTE` will be included.
- Whether `XGBoost` will be included as an additional baseline.
- Whether the downstream final comparison will include only the official promoted `FlowGen` winner or also the local `train_only` promoted finalist.
- The final `closure seed set` and frozen `MLP` base config for the final comparison.
- Which derived datasets under `data/sets/` should ultimately be promoted to official reproducible artifacts.
