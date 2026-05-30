# F7 Project Closeout Checklist

## Purpose

This document records, in one place, what still needs to happen to close the project well after the `F7` campaign infrastructure and launch path have been completed.

It is intended to cover the final stretch of the thesis, including:

- final campaign completion;
- post-run analytical layer;
- final statistical analysis;
- figures and tables;
- final notebooks;
- thesis writing;
- repo / public-surface polish;
- defense-oriented extras with strong methodological signal.

This is not a speculative wishlist.

It is a practical closeout checklist, ordered by:

- what is genuinely required to finish the project;
- what gives high methodological ROI;
- what provides extra signal for a high-grade thesis / strong ETH-style presentation.

## Scope

This checklist assumes the following are already materially closed or substantially closed:

- `F1` to `F6`;
- `F7` pre-campaign hardening;
- `FlowPre` and `FlowGen` operational closeout;
- official temporal split and anti-leakage setup;
- campaign/runner/lineage hardening;
- analysis grammar freeze;
- launch-readiness layer;
- artifact dedup without loss of analytical surface.

What this document covers is everything that still sits between:

- “the final campaign is running or finished”
- and “the TFG is closed, defendable, reproducible, and presentation-ready”.

## Current snapshot

At the time of writing, the project is already strong on:

- campaign infrastructure;
- comparability contracts;
- per-run artifacts;
- multi-seed lineage aggregation;
- interpretability persistence;
- anti-leakage methodology;
- final launch readiness.

The main remaining risk is no longer infrastructure.

The main remaining risk is synthesis:

- converting the experimental surface into a rigorous final analysis;
- converting that analysis into a thesis-quality narrative;
- and making sure the final written story reflects the actual canonical pipeline and decisions.

## Reading guide

This checklist is split into three layers:

- `Must-have`
- `High-ROI`
- `Nice-to-have`

Inside each layer, items are grouped by area:

- campaign completion and operational closeout;
- analytical layer and statistics;
- results presentation;
- thesis writing;
- repo / reproducibility polish;
- defense extras.

Each item should be read as having four implicit fields:

- `goal`
- `deliverable`
- `why it matters`
- `done when`

## Must-have

These are the items that should be treated as required for a serious and defendable closeout.

### 1. Final campaign completes cleanly

Goal:

- finish the full `30 + 30 + 30 + 30` chain;
- leave all four campaigns in a clean terminal state.

Deliverables:

- `campaign_closeout.json` for:
  - `f7_campaign_v1`
  - `f7_campaign_extension1_v1`
  - `f7_campaign_extension2_v1`
  - `f7_campaign_extension3_v1`
- refreshed:
  - `campaign_report.json/.md`
  - `lineage_report.json/.md`
  - `storage_footprint` report
  - readiness snapshot after completion

Why it matters:

- no final analysis should be written on an incomplete pooled lineage if the project framing is explicitly based on the full closure seed panel.

Done when:

- all campaigns are `closed_success`;
- lineage is `lineage_pool_ready = true`;
- the expected `120` seeds are fully represented;
- no extension remains `open` or `in_progress`.

### 2. Freeze the final analysis manifest

Goal:

- make the final statistical analysis reproducible and not notebook-dependent.

Deliverables:

- one canonical `analysis_manifest` file;
- versioned analysis spec for the final run of the statistics layer;
- frozen references to:
  - root campaign id;
  - included campaigns;
  - included seeds;
  - endpoints;
  - contrasts;
  - bootstrap settings;
  - model formulas;
  - plot list;
  - output tables.

Why it matters:

- this is the bridge between “campaign outputs exist” and “the final analysis can be rerun exactly”.

Done when:

- the analysis can be regenerated from campaign outputs plus manifest only, without hidden notebook logic.

### 3. Materialize the final analytical layer post-run

Goal:

- convert the final campaign outputs into canonical analytical panels used by the statistics layer.

Deliverables:

- final campaign-level and lineage-level panels for:
  - predictive metrics;
  - detailed `per_class` metrics;
  - `worst_class`;
  - dispersion / Simpson diagnostics;
  - interpretability aggregates and stability outputs.

Why it matters:

- the final thesis should not depend on reading hundreds of per-run files ad hoc.

Done when:

- there is a small set of canonical analysis-ready data products that fully feed tables, statistics, and plots.

### 4. Run the final statistical analysis

Goal:

- execute the methodology already frozen in the `F7` statistical plan.

Deliverables:

- descriptive multi-seed summaries;
- paired / blocked comparisons;
- mixed-effects analysis;
- ANOVA support tables;
- directed post-hoc contrasts;
- bootstrap confidence intervals;
- Simpson-focused outputs;
- interpretability stability outputs.

Why it matters:

- this is the scientific core of the thesis close.

Done when:

- every final claim in the thesis can be traced to a canonical output produced by this analysis layer.

### 5. Produce the final comparative tables

Goal:

- make final decision-making and thesis writing simple.

Deliverables:

- one master result table by structural variant;
- one shortlist/finalist table;
- one cost-aware comparison table;
- one interpretability stability summary table.

Minimum expected columns should include:

- variant identifiers;
- `val` metrics;
- `test` metrics;
- `macro`;
- `per_class` aggregate summaries;
- `worst_class`;
- dispersion / Simpson diagnostics;
- runtime / cost;
- seed stability indicators.

Why it matters:

- without these tables, the final writing becomes diffuse and manually assembled.

Done when:

- the thesis results chapter can be written mostly from these tables.

### 6. Build the core figures of the TFG

Goal:

- distill the final findings into a small, strong figure set.

Deliverables:

- figure for campaign / comparison overview;
- figure for main predictive ranking;
- figure for Simpson-risk / aggregate-vs-classwise tension;
- figure for interpretability stability;
- figure for cost vs performance tradeoff.

Why it matters:

- strong figures dramatically improve both thesis readability and defense quality.

Done when:

- the key story of the thesis can be explained through a compact figure set without needing to inspect raw outputs.

### 7. Write the final methods and results sections coherently

Goal:

- align the thesis narrative with the actual canonical repo and final experimental design.

Deliverables:

- methods chapter final revision;
- results chapter final revision;
- discussion chapter final revision;
- limitations section final revision;
- conclusion and recommendation section.

Why it matters:

- a technically strong repo can still yield a weak thesis if the written argument is loose or inconsistent.

Done when:

- the written narrative matches:
  - the actual split policy;
  - the actual campaign design;
  - the actual statistical layer;
  - the actual meaning of `val` vs `test`.

### 8. Explicit limitations section

Goal:

- state clearly what the project does and does not establish.

Deliverables:

- a dedicated limitations subsection that explicitly covers:
  - one official temporal split only;
  - robustness by seeds vs robustness across alternative temporal cuts;
  - uncertainty not being full population uncertainty;
  - interpretability limits under correlated features;
  - the status of `XGBoost` as baseline rather than full second study.

Why it matters:

- this increases credibility and is strongly aligned with rigorous statistical presentation.

Done when:

- the thesis cannot be accused of overclaiming what the evidence supports.

## High-ROI

These are not as structurally mandatory as the previous block, but they offer very strong payoff for relatively modest effort.

### 9. Contrast registry as explicit thesis artifact

Goal:

- make confirmatory vs exploratory reasoning explicit.

Deliverables:

- one `contrast_registry` artifact dividing comparisons into:
  - confirmatory;
  - secondary;
  - exploratory.

Why it matters:

- this prevents post-hoc storytelling and gives a much cleaner methodological impression.

Done when:

- the final analysis and the thesis text both refer to the same registry.

### 10. Practical significance layer

Goal:

- avoid telling the story only in terms of statistical significance.

Deliverables:

- explicit minimum practical effect interpretation;
- or at least a structured discussion of when deltas are operationally negligible.

Why it matters:

- this is a high-value methodological signal and helps distinguish strong analysis from p-value-driven analysis.

Done when:

- final comparisons include practical relevance, not only superiority by metric or significance.

### 11. Strong Simpson framing as a signature contribution

Goal:

- turn the aggregate-vs-classwise issue into one of the strongest intellectual contributions of the thesis.

Deliverables:

- explicit subsection in methods;
- explicit results subsection;
- at least one dedicated figure;
- one dedicated table;
- one explicit discussion paragraph explaining why aggregate-only reading is insufficient here.

Why it matters:

- this is likely one of the highest-signal parts of the entire project.

Done when:

- the thesis makes it impossible to miss why Simpson-type effects matter for this problem.

### 12. Interpretability as stability, not just feature ranking

Goal:

- present interpretability as a structured and reproducible layer.

Deliverables:

- seed-level stability summaries;
- top-k overlap or frequency summaries;
- family-level or semantic-group discussion where appropriate;
- a small interpretation section that is careful about correlated features.

Why it matters:

- interpretability is much stronger when treated as a stable phenomenon rather than a single ranking screenshot.

Done when:

- the thesis can argue not only “what features appear important”, but also “how stable that importance is”.

### 13. Cost-performance discussion

Goal:

- connect predictive quality to runtime / practical feasibility.

Deliverables:

- one explicit section or subsection comparing:
  - quality;
  - robustness;
  - cost;
  - operational plausibility.

Why it matters:

- this is highly relevant for an applied industrial thesis and prevents the conclusion from sounding purely academic.

Done when:

- there is a clear recommendation of what should be preferred in near-term operational use.

### 14. Negative results / what did not help

Goal:

- explicitly show what was tested but did not justify itself.

Deliverables:

- concise subsection on:
  - methods or branches that did not improve enough;
  - why they are not being promoted;
  - what this teaches about the problem.

Why it matters:

- strong theses often become more credible when they include disciplined negative findings.

Done when:

- the reader sees that the final recommendations emerged through comparison, not only selective reporting.

## Nice-to-have

These are optional from a strict completion perspective, but can still add signal if time permits.

### 15. Defense-ready appendix pack

Goal:

- make supporting detail easy to retrieve during the defense.

Deliverables:

- compact appendix or annex with:
  - campaign ids;
  - seed panel ids;
  - contract ids;
  - final analysis spec reference;
  - glossary of final variant naming.

Why it matters:

- this reduces friction when answering detailed committee questions.

### 16. One polished result notebook per major axis

Goal:

- keep the final visual exploration layer usable and thesis-friendly.

Deliverables:

- one notebook for predictive comparisons;
- one notebook for Simpson/classwise analysis;
- one notebook for interpretability/stability.

Constraints:

- they should consume canonical post-run panels;
- they should not contain hidden analytical logic that does not exist elsewhere.

### 17. One master “project close” README for local use

Goal:

- reduce future confusion if the project is reopened later.

Deliverables:

- a concise local README that points to:
  - final campaign outputs;
  - final reports;
  - final analysis entrypoint;
  - final thesis artifacts.

### 18. Final public-surface sanity pass

Goal:

- make sure the repo is coherent as a public-facing engineering artifact.

Deliverables:

- quick pass over:
  - `docs/`
  - public-safe configs
  - public-safe rationale docs
  - public-safe finalists metadata

Why it matters:

- useful if the repo is shown in interviews, committee review, or later public reference.

## ETH-signal extras

These are not necessarily separate tasks, but they are presentation standards worth aiming for across the final material.

### A. Explicit estimands

Every important comparison should ideally be expressible as:

- “we estimate the effect of X on Y under comparable structural conditions”.

Not just:

- “we compared a lot of metrics and this one looked good”.

### B. Confirmatory vs exploratory separation

The final text should distinguish clearly between:

- claims that were structurally pre-specified;
- claims that are descriptive but exploratory.

### C. Practical significance

The final thesis should repeatedly avoid a “significance only” tone.

### D. Limits of uncertainty

The final statistical section should explicitly distinguish:

- seed-level uncertainty / algorithmic robustness;
- temporal generalization under the one official split;
- what is not identified as full population uncertainty.

### E. No silent metric drift

All final tables and figures should use the same canonical metric semantics already fixed in the repo.

### F. No silent test-driven selection

The final shortlist / finalist logic must remain validation-led in writing and in artifacts.

## Recommended execution order from now on

The most practical order to finish well is:

1. let the full final campaign finish;
2. regenerate final campaign and lineage reports;
3. freeze final analysis manifest and contrast registry;
4. materialize the final analysis-ready panels;
5. run the final statistical layer;
6. generate canonical final tables and figures;
7. write / revise final methods, results, discussion, limitations, conclusions;
8. polish notebooks and appendices;
9. perform final public-surface sanity pass.

## Exit criteria for project close

The project should be considered closed only when all of the following hold:

- final campaign complete;
- lineage complete and pooled;
- final analysis manifest frozen;
- final statistical layer run and archived;
- final tables and figures generated;
- final thesis text aligned with actual repo methodology;
- explicit limitations section written;
- final recommendation stated;
- minimal public / local surface sanity pass completed.

## Minimal closeout checklist

If time becomes tight, the minimum serious closeout is:

- finish the campaign;
- run the final analysis;
- generate final tables and figures;
- write methods/results/discussion/limitations/conclusion;
- preserve reproducibility of the final analysis.

Everything else is secondary to that.

## Final note

The project is already strong on engineering and methodological infrastructure.

The remaining work with the biggest payoff is not “more experimentation”.

It is:

- disciplined synthesis;
- rigorous final analysis;
- and a written thesis that makes the structure, tradeoffs, and limits of the evidence easy to defend.
