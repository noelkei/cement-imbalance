# Artifact Lineage

Lineage operativo de artefactos entre fases cerradas y la siguiente fase pendiente.

Formato:

- `input`: que superficie entra a la fase;
- `proceso`: que transformacion o cierre ocurre;
- `output`: que artefactos principales salen;
- `referencia vigente`: donde mirar hoy si esa fase ya esta cerrada;
- `estatus`: `cerrada` o `pendiente`.

| fase | input | proceso | output | referencia vigente | estatus |
| --- | --- | --- | --- | --- | --- |
| `F2` split temporal oficial | `data/processed/df_processed.csv` | construccion del split temporal contiguo, sin partir fechas y con drift minimo asociado | `data/splits/official/init_temporal_processed_v1/` | `data/splits/official/init_temporal_processed_v1/manifest.json` | `cerrada` |
| `F3` cleaning split-aware | `F2` split oficial + source `processed_pre_statistical_cleaning` | fit de cleaning learned en `train`, holdout en `flag-only`, guardrail de drop | `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/` | `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/manifest.json` | `cerrada` |
| `F5` raw/scaled canonical bundles | `F3` cleaning oficial + contrato experimental base | materializacion del raw bundle canonico y de los `16` datasets clasicos `X x Y` | `data/sets/official/init_temporal_processed_v1/raw/...` y `data/sets/official/init_temporal_processed_v1/scaled/` | `config/closure_contract_v1.yaml` y manifests de `data/sets/official/init_temporal_processed_v1/` | `cerrada` |
| `F6a` cierre de `FlowPre` | split oficial + bundle canonico + exploraciones/reseed oficiales | evaluacion por lentes, filtro de reconstruccion y materializacion de finalistas | `outputs/models/official/flowpre_finalists/` + export tracked ligero en `config/finalists/official_flowpre_*.yaml` | `outputs/models/official/flowpre_finalists/README.md`, `outputs/reports/f6_reseed_topcfgs_v4_continue/resume_summary.md` y `docs/finalists/official_flowpre.md` | `cerrada` |
| `F6b` cierre de `FlowGen` | `candidate_1` / `candidate_2` de `FlowPre` + exploracion oficial + reseed final | agregacion post-reseed, eleccion de familia ganadora y promocion del winner unico | `outputs/models/official/flowgen_finalist/` + export tracked ligero en `config/finalists/official_flowgen_winner.yaml` | `outputs/models/official/flowgen_finalist/README.md`, `RATIONALE.md`, `flowgen_final_selection_manifest.json` y `docs/finalists/official_flowgen.md` | `cerrada` |
| rama experimental `train_only` | split oficial + politica `monitoring_policy="train_only"` + cierres locales de `FlowPre` y `FlowGen` | seleccion de bases locales `FlowPre`, exploracion/reseed local de `FlowGen` y promocion de un finalista local downstream | `outputs/models/experimental/train_only/flowpre_finalists/` y `outputs/models/experimental/train_only/flowgen_finalist/` + export tracked ligero en `config/finalists/trainonly_*.yaml` | `docs/experimental_train_only.md`, `outputs/models/experimental/train_only/flowpre_finalists/README.md`, `outputs/models/experimental/train_only/flowgen_finalist/README.md`, `docs/finalists/trainonly_flowpre.md` y `docs/finalists/trainonly_flowgen.md` | `cerrada/local` |
| `F7` cierre principal con `MLP` | datasets clasicos oficiales + upstreams cerrados de `FlowPre` / `FlowGen` + reglas de evaluacion canonica | comparacion final pendiente de definir y ejecutar | por decidir | pendiente | `pendiente` |

## Lectura practica

- `F2 -> F3 -> F5` fijan la base de datos y derivados canonicos usados aguas abajo.
- `F6a` cierra el upstream transformador.
- `F6b` cierra el upstream generativo.
- `config/finalists_registry.yaml` y `docs/finalists_registry.md` son la vista tracked ligera que resume esos cierres sin subir los artifacts pesados.
- la rama `train_only` queda como lineage paralelo local que puede alimentar `F7` como candidato experimental, pero no reabre el camino canónico.
- `F7` todavia no tiene shortlist ni artefactos finales congelados, por eso su referencia vigente sigue siendo `pendiente`.

## Nota de visibilidad

La existencia en este lineage no implica que el artefacto sea publicable.
La frontera mostrable/local se regula en `docs/repo_visibility_matrix.md`.
