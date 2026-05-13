# Implementation Status

## Estado actual tras F1, F2, F3, F4, F5 y el cierre de FlowPre y FlowGen

El proyecto ya ha cerrado:

- `F1`: canon documental minimo sobre que es canonico, que esta pendiente y que queda como experimental o historico.
- `F2`: split temporal oficial y contrato de split canonico.
- `F3`: preparacion split-aware y anti-leakage para cleaning y dataset building.
- `F4`: capa canonica de evaluacion para metricas, normalizacion de resultados, agregacion por seeds y comparacion basica por familia.
- `F5`: canonizacion de derivados + contrato experimental base.
- cierre operativo de `FlowPre` bajo el split temporal oficial.
- cierre operativo de `FlowGen` con winner final promovido.

## Superficie visible y frontera publico/local

- `docs/repo_visibility_matrix.md` es la referencia practica para distinguir material mostrable/publicable, local/privado/NDA y piezas mixtas.
- en esta iteracion, la prioridad no es congelar `F7`, sino endurecer la frontera NDA, aclarar la superficie visible del repo y mejorar la trazabilidad documental.
- las tracked copies de `config/type_mapping.yaml`, `config/column_groups.yaml` y `config/cleaning_contract.yaml` pasan a leerse como copias public-safe o contractuales del repo; las versiones operativas privadas viven bajo `config/local/` y quedan fuera de git.
- la semantica sensible del arranque de preprocess ya no debe vivir hardcodeada en `data/cleaning.py`; ahora pasa por `config/cleaning_contract.yaml` + `config/local/cleaning_contract.yaml` para separar logica publica de detalles operativos privados del raw.
- `data/raw/`, `data/processed/`, `data/cleaned/`, `data/splits/`, `data/sets/` y los artefactos generados bajo `outputs/` deben tratarse como superficies locales para git/publicacion aunque sigan siendo valiosas en local.

## Estado metodologico actual

- el split oficial canonico es `init_temporal_processed_v1`
- vive en `data/splits/official/init_temporal_processed_v1/`
- usa una politica temporal, contigua, determinista y sin partir fechas
- `test` es el bloque mas reciente y `val` es el bloque inmediatamente anterior
- `val/test` mantienen la distribucion cronologica natural; no se rebalancean artificialmente a `1:1:1`
- el dataset building canonico ya consume el split oficial
- la limpieza estadistica aprendida ya se fittea solo en `train`
- en el camino canonico actual, el drop efectivo por cleaning learned solo puede ocurrir en `train`, con cap global `<= 1%`, y `val/test` quedan en `flag-only`
- `evaluation/` ya es la capa canonica para metricas reutilizables, resultados normalizados, drift, agregacion por seeds y comparacion basica
- la comparacion sigue siendo por familia; no existe todavia un ranking global canonico del proyecto
- `test` queda bloqueado por defecto como superficie de seleccion en las APIs de comparacion/agregacion
- `training/train_*` ya no calculan ni guardan metricas de `test` por defecto; requieren opt-in explicito
- `MLP` ya dispone de evaluacion canonica en `raw/real space` cuando se proporciona el scaler correcto de la variante
- `training/optuna_mlp.py` conserva solo un papel opcional de search workflow; sus helpers reutilizables de evaluacion pasan a consumirse desde `evaluation/`
- el canon operativo del repo llega ya hasta `data/sets/official/init_temporal_processed_v1/scaled/` para los `16` datasets clasicos `X classical x Y classical`
- `data/sets/scaled_sets/` queda como namespace legacy/historico; los bundles `FlowPre-based` persistidos ahi no pasan a canon antes de `FlowGen`

## Estado real de FlowPre

- `FlowPre` ya no esta solo en estado `implementado`: la fase experimental y de seleccion esta cerrada
- `outputs/models/official/flow_pre/` contiene `162` runs oficiales completas:
  - `22` de `revalidate_v1`
  - `20` de `explore_v2`
  - `30` de `explore_v3`
  - `11` de `explore_v4`
  - `79` del reseed final
- la auditoria actual de filesystem no detecta ninguna run oficial incompleta; todas las carpetas oficiales contienen `run_manifest.json`, `results.yaml` y `metrics_long.csv`
- el reseed final a `4` seeds por `cfg_signature` quedo cerrado en `79/79` runs completas, `0` incompletas y `0` pendientes
- el evaluador actual ya expone una capa ampliada y operativa para `FlowPre`:
  - vistas por familia `rrmse`, `mvn` y `fair`
  - lente adicional `flowgencandidate`
  - lectura operativa `rrmse_primary`
  - capa global de sanidad de reconstruccion `global_reconstruction_status`
- esa capa evaluativa ya se ha usado para cerrar `FlowPre`; no debe leerse ya como una fase abierta
- el cierre oficial queda trazado por:
  - `outputs/reports/f6_reseed_topcfgs_v4_continue/resume_summary.md`
  - `outputs/models/official/flowpre_finalists/README.md`
  - y la carpeta `outputs/models/official/flowpre_finalists/`

## Finalistas cerrados de FlowPre

- `rrmse_primary`
  - cfg: `hf256|l3|rq1x3|frq3|lr1e-4|mson|skoff`
  - seed ganadora: `5678`
  - run final: `flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1`
- `mvn`
  - cfg: `hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff`
  - seed ganadora: `5678`
  - run final: `flowpre_rrmse_tpv1_rq5_seed5678_v1`
- `fair`
  - cfg: `hf192|l3|rq1x6|frq6|lr1e-3|mson|skoff`
  - seed ganadora: `5678`
  - run final: `flowprex4_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed5678_v1`
- `flowgencandidate_priorfit`
  - cfg: `hf192|l3|rq1x6|frq6|lr1e-3|mson|skoff`
  - seed ganadora: `9101`
  - run final: `flowprers1_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed9101_v1`
- `flowgencandidate_robust`
  - cfg: `hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff`
  - seed ganadora: `2468`
  - run final: `flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1`
- `flowgencandidate_hybrid`
  - cfg: `hf256|l4|rq1x5|frq5|lr1e-3|mson|skoff`
  - seed ganadora: `2468`
  - run final: `flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1`

## Materializacion operativa del cierre

La carpeta `outputs/models/official/flowpre_finalists/` contiene:

- `rrmse/`
- `mvn/`
- `fair/`
- `candidate_1/`
- `candidate_2/`

Cada carpeta contiene:

- la run ganadora copiada integra
- un `RATIONALE.md` con el proceso de seleccion y tradeoffs

Rol vigente de cada grupo:

- `rrmse`, `mvn` y `fair` son finalistas descriptivos de `FlowPre`. Su funcion activa es escalar el dataset y servir como upstreams/scalers especializados para derivar variantes de dataset. No son las bases principales de entrenamiento de `FlowGen`.
- `candidate_1` y `candidate_2` son las dos bases reales de trabajo elegidas para arrancar y desarrollar la fase `FlowGen`.

## Handoff tecnico heredado y handoff semantico a FlowGen

- existe un contrato tecnico heredado en implementaciones legacy/provisionales de `FlowGen` que referencia un manifest promovido de rama `rrmse`
- ese manifest existe y queda formalizado en:
  - `outputs/models/official/flow_pre/flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1/flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1_promotion_manifest.json`
- el `source_id` asociado a ese contrato tecnico heredado es:
  - `flowpre__rrmse__init_temporal_processed_v1__v1`
- ese detalle tecnico heredado no convierte a `rrmse`, `mvn` o `fair` en las bases principales de entrenamiento de `FlowGen`; esos tres artefactos siguen siendo scalers/upstreams especializados para dataset scaling y derivacion de variantes
- `candidate_1` y `candidate_2` son las dos bases reales de trabajo de la fase `FlowGen`
- una vez `FlowGen` quede cerrado y existan outputs promovidos, `candidate_1` y `candidate_2` pasaran a ser artefactos historicos de trazabilidad y dejaran de ser artefactos activos/canonicos de uso
- por tanto, el handoff semantico real hacia `FlowGen` no es `rrmse`, sino `candidate_1` y `candidate_2`

## Estado real de FlowGen

- `outputs/models/official/flowgen/` ya no representa una fase vacia o pendiente de arranque: contiene la exploracion temporal oficial materializada de `FlowGen` y el reseed final ya cerrado
- el estado observado actual son `50` runs oficiales completas comparables, mas las bases oficiales `flowgen_tpv1_c1_base_seed9101` y `flowgen_tpv1_c2_base_seed2468`
- ese total se reparte en:
  - `30` runs de exploracion oficial pre-reseed
  - `20` runs del reseed final
- la exploracion materializada cubre `v1`, `v2` y `v3`
- el ranking historico v6-style sigue vivo como trazabilidad bajo `outputs/models/official/flowgen/campaign_summaries/rankings/`
- la capa historica de cierre exploratorio sigue viviendo en `scripts/f6_flowgen_rank_official_v2.py`
- sus artefactos historicos se materializan bajo `outputs/models/official/flowgen/campaign_summaries/final_rankings/`
- la capa canonica de agregacion post-reseed vive en `outputs/models/official/flowgen/campaign_summaries/post_reseed/`
- el winner de familia tras el post-reseed es `E03`
- el finalista oficial unico de `FlowGen` ya esta materializado en:
  - `outputs/models/official/flowgen_finalist/`
  - con run final promovida:
    - `flowgen_tpv1_c2_train_e03_seed2468_v1`
- `outputs/reports/f6/flowgen_selected.csv` y `outputs/models/official/flowgen/FINAL_SELECTION.md` deben leerse ya como cierre exploratorio pre-reseed, no como estado final vigente
- la exploracion local de nuevas configs y el reseed de `FlowGen` se consideran cerrados
- la siguiente fase activa recomendada ya es el cierre downstream con `MLP`

## Estado real de la rama experimental train_only

- la rama `train_only` ya no es solo una hipotesis metodologica: esta materializada y cerrada localmente bajo `outputs/models/experimental/train_only/`
- esta rama permanece fuera del canon `official/` y no sustituye el winner temporal oficial del proyecto
- su finalidad vigente es servir como candidata experimental aguas abajo del cierre con `MLP`

### `FlowPre train_only`

- el cierre local de `FlowPre train_only` ya esta materializado en:
  - `outputs/models/experimental/train_only/flowpre_finalists/`
- esa carpeta contiene dos bases de trabajo finales para la rama:
  - `candidate_trainonly_1`
  - `candidate_trainonly_2`
- esas dos bases son los dos priors finales cerrados para arrancar `FlowGen train_only`; no son winners canonicos del proyecto

### `FlowGen train_only`

- la exploracion, confirmaciones, reseed y seleccion final de `FlowGen train_only` tambien quedaron cerrados localmente
- el finalista local unico vigente ya esta materializado en:
  - `outputs/models/experimental/train_only/flowgen_finalist/`
- la run promovida vigente es:
  - `flowgen_trainonly_tpv1_ct1_reseedfinal_r3a2_t06_clip125_seed15427_v1`
- este finalista local debe leerse como:
  - candidato experimental downstream para balancing/augmentation de `train`
  - no como winner temporal oficial
  - no como reapertura de `F6b`
- la siguiente decision relevante para esta rama ya no es explorar mas `FlowGen`, sino decidir si entra o no en la shortlist de `F7`

## Fases completadas

- `F1. Canon documental y frontera`
- `F2. Split temporal oficial y contrato de split`
- `F3. Preparacion split-aware y anti-leakage`
- `F4. Evaluation canonica`
- `F5. Canonizacion de derivados + contrato experimental`
- `F6a. Cierre de FlowPre`
- `Pre-F5. Correccion metodologica y operativa corta`

## Decisiones cerradas hasta ahora

- los mejores hiperparametros actuales de `FlowPre` y `FlowGen` son priors operativos heredados del split aleatorio, no canon final temporal
- `FlowGen temperature tuning` queda fuera del camino canonico por ahora y se trata como experimental
- en codigo todavia existe una referencia heredada/legacy a `FlowPre rrmse` en ciertos entrypoints tecnicos de `FlowGen`, pero no debe leerse como la base conceptual ni operativa principal de la fase generativa
- el split oficial canonico se fija antes de F3+
- el split oficial canonico es `init_temporal_processed_v1`
- la fuente de F2 es `data/processed/df_processed.csv` bajo el estatus `processed_pre_statistical_cleaning`
- la politica de split es temporal, contigua, determinista y sin partir una misma fecha
- `val/test` no se rebalancean artificialmente; la comparabilidad entre clases vendra de metricas por clase y macro-average
- la superficie de `X` e `y` del bundle oficial se congela frente al camino legacy:
  mismos nombres, mismo orden, mismo papel de `type` y `post_cleaning_index`, mismos renames y rescalados
- `split_id`, `split`, `date`, `source_row_number` y `split_row_id` no entran en `X` ni en `y`; viven en `removed` y manifests
- `post_cleaning_index` queda como clave tecnica opaca, unica y estable dentro del bundle oficial
- `sum_chem` / `sum_phase` quedan documentados como `quality filter` de dominio dentro del preprocess global, no como cleaning learned
- la politica de cleaning se versiona por separado del `split_id`
- el raw bundle oficial se versiona por separado del `split_id`
- `data.sets.load_or_create_raw_splits()` mantiene el contrato downstream y pasa a consumir por defecto el raw bundle oficial versionado
- el cierre de `FlowPre` ya no esta abierto a nuevas cfgs o seeds dentro de esta fase

## Artefactos canonicos creados o consolidados

Rutas canonicas de datos:

- `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/`
- `data/sets/official/init_temporal_processed_v1/raw/df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1/`
- `data/sets/official/init_temporal_processed_v1/scaled/`

Rutas canonicas de modelos y cierre:

- `outputs/models/official/flow_pre/`
- `outputs/models/official/flowpre_finalists/`
- `outputs/models/official/flowgen/`
- `outputs/models/official/flowgen/campaign_summaries/final_rankings/`
- `outputs/models/official/flowgen/campaign_summaries/post_reseed/`
- `outputs/models/official/flowgen_finalist/`

Rutas canonicas de seleccion y reporting operativo:

- `outputs/reports/f6/flowgen_selected.csv`
- `outputs/models/official/flowgen/FINAL_SELECTION.md`
- `outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json`
- `outputs/models/official/flowgen_finalist/README.md`

Rutas canonicas de split:

- `data/splits/official/init_temporal_processed_v1/`

## Legacy / historico explicito

- `data/splits/df_X_input_init.csv`
- `data/splits/df_y_input_init.csv`
- `data/splits/df_removed_input_init.csv`
- `data/cleaned/df_cleaned_init.csv`
- `data/sets/df_input/`
- `data/cleaned/official/init_temporal_processed_v1/` como politica antigua `or_drop_holdout_v1`
- `data/sets/official/init_temporal_processed_v1/raw/df_input/` como raw bundle antiguo ligado a esa politica
- `data/sets/scaled_sets/` como namespace fisico legacy/historico para derivados previos y bundles `FlowPre-based` aun no promovidos
- `data/sets/augmented_scaled_sets/`
- notebooks operativas e historicas
- `training_scripts/` como capa historica o semioperativa
- `outputs/reports/f6/`, `outputs/reports/f6_explore_v2/`, `outputs/reports/f6_explore_v2_results/`, `outputs/reports/f6_explore_v3/` y `outputs/reports/f6_explore_v4/` como snapshots historicos salvo nota explicita de vigencia

## Pendiente a partir de ahora

- definir la shortlist final de combinaciones/datasets para `F7`
- ejecutar campañas comparables con `MLP` sobre esa shortlist
- decidir si el finalista local `train_only` entra en la comparacion downstream final
- rehacer notebooks finales
- preparar repo publico limpio

Nota de fase:

- `F7` sigue pendiente y no se congela en esta iteracion documental.
- antes de planificar `F7`, el repo deja fijadas la frontera NDA, la superficie visible y los artefactos de señal/trazabilidad.

## Open decisions reales

- shortlist final de combinaciones a comparar con `MLP`
- inclusion o no del finalista local `train_only` de `FlowGen` dentro de `F7`
- inclusion o no de `XGBoost`
- inclusion o no de `SMOTE` / `KMeans-SMOTE`
- `closure seed set` final
- `MLP base config` final congelado
- que datasets derivados y sintéticos se promocionaran finalmente a artefactos oficiales

## Siguiente paso recomendado

Cierre downstream tras el cierre final de `FlowGen official` y de la rama local `train_only`:

- usar `candidate_1` y `candidate_2` solo como bases historicas de procedencia de la exploracion ya materializada
- tratar cualquier referencia tecnica al promotion manifest `rrmse` solo como compatibilidad heredada de entrypoints legacy/provisionales
- usar `data/sets/official/init_temporal_processed_v1/scaled/` como unica base canonica de datasets clasicos
- usar `outputs/models/official/flowpre_finalists/README.md` y `outputs/reports/f6_reseed_topcfgs_v4_continue/resume_summary.md` como referencias vigentes del cierre de `FlowPre`
- usar `outputs/models/official/flowgen/campaign_summaries/post_reseed/`, `outputs/models/official/flowgen_finalist/README.md` y `outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json` como referencias vigentes del cierre final de `FlowGen`
- usar `outputs/models/experimental/train_only/flowpre_finalists/README.md` y `outputs/models/experimental/train_only/flowgen_finalist/README.md` como referencias vigentes de la rama local `train_only`
- tratar `outputs/reports/f6/`, `outputs/reports/f6_explore_v2/`, `outputs/reports/f6_explore_v2_results/`, `outputs/reports/f6_explore_v3/` y `outputs/reports/f6_explore_v4/` como snapshots historicos
- tratar `flowgen_tpv1_c2_train_s01_e38_softclip_seed2468_v2` solo como ancla historica pre-reseed
- tomar `flowgen_tpv1_c2_train_e03_seed2468_v1` como winner/promoted finalist vigente de `FlowGen`
- decidir explicitamente si `flowgen_trainonly_tpv1_ct1_reseedfinal_r3a2_t06_clip125_seed15427_v1` entra como candidato experimental de `F7`
- ejecutar campañas comparables con `MLP` usando `config/mlp_closure_base_v1.yaml`, `closure_5x_v1`, seleccion en `val` y `test` bloqueado por defecto cuando corresponda

## Roadmap ahead

- `F4. Evaluation canonica`
  Cerrada. `evaluation/` pasa a ser la capa canonica de metricas, drift, normalizacion de resultados, agregacion por seeds y comparacion basica por familia.

- `F5. Canonizacion de derivados + contrato experimental`
  Cerrada. Se separan `dataset-level axes` de `run-level axes`, se fija la gramatica del espacio soportado, se materializan los `16` clasicos bajo el arbol oficial y se congela el contrato comparable de `MLP`.

- `F6. Retune local de FlowPre/FlowGen`
  Subfase `FlowPre`: cerrada. Existen `162` runs oficiales completas, reseed final cerrado a `4` seeds por cfg y finalistas materializados en `outputs/models/official/flowpre_finalists/`. `rrmse`, `mvn` y `fair` quedan como scalers/upstreams especializados; `candidate_1` y `candidate_2` quedan como bases reales de trabajo de `FlowGen`.
  Subfase `FlowGen`: cerrada. Existen `50` runs oficiales comparables entre exploracion y reseed final, una capa historica de ranking exploratorio bajo `scripts/f6_flowgen_rank_official_v2.py`, una capa canonica de agregacion post-reseed bajo `outputs/models/official/flowgen/campaign_summaries/post_reseed/`, y un finalista unico materializado en `outputs/models/official/flowgen_finalist/` con winner final `flowgen_tpv1_c2_train_e03_seed2468_v1`. Si algun entrypoint tecnico sigue referenciando un promotion manifest `rrmse`, debe leerse solo como compatibilidad heredada.
  Rama experimental `train_only`: cerrada localmente. Existen bases finales de `FlowPre train_only` en `outputs/models/experimental/train_only/flowpre_finalists/` y un finalista local unico de `FlowGen train_only` en `outputs/models/experimental/train_only/flowgen_finalist/`. Esta rama no reabre `official/`; queda como candidata experimental para `F7`.

- `F7. Cierre experimental principal con MLP`
  Ejecutar la comparacion final de combinaciones de datasets con `MLP`, mismas seeds y mismo base config, seleccionar en `val` y confirmar una sola vez en `test`.

- `F8. Notebooks finales del TFG`
  Reconstruir notebooks limpias de cierre que consuman artefactos canonicos ya generados y no funcionen como pipeline operativo.

- `F9. Repo publico limpio`
  Preparar la version publicable del repo, separar material local/sensible y dejar documentacion final reproducible.
