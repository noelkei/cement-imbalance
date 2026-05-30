# F7 Block 13 Closeout

## Scope

El bloque `13` queda cerrado como la capa canónica de:

- materialización de campañas;
- ejecución y reanudación por trial;
- campañas de extensión por seeds;
- estado y reporting por campaña;
- pooling por lineage;
- y superficie upstream lista para el análisis estadístico posterior.

Este cierre no introduce todavía la capa analítica final del bloque `14`. Su objetivo es dejar el runner, la metadata y los artefactos de campaña en un estado suficientemente robusto, reproducible y metodológicamente limpio para soportar:

- la campaña grande `17400` con `30` seeds;
- extensiones encadenadas de `30` seeds adicionales;
- y el análisis posterior multi-seed sin lógica ad hoc ni reapertura manual de artefactos históricos.

## Canonical Components

Piezas principales del bloque:

- [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
- [scripts/report_f7_campaign.py](../scripts/report_f7_campaign.py)
- [scripts/report_f7_lineage.py](../scripts/report_f7_lineage.py)
- [evaluation/f7_campaign_spec.py](../evaluation/f7_campaign_spec.py)
- [evaluation/f7_campaign_state.py](../evaluation/f7_campaign_state.py)
- [evaluation/f7_campaign_runner.py](../evaluation/f7_campaign_runner.py)
- [evaluation/f7_campaign_lineage.py](../evaluation/f7_campaign_lineage.py)
- [evaluation/meta_context.py](../evaluation/meta_context.py)

Contratos versionados nuevos o formalizados:

- [config/f7_class_ontology_v1.yaml](../config/f7_class_ontology_v1.yaml)
- [config/f7_target_contract_v1.yaml](../config/f7_target_contract_v1.yaml)
- [config/f7_metric_grammar_v1.yaml](../config/f7_metric_grammar_v1.yaml)
- [config/f7_metric_availability_contract_v1.yaml](../config/f7_metric_availability_contract_v1.yaml)
- [config/f7_metric_aggregation_contract_v1.yaml](../config/f7_metric_aggregation_contract_v1.yaml)
- [config/f7_evaluation_population_contract_v1.yaml](../config/f7_evaluation_population_contract_v1.yaml)
- [config/f7_prediction_row_join_contract_v1.yaml](../config/f7_prediction_row_join_contract_v1.yaml)
- [config/f7_feature_schema_contract_v1.yaml](../config/f7_feature_schema_contract_v1.yaml)
- [config/f7_factor_parser_contract_v1.yaml](../config/f7_factor_parser_contract_v1.yaml)

## What Is Frozen Now

La campaña congela explícitamente, antes de ejecutar trials:

- inventories de datasets, run specs y trials;
- `analysis_contracts`;
- `expected_replication`;
- parser de factores y versiones de gramática relevantes;
- semántica de target y de clases;
- contrato de join de predicciones;
- contrato de disponibilidad y agregación de métricas;
- contrato de namespaces y superficies de interpretabilidad.

El `trial inventory` persiste ya:

- `x_transform`
- `y_transform`
- `synthetic_policy`
- `run_policy`
- `flowpre_usage`
- `flowgen_usage`
- `lineage_trial_group_id`
- `expected_seed_count`
- `panel_build_version`
- `panel_build_timestamp`

## What The Runner Guarantees

Por run válida, el runner persiste:

- `run_manifest.json`
- `results.yaml`
- `metrics_long.csv`
- `predictions_eval_raw.csv.gz`
- summary de interpretabilidad
- CSVs de interpretabilidad de la run

Y además sube al `trial_state` / `trial_ledger.csv` una superficie plana suficiente para análisis y pooling:

- rutas de artefactos;
- factores parseados;
- ids/versiones de contratos;
- semántica del target;
- `analysis_ready_comparable`;
- `analysis_ready_blockers`;
- warnings clasificados;
- provenance (`panel_build_version`, `panel_build_timestamp`, `git_commit` cuando existe);
- identidad de superficie de interpretabilidad;
- `expected_seed_count` y `observed_seed_count`.

El runner y `rebuild-state` quedan alineados: no hay campos críticos que dependan solo del camino live de ejecución.

## Lineage Guarantees

La capa de lineage valida y agrega:

- igualdad de contratos a lo largo de `primary + extensions`;
- compatibilidad estructural;
- no solape de seeds;
- igualdad exacta entre seeds esperadas y observadas por campaña;
- completitud por `lineage_trial_group_id`;
- cobertura de `analysis_ready_comparable`.

Artefactos canónicos de lineage:

- `lineage_trial_registry.csv`
- `lineage_metric_panel.csv`
- `lineage_metric_aggregate.csv`
- `lineage_metric_panel_detailed.csv`
- `lineage_metric_aggregate_detailed.csv`
- agregados y estabilidad de interpretabilidad por superficie
- `lineage_summary.json`
- `lineage_report.json`
- `lineage_report.md`

## Validation Executed

### Code and Test Validation

Se validó el bloque con compilación y tests dirigidos sobre:

- loaders de contratos y parser;
- materialización de campañas;
- estado, ledger y reporting;
- clasificación de warnings;
- `analysis_ready_comparable`;
- validación de extensiones;
- pooling de lineage;
- surfaces detalladas de métricas e interpretabilidad.

Resultado final de la batería relevante:

- `20` tests dirigidos del bloque, `OK`

### Real End-to-End Validation

Se reruneó desde cero la cadena pequeña de validación:

- `primary`: `104` runs
- `extension_1`: `52` runs
- `extension_2`: `52` runs

Total:

- `208` runs
- `4` seeds por grupo estructural en el lineage final

Resultados de esa validación:

- `208/208` completadas y válidas
- `analysis_ready_comparable = true` en `208/208`
- `analysis_ready_blockers = []` en `208/208`
- `lineage_pool_ready = true`
- `lineage_pool_blockers = []`
- `expected_seed_count = observed_seed_count = 4` en los `52` grupos estructurales

## Statistical Analysis Readiness

Con este cierre, la campaña ya produce la superficie que el plan estadístico necesita aguas arriba.

### Predictive Metrics

La capa de lineage expone:

- panel compacto para objetivo canónico `macro`;
- panel detallado en `raw_real` con:
  - `train`
  - `val`
  - `test`
  - `overall`
  - `overall_quantile`
  - `macro`
  - `worst_class`
  - `per_class`
  - `per_class_quantile`

Esto deja soportado el análisis posterior de:

- performance agregada;
- Simpson-risk;
- per-class;
- worst-case;
- quantiles;
- gaps entre splits;
- pooling multi-seed.

### Interpretability

Se conservan los artefactos por run individual y además se exponen agregados multi-seed por lineage para:

- `semantic_bridge_perturbation`
- `xgb_native_shap`
- `mlp_flowpre_native_latent_perturbation`

Incluyendo:

- `global`
- `per_class`
- estabilidad de rankings y top-k entre seeds

### Provenance

Se valida y persiste:

- `panel_build_version`
- `panel_build_timestamp`
- `lineage_aggregate_build_version`
- `git_commit` cuando está disponible

## Non-Blocking Note

Queda una observación no bloqueante:

- aparecen `2` warnings surfaced de `nflows.transforms.lu` / `torch.triangular_solve`

Estos warnings:

- no rompen ejecución;
- no alteran comparabilidad;
- no afectan a results ni interpretabilidad;
- y hoy se interpretan como deuda técnica upstream, no como invalidez metodológica.

Se pueden reclasificar más adelante como `silenced_known_noise` si se quiere reducir ruido operativo, pero no condicionan el cierre del bloque.

## Out Of Scope

Este cierre no implica todavía:

- implementación de la capa analítica final del bloque `14`;
- selección final de variantes;
- inferencia estadística final;
- ni decisión final sobre shortlist del TFG.

El bloque `13` deja preparada la infraestructura y la superficie upstream para que ese trabajo pueda hacerse de forma limpia y reproducible.

## Closeout Decision

Con base en:

- la validación unitaria e integrada del runner;
- la validación real de `primary + 2 extensions`;
- la persistencia completa de artefactos por run;
- el pooling correcto por lineage;
- y la disponibilidad de la superficie requerida por el plan estadístico;

el bloque `13` puede considerarse **cerrado**.

Lo siguiente ya no es más hardening del runner, sino:

- cierre documental del bloque en el mapa global del proyecto;
- y preparación / readiness final de la campaña grande.
