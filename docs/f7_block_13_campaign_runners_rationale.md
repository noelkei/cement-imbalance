# F7 Block 13: Campaign Runners, State, and Extensions

Este documento recoge el rationale de diseño del bloque `13`.

El estado final validado y la decisión formal de cierre quedan en:

- [docs/f7_block_13_closeout.md](f7_block_13_closeout.md)

El bloque `13` queda cerrado con un runner canónico de campaña que consume directamente el `trial inventory` congelado y deja trazabilidad explícita por trial, por intento, por campaña y por lineage.

Piezas principales:

- [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
- [scripts/report_f7_campaign.py](../scripts/report_f7_campaign.py)
- [scripts/report_f7_lineage.py](../scripts/report_f7_lineage.py)
- [evaluation/meta_context.py](../evaluation/meta_context.py)
- [evaluation/f7_campaign_spec.py](../evaluation/f7_campaign_spec.py)
- [evaluation/f7_campaign_runner.py](../evaluation/f7_campaign_runner.py)
- [evaluation/f7_campaign_state.py](../evaluation/f7_campaign_state.py)
- [evaluation/f7_campaign_lineage.py](../evaluation/f7_campaign_lineage.py)

## Qué queda fijado

- existe una sola entrada CLI para:
  - `preflight`
  - `run`
  - `resume`
  - `rerun-failed`
  - `close`
  - `rebuild-state`
- el runner consume el `trial inventory` materializado; no rederiva combinaciones en runtime;
- la mutación de estado vive en:
  - `outputs/campaigns/<campaign_id>/state/trials/<trial_id>.json`
- la superficie humana canónica queda en:
  - `outputs/campaigns/<campaign_id>/trial_ledger.csv`
- la campaña congela también:
  - `analysis_contracts`
  - `expected_replication`
  - parser de factores
  - semántica de target y clases
- existe historial append-only de intentos:
  - `outputs/campaigns/<campaign_id>/trial_attempts.jsonl`
- existe registro global de campañas:
  - `outputs/campaigns/f7_campaign_registry.csv`
- `completed` solo se asigna cuando `campaign_valid_f7=true`
- campañas de extensión por seeds se modelan como campañas nuevas con linaje explícito, sin mutar la campaña base.
- el ledger expone directamente la metadata necesaria para pooling y análisis, sin reabrir manifests arbitrarios en la capa estadística.

## Qué valida el runner antes de lanzar

- consumo estructural del trial:
  - manifests
  - config base
  - contratos
  - ids derivados
  - panel de seeds
- replicación esperada congelada
- coherencia de linaje para `seed_extension`
- requisitos runtime mínimos adicionales por familia
- disponibilidad mínima de superficies necesarias para `analysis_ready_comparable`

Nota importante:

- en `MLP`, el preflight del runner ya detecta datasets que no pueden reconstruir `raw_real` porque les falta `target scaler` aunque el dataset-level `y_transform` no sea `raw`.
- eso no cambia la semántica del bloque `12`; simplemente añade una validación operativa más estricta para ejecución real.

## Qué añade el estado final

Además del runner base, el cierre final del bloque incorpora:

- contratos versionados para:
  - ontología de clases
  - target
  - gramática/disponibilidad/agregación de métricas
  - población de evaluación
  - join de predicciones
  - schema de interpretabilidad
  - parser de factores
- `analysis_ready_comparable` y `analysis_ready_blockers`
- warnings capturados y clasificados en:
  - `silenced_known_noise`
  - `surfaced`
- provenance explícito:
  - `panel_build_version`
  - `panel_build_timestamp`
  - `lineage_aggregate_build_version`
  - `git_commit` cuando existe
- pooling por lineage con validación de:
  - igualdad de contratos
  - no solape de seeds
  - igualdad exacta entre seeds esperadas y observadas
  - completitud por `lineage_trial_group_id`
- agregados canónicos de lineage para:
  - métricas compactas `macro`
  - métricas detalladas `raw_real`
  - interpretabilidad multi-seed

## Superficie que deja para el análisis posterior

El bloque `13` deja lista la superficie upstream que necesita el plan estadístico:

- artefactos completos por run:
  - `run_manifest.json`
  - `results.yaml`
  - `metrics_long.csv`
  - `predictions_eval_raw.csv.gz`
  - summaries y CSVs de interpretabilidad
- pooling multi-seed por lineage:
  - `lineage_trial_registry.csv`
  - `lineage_metric_panel.csv`
  - `lineage_metric_aggregate.csv`
  - `lineage_metric_panel_detailed.csv`
  - `lineage_metric_aggregate_detailed.csv`
  - agregados y estabilidad de interpretabilidad

En particular, la capa detallada de lineage ya expone en `raw_real`:

- `train`
- `val`
- `test`
- `overall`
- `overall_quantile`
- `macro`
- `worst_class`
- `per_class`
- `per_class_quantile`

## Validación realizada

- compilación de módulos nuevos y puntos tocados:
  - `evaluation/meta_context.py`
  - `evaluation/f7_campaign_lineage.py`
  - `evaluation/f7_campaign_state.py`
  - `evaluation/f7_campaign_runner.py`
  - `scripts/run_f7_campaign.py`
  - `scripts/report_f7_campaign.py`
  - `scripts/report_f7_lineage.py`
  - `evaluation/f7_campaign_spec.py`
  - `evaluation/f7_campaign_trial_consumption.py`
  - `evaluation/results.py`
  - `training/train_mlp.py`
  - `training/train_xgboost.py`
- tests estructurales y de integración mínima:
  - `tests.test_f7_campaign_spec`
  - `tests.test_f7_campaign_trial_consumption`
  - `tests.test_f7_campaign_state`
  - `tests.test_f7_campaign_runner`
- smoke real del CLI:
  - `preflight` sobre la spec canónica `F7`

Validación final adicional de cierre:

- rerun real desde cero de la cadena pequeña:
  - `primary`: `104`
  - `extension_1`: `52`
  - `extension_2`: `52`
- total:
  - `208` runs
  - `4` seeds por grupo estructural
- resultado:
  - `208/208` válidas
  - `analysis_ready_comparable=true` en `208/208`
  - `lineage_pool_ready=true`
  - pooling correcto de results e interpretabilidad
  - artifacts completos por run y agregados completos por lineage

## Resultado metodológico

Con este bloque ya no dependemos de:

- scripts ad hoc por familia para lanzar campaña;
- reconstrucción manual de estado;
- convención informal para retries, bloqueos o campañas extendidas.
- inferencia ad hoc de metadata crítica en la capa estadística;
- ni reapertura manual de manifests históricos para agregar seeds.

El siguiente paso natural ya no es “cómo ejecutar”, sino:

- congelar la gramática del análisis principal (`14`);
- y después medir coste real de campaña completa (`15`).
