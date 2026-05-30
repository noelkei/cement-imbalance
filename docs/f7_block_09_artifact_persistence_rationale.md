# F7 Block 09 Artifact Persistence Rationale

## Decision

El bloque `9` queda cerrado como **política canónica de persistencia por run**, no solo como guía narrativa.

Artefacto contractual:

- [config/f7_artifact_persistence_contract_v1.yaml](../config/f7_artifact_persistence_contract_v1.yaml)

Validación operativa ejecutada el `2026-05-18`:

- smoke `MLP selection_run`
- smoke `MLP holdout_run`
- smoke `XGBoost selection_run`
- smoke `XGBoost holdout_run`

En los cuatro casos se verificó:

- `run_manifest.json` con `artifact_policy_id`, `artifact_availability` y `artifact_paths`
- `predictions_eval_raw.csv.gz` presente y parseable
- `val` persistido siempre
- `test` persistido solo cuando `test_enabled=true`

## Qué queda fijado

La campaña `F7` distingue explícitamente tres niveles:

- `must_persist_per_run`
- `must_persist_per_shortlist`
- `must_persist_per_finalist`

Y congela que toda run canónica debe dejar, como mínimo:

- `results.yaml`
- `run_manifest.json`
- `metrics_long.csv`
- snapshot de config
- ids meta
- contrato raw y su validación
- sidecar de predicciones evaluadas en `raw`:
  - `val` siempre
  - `test` solo cuando la run es `holdout_run`
- interpretabilidad por run:
  - `interpretability_summary.json`
  - superficies familiares globales y por clase requeridas por `analysis_ready_comparable`

Y fija además la política de nombres para nuevas runs `F7`:

- una sola copia estable de `results.yaml`
- una sola copia estable de `run_manifest.json`
- una sola copia estable de `metrics_long.csv`
- una sola copia estable de `config.yaml`

Los aliases versionados históricos siguen siendo legibles en lectores tolerantes, pero dejan de emitirse en la escritura nueva.

## Predicción persistida

La salida canónica ligera de predicciones queda en:

- `predictions_eval_raw.csv.gz`

La persistencia es deliberadamente:

- suficientemente rica para análisis posteriores por fila;
- suficientemente ligera para no disparar el volumen de `17400` runs.

Por eso:

- `train` no se persiste por defecto;
- `val` sí se persiste siempre;
- `test` se persiste cuando la run lo habilita explícitamente.

## Política por familia

Se congela además:

- `MLP`
  - no persiste pesos por defecto para toda run
  - `model_artifact_policy = shortlist_or_finalist_only`
- `XGBoost`
  - sí persiste el booster de cada run
  - `model_artifact_policy = persist_every_run`

## Interpretabilidad

El contrato vigente ya refleja el estado final tras cerrar bloques `10` y `11`:

- `interpretability_required_now = true`
- `family_specific_implementation_pending = false`
- persistencia completa por run de la superficie familiar necesaria para agregación posterior

## Rationale resumido

- `F7` necesitaba una política real de artefactos por run;
- no bastaba con `results.yaml` y `metrics_long.csv` sin una capa explícita de disponibilidad y trazabilidad;
- las predicciones `raw` de `val/test` son una pieza clave para análisis estadístico serio;
- `MLP` y `XGBoost` debían compartir una misma gramática de persistencia, con diferencias controladas en el artefacto de modelo;
- y, una vez cerrada la interpretabilidad por familia, la optimización de disco correcta ya no es recortar superficie analítica, sino eliminar únicamente duplicados físicos redundantes.
