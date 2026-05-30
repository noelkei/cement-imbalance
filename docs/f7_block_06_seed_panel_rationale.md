# F7 Block 06 Seed Panel Rationale

## Decision

El bloque `6` congela un panel canónico nuevo de `30` seeds para toda la campaña `F7`.

Los artefactos canónicos son:

- [config/f7_seed_panel_v1.yaml](../config/f7_seed_panel_v1.yaml)
- [config/f7_seed_panel_v1.csv](../config/f7_seed_panel_v1.csv)

El `seed_set_id` congelado es:

- `f7_seed_panel_v1`

## Qué queda fijado

El panel es:

- único y común para `MLP` y `XGBoost`;
- explícito y machine-readable;
- estable durante toda la campaña;
- y no debe cambiarse silenciosamente a mitad de `F7`.

No se reutiliza el panel histórico `closure_5x_v1`.

## Por que se crea un panel nuevo

`F7` ya no es una extensión menor de cierres anteriores.

Necesitaba un panel propio porque:

- la campaña ya tiene una gramática nueva;
- `MLP` y `XGBoost` deben compartir exactamente el mismo eje de réplica;
- y hacía falta una identidad canónica nueva, fácil de referenciar en manifests, specs y agregaciones.

## Criterio de construcción

La prioridad no ha sido “dispersar” seeds por estética numérica, sino:

- reproducibilidad fuerte;
- simplicidad;
- unicidad;
- y orden estable.

Por eso el panel se fija como una lista explícita de enteros simples y únicos, en vez de depender de:

- un generador pseudoaleatorio adicional;
- un panel histórico heredado;
- o una regla implícita no persistida.

## Semántica común por familia

La misma seed de campaña existe como eje común de réplica incluso si cada familia la usa internamente de forma distinta.

La semántica fijada es:

- `MLP`:
  - la seed de campaña gobierna la réplica de entrenamiento;
  - debe bindearse al menos a `python random`, `numpy` y `torch`;
- `XGBoost`:
  - la seed de campaña gobierna la réplica del booster;
  - debe persistirse como `random_state` de la run.

Esto permite que:

- la réplica `k` de `MLP` y la réplica `k` de `XGBoost` pertenezcan al mismo panel de campaña;
- pero sin forzar una igualdad artificial de mecanismos internos.

## Qué no implica esta decisión

Este bloque no reabre la materialización de datasets sintéticos.

La seed de campaña:

- no rematerializa datasets `F7`;
- no cambia los pools sintéticos congelados;
- y no sustituye a la validación de trazabilidad de `trial_id`, `run_spec_id` o `dataset_candidate_id`.

## Propagación operativa

El `seed_set_id` canónico debe quedar reflejado al menos en:

- [config/f7_mlp_base_v1.yaml](../config/f7_mlp_base_v1.yaml)
- [config/f7_xgb_base_v1.yaml](../config/f7_xgb_base_v1.yaml)

Y debe usarse luego en:

- campaign spec;
- runners;
- manifests de run;
- y agregación estadística.

## Rationale resumido

- `F7` necesitaba un panel nuevo y explícito;
- el panel debe ser común a `MLP` y `XGBoost`;
- la prioridad es reproducibilidad y simplicidad, no una dispersión numérica artificial;
- la semántica de uso por familia queda escrita desde ya;
- y el panel queda congelado como artefacto machine-readable con `seed_set_id` canónico.
