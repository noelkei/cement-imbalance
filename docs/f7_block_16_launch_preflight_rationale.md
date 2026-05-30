# F7 Block 16 Launch Preflight Rationale

## Decision

El bloque `16` queda cerrado como preflight final de lanzamiento de la campaña grande `F7`.

La decisión metodológica fijada para este bloque fue:

- no exigir una nueva mini-campaña de entrenamiento solo para preflight;
- reutilizar como validación real previa la cadena canónica pequeña `104 + 52 + 52` ya cerrada en el bloque `13`;
- y exigir además una `readiness pass` no mutante sobre la cadena grande completa:
  - `primary` de `30` seeds
  - `extension_1` de `30` seeds
  - `extension_2` de `30` seeds
  - `extension_3` de `30` seeds

## Canonical Artifacts

Piezas principales añadidas o usadas para este cierre:

- [config/f7_campaign_spec_v1.yaml](../config/f7_campaign_spec_v1.yaml)
- [config/f7_campaign_extension1_v1.yaml](../config/f7_campaign_extension1_v1.yaml)
- [config/f7_campaign_extension2_v1.yaml](../config/f7_campaign_extension2_v1.yaml)
- [config/f7_campaign_extension3_v1.yaml](../config/f7_campaign_extension3_v1.yaml)
- [config/f7_seed_panel_v1.yaml](../config/f7_seed_panel_v1.yaml)
- [config/f7_seed_panel_extension1_v1.yaml](../config/f7_seed_panel_extension1_v1.yaml)
- [config/f7_seed_panel_extension2_v1.yaml](../config/f7_seed_panel_extension2_v1.yaml)
- [config/f7_seed_panel_extension3_v1.yaml](../config/f7_seed_panel_extension3_v1.yaml)
- [evaluation/f7_launch_readiness.py](../evaluation/f7_launch_readiness.py)
- [scripts/report_f7_launch_readiness.py](../scripts/report_f7_launch_readiness.py)

Reporte canónico generado:

- [f7_launch_readiness_v1.json](../outputs/reports/f7_launch_readiness/f7_launch_readiness_v1.json)
- [f7_launch_readiness_v1.md](../outputs/reports/f7_launch_readiness/f7_launch_readiness_v1.md)

## What Was Required

El criterio congelado de `go/no-go` fue estrictamente `all-green-or-no-go`.

Condiciones requeridas:

1. materialización íntegra de `primary + 3 extensions`;
2. conteos exactos esperados por campaña y globales;
3. `preflight` limpio por campaña;
4. contratos frozen coherentes a lo largo de toda la cadena;
5. linaje esperado consistente hasta `120` seeds totales;
6. `environment freeze` registrado;
7. reporte canónico propio de readiness;
8. `test` habilitado como output, pero no como superficie iterativa de selección.

## Observed Launch Readiness Result

Resultado observado del readiness pass:

- `go_no_go = go`
- `campaign_count = 4`
- `expected_total_trial_count = 69600`
- `observed_total_trial_count = 69600`
- `expected_total_seed_count = 120`
- `observed_total_seed_count = 120`
- `readiness_markers.blockers = []`

Resultado por campaña:

- `f7_campaign_v1`
  - `17400/17400` trials en preflight
  - `ok = true`
- `f7_campaign_extension1_v1`
  - `17400/17400` trials en preflight
  - `ok = true`
- `f7_campaign_extension2_v1`
  - `17400/17400` trials en preflight
  - `ok = true`
- `f7_campaign_extension3_v1`
  - `17400/17400` trials en preflight
  - `ok = true`

Freeze de entorno registrado en el reporte:

- Python `3.11.5`
- PyTorch `2.7.1`
- XGBoost `2.1.3`
- `git_commit = 739096743d3c1b5d0638d2025ea516accdaba27d`

## Why This Is The Right Preflight

Este cierre es el más robusto y limpio porque:

- no depende de inspección manual dispersa;
- no reabre infraestructura ya cerrada en bloques anteriores;
- valida la cadena real completa que luego se quiere lanzar;
- distingue entre evidencia de ejecución real previa y readiness estructural de la campaña grande;
- y deja un artefacto canónico de `go/no-go`, no solo salida de consola.

## Closeout

El bloque `16` se considera cerrado cuando:

- la validación pequeña real ya existe y es satisfactoria;
- la readiness pass grande no mutante da `go`;
- el freeze de entorno queda registrado;
- y la cadena completa `primary + 3 extensions` ya puede lanzarse sin depender de supuestos tácitos.
