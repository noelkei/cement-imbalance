# F7 Campaign Pilot V1

`f7_campaign_pilot_v1` es la campaña piloto operativa previa a la campaña completa `F7`.

## Propósito

Validar en entorno real, con la misma gramática de campaña que `f7_campaign_v1`, todo lo siguiente:

- runner canónico de campaña;
- ledger por trial e historial de intentos;
- rutas y artifacts por familia;
- cierre de campaña y registry;
- tiempos reales por familia y por `run_spec`;
- footprint de artifacts;
- calidad de la superficie meta para cerrar los bloques `14`, `15` y `16`.

## Alcance

- mismos `100` dataset candidates que la campaña grande:
  - `96` `MLP`
  - `4` `XGBoost`
- mismas `7` `run_spec`:
  - `6` `MLP`
  - `1` `XGBoost`
- mismo `run_mode = holdout_run`
- misma política de `test_enabled = true`
- mismo contrato raw y misma política de interpretabilidad
- solo `1` seed

Conteos:

- `MLP`: `96 * 6 * 1 = 576`
- `XGBoost`: `4 * 1 * 1 = 4`
- total: `580`

## Artefactos fuente

- spec:
  - [config/f7_campaign_pilot_v1.yaml](../config/f7_campaign_pilot_v1.yaml)
- seed panel:
  - [config/f7_seed_panel_pilot_v1.yaml](../config/f7_seed_panel_pilot_v1.yaml)
  - [config/f7_seed_panel_pilot_v1.csv](../config/f7_seed_panel_pilot_v1.csv)

## Materialización

- script dedicado:
  - [scripts/materialize_f7_campaign_pilot_v1.py](../scripts/materialize_f7_campaign_pilot_v1.py)
- runner canónico:
  - [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
- reporte post-run:
  - [scripts/report_f7_campaign.py](../scripts/report_f7_campaign.py)

## Lectura metodológica

Esta campaña no reemplaza a `f7_campaign_v1` y no debe confundirse con el análisis principal.

Se interpreta como:

- piloto operativo;
- benchmark real de coste;
- prueba de integridad de artefactos y trazabilidad;
- entrada empírica para cerrar:
  - `14` gramática del análisis principal
  - `15` benchmark real de tiempos
  - `16` preflight final de lanzamiento
