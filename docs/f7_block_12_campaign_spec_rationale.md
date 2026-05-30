# F7 Block 12 Campaign Spec Rationale

## Decision

El bloque `12` queda cerrado con una `campaign spec` raíz en YAML y con inventarios derivados materializados explícitamente.

Artefactos principales:

- [config/f7_campaign_spec_v1.yaml](../config/f7_campaign_spec_v1.yaml)
- [scripts/materialize_f7_campaign_spec.py](../scripts/materialize_f7_campaign_spec.py)
- [evaluation/f7_campaign_spec.py](../evaluation/f7_campaign_spec.py)

Artefactos derivados materializados:

- `outputs/reports/f7_campaign_spec/f7_campaign_dataset_candidates_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_run_specs_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_trials_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_expansion_manifest_v1.json`

## Que queda fijado

- `campaign_id = f7_campaign_v1`
- `campaign_spec_id = f7_campaign_spec_v1`
- `campaign_scope = full_f7_17400`
- `seed_set_id = f7_seed_panel_v1`
- `run_mode = holdout_run`
- `allow_test_holdout = true`
- `test_enabled = true`

Y además:

- `MLP` se expande desde ejes y compatibilidades válidas, no desde enumeración manual;
- `XGBoost` se expande desde una sola `run_spec` canónica;
- el inventario completo de `trial_id` queda materializado antes de runners;
- la spec declara explícitamente contratos globales y de interpretabilidad por familia.

## Punto metodologico importante

La spec no expande `MLP` como un producto cartesiano ingenuo `2 x 2 x 3`.

Aunque los ejes declarados incluyen:

- `batch_policy = [plain, imbalance_aware]`
- `cycling_policy = [no_cycling, cycling]`
- `loss_policy = [overall_rmse, per_class_equal_rmse, per_class_equal_rrmse]`

la expansión válida de campaña conserva la compatibilidad ya fijada en el plan de `17400` runs:

- `plain <-> no_cycling`
- `imbalance_aware <-> cycling`

Eso deja:

- `2` familias de training behavior
- `3` losses
- `6` `run_spec` `MLP`

y preserva el conteo total:

- `17280` runs `MLP`
- `120` runs `XGBoost`
- `17400` runs totales

## Por que esta implementacion es la mas limpia

- la fuente de verdad principal es una sola `campaign spec` revisable a mano;
- los CSV derivados eliminan lógica enterrada en runners;
- el inventario de datasets de `4B` se reutiliza, pero se normaliza a una tabla de `dataset_candidate_id` lista para campaña;
- `trial_id`, `comparison_group_id` y `replication_index` se materializan explícitamente;
- la gramática de ids queda alineada con `evaluation/meta_context.py`.

## Cierre operativo

El bloque `12` se considera cerrado cuando:

- la spec YAML carga y valida;
- la expansión materializa inventarios sin duplicados;
- los conteos coinciden exactamente con `17400`;
- y los runners futuros pueden consumir `trial inventory` sin volver a inferir ids básicos.
