# F7 XGBoost Baseline Revalidation v2

## Proposito

Esta `v2` es una micro-revalidacion final para cerrar el baseline de `XGBoost` con un criterio mas defendible que el simple ranking bruto por `val`.

Llega despues de la `v1`, donde ya se vio que:

- `XGBoost` puede entrar como baseline fuerte;
- SHAP funciona correctamente;
- pero las mejores cfgs mostraban gaps `train-val` lo bastante grandes como para justificar una ultima comprobacion fina.

## Superficie fija

La `v2` mantiene fija la misma superficie que la `v1`:

- dataset oficial raw/no-scale;
- `synthetic_policy = none`;
- representacion `raw_numeric_plus_type_onehot`;
- sin escalados, losses alternativas ni batching/cycling.

## Objetivo metodologico

La pregunta ya no es "que cfg gana por muy poco en `val`", sino:

- que cfg sigue siendo fuerte;
- que cfg tiene menos tension `train-val`;
- y cual queda mejor posicionada para congelarse como baseline final realmente defendible.

## Diseño

- `9` configuraciones intencionales;
- `3` seeds compartidas;
- total:
  - `27` runs;
- smoke test de SHAP sobre las `3` mejores.

## Foco de la micro-revalidacion

La malla se centra en tres zonas:

- la candidata defendible de `v1`:
  - `d4_mc8`
- variantes mas conservadoras alrededor de esa cfg:
  - mas `min_child_weight`
  - menor `learning_rate`
  - algo de `gamma`
- variantes regularizadas de la mejor cfg bruta de `v1`:
  - `d3_lr5e2_sub10`

## Fuente machine-readable

- [config/f7_xgb_baseline_revalidation_v2.yaml](../config/f7_xgb_baseline_revalidation_v2.yaml)
