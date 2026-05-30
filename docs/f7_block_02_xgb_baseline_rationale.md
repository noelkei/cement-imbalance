# F7 Block 02 XGBoost Baseline Rationale

Este bloque ya quedó razonado en detalle en:

- [docs/f7_xgb_baseline_final_v1.md](f7_xgb_baseline_final_v1.md)

Este archivo existe para dar una ruta de búsqueda estable y homogénea para la memoria del TFG.

## Decision final

El baseline final de `XGBoost` para `F7` queda congelado en:

- `objective = reg:squarederror`
- `eval_metric = rmse`
- `n_estimators = 1200`
- `learning_rate = 0.035`
- `max_depth = 4`
- `min_child_weight = 12`
- `subsample = 0.85`
- `colsample_bytree = 0.80`
- `reg_alpha = 0.02`
- `reg_lambda = 1.50`
- `gamma = 0.0`
- `tree_method = hist`
- `max_bin = 256`
- `early_stopping_rounds = 60`

Config canónica:

- [config/f7_xgb_base_v1.yaml](../config/f7_xgb_base_v1.yaml)

## Rationale completo

Para evitar duplicar y desalinear narrativa, el rationale metodológico completo del bloque `2` queda concentrado en:

- [docs/f7_xgb_baseline_final_v1.md](f7_xgb_baseline_final_v1.md)

## Revalidaciones de apoyo

- [docs/f7_xgb_baseline_revalidation_v1.md](f7_xgb_baseline_revalidation_v1.md)
- [docs/f7_xgb_baseline_revalidation_v2.md](f7_xgb_baseline_revalidation_v2.md)
