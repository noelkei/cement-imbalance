# F7 XGBoost Baseline Revalidation v1

## Proposito

Esta mini-revalidacion sirve para congelar un baseline fuerte de `XGBoost` para `F7` sin convertirlo en un segundo estudio del tamaño de `MLP`.

La idea es:

- mantener `XGBoost` acotado en ejes experimentales;
- pero darle una configuracion de entrenamiento realmente competitiva;
- y comprobar ademas que luego podremos usar SHAP sin sorpresas operativas graves.

## Superficie fija

La revalidacion se ejecuta sobre:

- dataset oficial raw/no-scale;
- `synthetic_policy = none`;
- representacion fija de `XGBoost`:
  - numericas raw;
  - `type` en one-hot;
  - sin `post_cleaning_index`.

No se abren aqui:

- escalados de `X`;
- escalados de `y`;
- training losses alternativas;
- batching o cycling.

## Regla metodologica

- la metrica guia de seleccion es `raw_real.macro.rrmse` en `val`;
- `test` no participa en la seleccion;
- la comparacion principal se hace con `3` seeds;
- una vez seleccionadas las `2-3` mejores cfgs, se hace un smoke test de SHAP para validar interpretabilidad practica.

## Diseño

- `20` configuraciones intencionales;
- `3` seeds compartidas:
  - `1234`
  - `2345`
  - `3456`
- total fase principal:
  - `60` runs
- fase SHAP:
  - `2-3` cfgs top
  - `1` modelo representativo por cfg

## Hiperparametros en observacion

Se revisan solo hiperparametros del booster:

- `max_depth`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `learning_rate`
- `reg_alpha`
- `reg_lambda`
- `gamma`
- `n_estimators`
- `early_stopping_rounds`

## Chequeo de SHAP

El objetivo del smoke test de SHAP no es producir ya la politica final de interpretabilidad, sino confirmar:

- que `TreeExplainer` funciona con la cfg elegida;
- que el tiempo de computo no es absurdo;
- y que no hace falta reabrir la familia de baseline por motivos de interpretabilidad.

## Fuente machine-readable

La especificacion ejecutable vive en:

- [config/f7_xgb_baseline_revalidation_v1.yaml](../config/f7_xgb_baseline_revalidation_v1.yaml)
