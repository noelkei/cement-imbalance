# F7 XGBoost Baseline Final v1

## Decision

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

## Que queda decidido y que no

Queda decidido dentro del punto `2`:

- una sola configuración base de `XGBoost` para toda `F7`;
- una sola representación fija de dataset para `XGBoost`;
- entrenamiento sobre dataset oficial raw/no-scale;
- `feature_policy = raw_numeric_plus_type_onehot`;
- baseline elegido por mini-revalidación y micro-revalidación final, no por herencia ciega.

No queda decidido dentro del punto `2`:

- panel final de `30` seeds de campaña;
- política exacta de persistencia de artefactos de campaña;
- gramática final de agregación estadística de interpretabilidad;
- shortlist final de runs a destacar en la memoria.

Eso se termina de cerrar en los bloques posteriores de seeds, persistencia e interpretabilidad.

## Alcance de la congelacion

Esta decisión congela:

- la representación base de `XGBoost`;
- la configuración del booster;
- la semántica de entrenamiento;
- y la viabilidad práctica de SHAP.

No abre:

- escalados alternativos;
- training losses alternativas;
- batching o cycling;
- ni un subestudio paralelo de tuning para `XGBoost`.

## Rationale del bloque

### 1. Por que no se congeló directamente la base historica

La `DEFAULT_XGB_CONFIG` histórica se tomó como prior fuerte, pero no se aceptó sin comprobarla porque:

- `XGBoost` debía entrar como baseline fuerte y no como script heredado;
- la comparación con `MLP` exigía una elección defendible;
- y además necesitábamos verificar que la capa de interpretabilidad posterior no obligara a reabrir la familia.

La decisión correcta era:

- reutilizar la base histórica razonable;
- revalidarla sobre el dataset oficial raw/no-scale;
- y ajustar solo lo necesario para cerrar una versión canónica.

### 2. Que superficie se fijó

La revalidación se hizo con una superficie deliberadamente acotada:

- dataset oficial raw/no-scale;
- `synthetic_policy = none`;
- numericas raw;
- `type` en one-hot;
- sin `post_cleaning_index`;
- sin barrer escalados de `X` o `y`;
- sin abrir batching, cycling ni losses de entrenamiento alternativas.

Eso era importante porque `XGBoost` no debía convertirse en un segundo estudio del tamaño de `MLP`.

### 3. Que se quiso optimizar realmente

El objetivo no fue solo la mejor `val` bruta.

Se buscó una configuración que fuera:

- competitiva en `val`;
- metodológicamente defendible;
- con menor tensión `train-val` que las opciones más agresivas;
- barata de ejecutar;
- y compatible con SHAP real.

Por eso la lectura del bloque combinó:

- `val raw_real.macro.rrmse`;
- comportamiento en `train`;
- gap `train-val`;
- coste por run;
- y chequeo explícito de SHAP.

### 4. Por que no se eligió la familia `d3_*`

La micro-revalidación final confirmó que la familia más agresiva de profundidad `3` seguía liderando por muy poco en `val`.

Sin embargo:

- la ventaja sobre las mejores variantes `d4_*` fue muy pequeña;
- los gaps `train-val` siguieron siendo altos;
- y el bloque ya no estaba buscando un ganador bruto, sino un baseline fuerte y mejor defendible.

No se rechazó `d3_*` porque fuera mala, sino porque no daba una mejora suficiente como para justificar aceptar una configuración más tensa como baseline canónica.

### 5. Por que sí se eligió `d4_mc12`

`d4_mc12` quedó como la mejor decisión de compromiso porque:

- quedó prácticamente empatada con las mejores en `val`;
- regulariza más que el ancla `d4_mc8`;
- reduce algo la tensión de ajuste respecto a opciones más agresivas;
- sigue siendo muy barata;
- y es una configuración sencilla de explicar y defender en la campaña.

En otras palabras:

- no se eligió la mejor `val` absoluta;
- se eligió la mejor combinación de rendimiento, sobriedad y defendibilidad.

### 6. SHAP queda validado y exigido

La `v1` y la `v2` confirmaron que:

- `XGBoost` funciona correctamente con `TreeExplainer`;
- el tiempo de cómputo observado en smoke test fue razonable;
- y no hace falta moverse a `LightGBM` por motivos de interpretabilidad.

Además, esta decisión deja una exigencia operativa para la campaña:

- cuando `XGBoost` entre en la parrilla de `F7`, la persistencia SHAP debe ser lo más completa posible dentro de un coste razonable.

Eso significa que la capa final de interpretabilidad de campaña debe permitir, como mínimo:

- valores SHAP por muestra y por feature;
- valores firmados;
- agregados absolutos;
- `expected_value`;
- nombres de features;
- y persistencia suficiente para luego calcular medias, desviaciones estándar y análisis agregados entre runs.

La gramática exacta de esos artefactos se cerrará en los bloques posteriores de persistencia e interpretabilidad, pero la exigencia ya queda fijada aquí.

## Artefactos de apoyo

- [docs/f7_xgb_baseline_revalidation_v1.md](f7_xgb_baseline_revalidation_v1.md)
- [docs/f7_xgb_baseline_revalidation_v2.md](f7_xgb_baseline_revalidation_v2.md)
