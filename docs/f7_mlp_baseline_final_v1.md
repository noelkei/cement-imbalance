# F7 MLP Baseline Final v1

## Decision

El baseline estructural final de `MLP` para `F7` queda congelado en:

- `hidden_dim = 320`
- `num_layers = 4`
- `embedding_dim = 12`
- `batch_size = 198`
- `learning_rate = 1e-4`
- `num_epochs = 300`
- `optimizer = adam`
- `lr_scheduler = plateau`
- `early_stopping_patience = 20`

Config canónica:

- [config/f7_mlp_base_v1.yaml](../config/f7_mlp_base_v1.yaml)

## Que queda decidido y que no

Queda decidido dentro del punto `1`:

- una sola arquitectura base de `MLP` para toda `F7`;
- una sola base estructural compartida por toda la campaña;
- `batch_size` unico de campaña;
- `num_epochs` maximos de campaña;
- familia base elegida por mini-revalidacion y no por herencia ciega.

No queda decidido dentro del punto `1`:

- panel final de seeds;
- `loss_reduction` final de campaña;
- `regression_group_metric` final por brazo;
- `dataloader_mode`;
- `cycle_reals`;
- politica final de comparacion entre `plain` e `imbalance_aware`.

Eso se deja para bloques posteriores porque son ejes run-level, no parte del baseline estructural.

## Alcance de la congelacion

Esta decision congela solo la parte comun del baseline estructural de campaña.

No quedan congelados dentro de este baseline:

- panel de seeds;
- `loss_reduction`;
- `regression_group_metric`;
- `dataloader_mode`;
- `cycle_reals`.

Esos ejes siguen tratándose como variaciones run-level de `F7`.

## Rationale del bloque

### 1. Por que no heredamos literalmente `mlp_closure_base_v1`

La base historica [config/mlp_closure_base_v1.yaml](../config/mlp_closure_base_v1.yaml) se tomó como prior fuerte, pero no como config literal final, por tres razones:

- mezclaba hiperparametros estructurales con decisiones run-level;
- fijaba una `seed_set` historica que no coincide con el diseño de `F7`;
- llevaba dentro una combinacion concreta de:
  - `loss_reduction`
  - `regression_group_metric`
  - `dataloader_mode`
  - `cycle_reals`
  
que en `F7` debían quedar abiertas como variaciones posteriores.

La decision correcta no era reescribir desde cero, sino:

- conservar lo estructural util;
- extraerlo de sus decisiones historicas acopladas;
- y revalidar solo lo necesario.

### 2. Por que se fijaron primero `batch_size = 198` y `num_epochs = 300`

La primera revalidacion mostró dos cosas importantes:

- `256` no servia como batch comun de campaña porque no era compatible con el regimen `balanced + cycle_reals`;
- `198` ofrecia un comportamiento competitivo y compatible con la futura familia `imbalance_aware`.

Sobre `num_epochs`:

- `300` funcionó como techo prudente;
- las runs reales pararon mucho antes por `early stopping`;
- y no apareció señal seria de que `500` aportara mejora estructural suficiente para justificar el aumento de coste.

Por eso `198` y `300` quedaron fijados antes de abrir la frontera final de capacidad.

### 3. Que se quiso optimizar realmente en este bloque

El criterio no fue "la mejor `val` a cualquier precio".

Se buscó un baseline que maximizara señal metodológica para campaña:

- buen rendimiento en `val`;
- buen comportamiento en `train`;
- ausencia de señales claras de degeneracion u overfitting fuerte;
- robustez razonable bajo cambios de `loss` y `policy`;
- y coste compatible con una campaña masiva.

Por eso, a lo largo de `v2` y `v3`, la lectura se hizo mirando conjuntamente:

- `val`;
- `train`;
- score `50/50`;
- `runtime_s`;
- `epochs_ran`;
- y consistencia entre seeds.

### 4. Por que no elegimos `512x6`

La `v3` confirmó que `512x6` era una familia realmente fuerte y no solo un outlier:

- rindió muy bien en `overall + rmse`;
- también rindió bien con `per_class_equal + rmse`;
- y también rindió bien con `per_class_equal + rrmse`.

Eso significa que sí había margen real de capacidad adicional.

Sin embargo, no se eligió como baseline final por una razón metodológicamente importante:

- su coste medio por run se alejaba demasiado del objetivo práctico para `F7`.

En la `v3`, la zona `512x6` quedó aproximadamente en:

- `7.6 s` para la variante estructural `overall + rmse`;
- `8.1-8.7 s` en variantes `plain` con `per_class_equal`;
- y por encima de `11 s` en alguna variante `aware`.

Para una campaña del tamaño de `F7`, eso deja de ser un detalle menor:

- multiplica el coste total;
- estrecha margen operativo;
- y obliga a pagar mucho por una mejora que, aunque real, ya está en la zona de rendimientos decrecientes respecto a alternativas más baratas.

La decision no fue "rechazar lo mejor", sino:

- reconocer que `512x6` es una candidata fuerte;
- pero no aceptarla como baseline comun de campaña porque rompe el equilibrio entre señal y coste.

### 5. Por que sí elegimos `320x4`

`320x4` quedó como la mejor familia de compromiso por varias razones a la vez:

- mejoró claramente respecto a familias más pequeñas como `256x4`;
- se mantuvo en una zona de coste razonable;
- respondió bien al cambiar `loss` y `policy`;
- y no mostró una dependencia rara de una sola variante.

En la `v3`, la familia `320x4` quedó aproximadamente en:

- `3.1 s` para `overall + rmse`;
- `3.2 s` para `per_class_equal + rrmse` en `plain`;
- `4.6-5.5 s` para variantes `aware`;
- media familiar alrededor de `4.0 s`.

Eso la deja en una zona muy útil para `F7`:

- claramente más potente que el baseline pequeño;
- claramente más barata que `512x6`;
- y suficientemente flexible para que los ejes run-level todavía expresen diferencias reales.

### 6. Lectura final del trade-off

La decision final del bloque puede resumirse asi:

- `512x6` mostró el techo alto de capacidad;
- `320x4` mostró el mejor punto operativo para campaña;
- y `384x4` quedó como alternativa fuerte, pero menos equilibrada que `320x4` en la frontera coste-beneficio que queríamos respetar.

En otras palabras:

- no se eligió la familia con mejor resultado absoluto;
- se eligió la familia con mejor equilibrio entre:
  - señal estadística;
  - estabilidad;
  - comparabilidad;
  - y coste total de campaña.

### 7. Rationale resumido
- `v1` fijó `batch_size = 198` y `num_epochs = 300`.
- `v2` mostró que la zona fuerte de capacidad estaba por encima del baseline pequeño.
- `v3` confirmó que `512x6` mejora de verdad, pero se sale del objetivo práctico de coste para campaña.
- `320x4` quedó como mejor compromiso entre:
  - rendimiento;
  - estabilidad;
  - robustez bajo variaciones run-level;
  - y tiempo medio por run cercano al objetivo práctico.

## Artefactos de apoyo

- [docs/f7_mlp_baseline_revalidation_v1.md](f7_mlp_baseline_revalidation_v1.md)
- [docs/f7_mlp_baseline_revalidation_v2.md](f7_mlp_baseline_revalidation_v2.md)
- [docs/f7_mlp_baseline_revalidation_v3.md](f7_mlp_baseline_revalidation_v3.md)
