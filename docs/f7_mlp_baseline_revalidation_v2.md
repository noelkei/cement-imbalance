# F7 MLP Baseline Revalidation v2

## Proposito

Esta es la segunda y ultima revalidacion acotada para cerrar el baseline de `MLP` de `F7`.

Llega despues de la `v1`, donde ya quedaron fijados dos puntos:

- `batch_size = 198`
- `num_epochs = 300`

La pregunta que resuelve esta `v2` ya no es de batch ni de budget bruto, sino de capacidad estructural del `MLP` y de robustez razonable frente a `loss` y `policy`.

## Superficie fija

Toda la `v2` comparte:

- dataset canonico: `dataset__x-standard__y-minmax__syn-none__v1`
- `synthetic_policy = none`
- `x_transform = standard`
- `y_transform = minmax`
- `batch_size = 198`
- `num_epochs = 300`
- `learning_rate = 1e-4` por defecto salvo dos cfgs de chequeo fino
- `embedding_dim = 12`
- metrica guia de seleccion:
  - `raw_real.macro.rrmse` en `val`

## Regla metodologica

- `test` no participa en la seleccion;
- `test` solo se habilita para un chequeo final de descarte sobre la configuracion ganadora;
- el analisis principal debe mirar:
  - `val`;
  - `train`;
  - balance `50/50` entre ambos para detectar mejoras falsas por sobreajuste.

## Que significa familia en esta iteracion

En esta `v2`, una familia es la parte estructural y de optimizacion comun del `MLP`:

- `batch_size`
- `num_epochs`
- `learning_rate`
- `early_stopping_patience`
- `lr_decay_patience`
- `lr_scheduler`
- `hidden_dim`
- `num_layers`
- `embedding_dim`

No forman parte de la familia:

- `loss_reduction`
- `regression_group_metric`
- `dataloader_mode`
- `cycle_reals`

Esos cuatro se tratan como variaciones run-level dentro o alrededor de una familia.

## Diseño

- `40` configuraciones intencionales;
- `3` seeds compartidas:
  - `1234`
  - `2345`
  - `3456`
- total fase principal:
  - `120` runs
- chequeo final:
  - `3` runs adicionales del ganador con `test` habilitado

## Bloques de configuraciones

### 1. `20` cfgs estructurales base

Sirven para responder:

- si merece la pena subir anchura;
- si merece la pena subir profundidad;
- si `6` u `8` capas empiezan a sobreajustar o a dejar de compensar;
- y si, en las familias mas profundas, hace falta tocar algo de optimizacion.

Cobertura:

- `hidden_dim`: `128`, `192`, `256`, `384`, `512`
- `num_layers`: `3`, `4`, `6`, `8`
- dos chequeos finos adicionales con:
  - `learning_rate = 5e-5`
  - `early_stopping_patience = 30`
  - `lr_decay_patience = 15`

Estas `20` cfgs usan por defecto:

- `loss = overall + rmse`
- regimen `plain`

### 2. `10` cfgs con `per_class_equal + rmse`

Seleccionan `5` familias estructurales candidatas y, para cada una, comparan:

- `plain`
- `imbalance_aware`

Esto deja ver si una familia prometedora sigue siendo razonable cuando cambiamos el criterio de loss y el regimen de batching/cycling.

### 3. `10` cfgs con `per_class_equal + rrmse`

Mismo diseño que el bloque anterior, pero con `rrmse`, para comprobar si la mejora observada en `v1` se sostiene en familias mas profundas o mas anchas.

## Que esperamos contestar con esta v2

- si la mejor familia sigue estando cerca de `4` capas o si compensa ir a `6`;
- si `8` capas ya entran en zona de sobrecoste / sobreajuste;
- si subir de `256` a `384` o `512` aporta algo real;
- si los cambios de optimizacion solo ayudan a modelos mas profundos;
- y que combinacion concreta merece congelarse como baseline final de `MLP` para `F7`.

## Artefactos esperados

La ejecucion debe dejar:

- runs canonicas `MLP` con `results.yaml`, `metrics_long.csv` y `run_manifest.json`;
- tabla por seed con:
  - metricas
  - `runtime_s`
  - `epochs_ran`
  - `best_epoch`
  - `stopped_early`
- resumen agregado por `cfg_id`;
- resumen agregado por familia estructural;
- resumen por variante dentro de familia;
- chequeos por hiperparametro y por compatibilidad;
- benchmark por dispositivo si se lanza con varios `run_label`;
- reporte breve con shortlist final.

## Fuente machine-readable

La especificacion ejecutable de esta iteracion vive en:

- [config/f7_mlp_baseline_revalidation_v2.yaml](../config/f7_mlp_baseline_revalidation_v2.yaml)
