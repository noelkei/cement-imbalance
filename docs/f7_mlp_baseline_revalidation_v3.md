# F7 MLP Baseline Revalidation v3

## Proposito

Esta `v3` es la ultima iteracion prevista para cerrar el baseline final de `MLP` de `F7`.

Hereda de la `v2` dos decisiones ya fijas:

- `batch_size = 198`
- `num_epochs = 300`

La pregunta ya no es si hace falta mas presupuesto bruto, sino:

- si la mejor zona estructural real esta en `256`, `320`, `384`, `448` o `512`;
- si `5` capas puede capturar parte de la mejora de `6` con menos coste;
- y si `512x6` es una mejora general o solo brilló en una parte estrecha del espacio.

## Restriccion practica

Queremos que la familia final no se dispare de coste.

Por eso la `v3` se diseña con un criterio explicito:

- el target practico es quedar alrededor de `5 s` de media por run;
- pueden existir algunas cfgs por encima para comprobar frontera de capacidad;
- pero la decision final debe ponderar rendimiento y coste real, no solo `val`.

## Superficie fija

Toda la `v3` comparte:

- dataset canonico: `dataset__x-standard__y-minmax__syn-none__v1`
- `synthetic_policy = none`
- `x_transform = standard`
- `y_transform = minmax`
- `batch_size = 198`
- `num_epochs = 300`
- `learning_rate = 1e-4` por defecto salvo un chequeo fino
- `embedding_dim = 12`
- metrica guia de seleccion:
  - `raw_real.macro.rrmse` en `val`

## Regla metodologica

- `test` no participa en la seleccion;
- `test` solo se habilita para un chequeo final del ganador;
- el analisis principal debe mirar:
  - `val`
  - `train`
  - score `50/50`
  - `runtime_s`
  - `epochs_ran`

## Diseño

- `40` configuraciones intencionales;
- `3` seeds compartidas;
- total fase principal:
  - `120` runs

## Bloques

### 1. `20` cfgs estructurales

Se centran en:

- `hidden_dim = 256, 320, 384, 448, 512`
- `num_layers = 3, 4, 5, 6`
- una unica comprobacion fina adicional:
  - `320x4` con `lr = 5e-5`
  - `early_stopping_patience = 30`
  - `lr_decay_patience = 15`

Esta malla sirve para:

- explorar anchuras nuevas no probadas en `v2`;
- medir si `5` capas es mejor compromiso que `6`;
- y revisar si `512x6` sigue mereciendo la pena frente a candidatas mas baratas.

### 2. `10` cfgs con `per_class_equal + rmse`

Se aplican a `5` familias estructurales:

- `256x4`
- `320x4`
- `384x4`
- `512x4`
- `512x6`

y para cada familia comparan:

- `plain`
- `imbalance_aware`

### 3. `10` cfgs con `per_class_equal + rrmse`

Mismo diseño que el bloque anterior, para comprobar si el comportamiento fuerte de `rrmse` se sostiene en esta frontera final.

## Que esperamos contestar

- si la mejor familia final esta cerca de `384x4`, `512x4` o `512x6`;
- si `320x4` o `320x5` capturan buena parte de la mejora con menos coste;
- si `5` capas es el mejor compromiso de profundidad;
- si `512x6` también rinde bien cuando cambiamos `loss` y `policy`;
- y qué baseline final deja mejor equilibrio entre rendimiento, estabilidad y tiempo.

## Fuente machine-readable

La especificacion ejecutable de esta iteracion vive en:

- [config/f7_mlp_baseline_revalidation_v3.yaml](../config/f7_mlp_baseline_revalidation_v3.yaml)
