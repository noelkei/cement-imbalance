# F7 MLP Baseline Revalidation v1

## Proposito

Este documento fija la mini-revalidacion que usaremos para cerrar el baseline de `MLP` de `F7` antes de congelar la config de campaña.

No es:

- la campaña `F7` final;
- un sweep abierto de hiperparametros;
- un sustituto del plan de `17400` runs.

Si es:

- una comprobacion acotada y metodologicamente limpia del baseline comun de `MLP`;
- la base para decidir si heredamos casi intacta `mlp_closure_base_v1` o si debemos ajustar algunos hiperparametros estructurales.

## Superficie fija

La mini-revalidacion se ejecuta sobre:

- dataset canonico: `dataset__x-standard__y-minmax__syn-none__v1`
- `synthetic_policy = none`
- `x_transform = standard`
- `y_transform = minmax`
- regimen `plain`:
  - `dataloader_mode = baseline`
  - `cycle_reals = false`
- loss de entrenamiento:
  - `loss_reduction = overall`
  - `regression_group_metric = rmse`

## Regla metodologica

- la metrica guia de seleccion debe ser la misma que se quiere usar en campaña:
  - `raw_real.macro.rrmse`
- la comparacion principal de esta mini-revalidacion se hace en `val`;
- `test` no participa en la seleccion;
- `test` solo se habilita para un chequeo final de descarte sobre la configuracion ganadora.

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

## Hiperparametros en observacion

La mini-revalidacion se centra principalmente en:

- `batch_size`
- `num_epochs`
- `learning_rate`
- `hidden_dim`
- `num_layers`

El objetivo no es abrir una optimizacion grande, sino responder preguntas concretas:

- si `256` puede mantenerse como referencia o no;
- si hace falta mover `batch_size` a un valor compatible con `balanced + cycling`;
- si `200` epochs son suficientes;
- si hay una mejora clara al cambiar capacidad o profundidad;
- y si las mejoras puntuales se sostienen cuando combinamos:
  - mas `hidden_dim`
  - mas `num_layers`
  - y mas `num_epochs`

## Familias de configuraciones

Las `40` cfgs se agrupan en seis familias:

- anclas y chequeo de `batch_size`:
  - `legacy_anchor_bs256_e200`
  - `f7_anchor_bs264_e200`
  - `bs198_e200`
  - `bs132_e200`
- chequeo de budget / learning rate / cambios aislados:
  - `bs264_e300`
  - `bs264_e500`
  - `bs264_lr5e5`
  - `bs264_lr3e4`
  - `bs264_hd192`
  - `bs264_l4`
- combinaciones de capacidad sobre `batch_size = 264`:
  - `bs264_hd192_l4_e300`
  - `bs264_hd192_l4_e500`
  - `bs264_hd256_l3_e300`
  - `bs264_hd256_l3_e500`
  - `bs264_hd256_l4_e300`
  - `bs264_hd256_l4_e500`
- combinaciones de capacidad sobre `batch_size = 198`:
  - `bs198_hd192_l3_e300`
  - `bs198_hd192_l4_e300`
  - `bs198_hd256_l3_e300`
  - `bs198_hd256_l4_e300`
- robustez run-level con `per_class_equal + rmse`:
  - `5` cfgs `plain`
  - `5` cfgs `imbalance_aware`
- robustez run-level con `per_class_equal + rrmse`:
  - `5` cfgs `plain`
  - `5` cfgs `imbalance_aware`

## Motivo de la ampliacion a `40` cfgs

La ampliacion no busca convertir este bloque en la campaña `F7` completa, sino comprobar dos cosas antes del freeze del baseline:

- si el baseline estructural aguanta bien al cambiar regimen `plain` vs `imbalance_aware`;
- si las conclusiones cambian mucho al pasar de:
  - `per_class_equal + rmse`
  - a `per_class_equal + rrmse`

Esto ayuda a separar:

- decisiones de capacidad / optimization;
- de decisiones run-level que luego existirán de verdad en `F7`.

## Compatibilidad con F7

La mini-revalidacion debe dejar trazado si una configuracion:

- funciona tecnicamente;
- mejora o empeora la metrica principal;
- es compatible o no con la futura familia `imbalance_aware`.

En particular, cualquier `batch_size` candidato a baseline comun de campaña debe ser compatible con el requisito actual de `balanced + cycle_reals`, que exige divisibilidad por numero de clases.

## Artefactos esperados

La ejecucion debe dejar:

- runs canonicas `MLP` con `results.yaml`, `metrics_long.csv` y `run_manifest.json`;
- una tabla de runs por seed;
- una tabla agregada por configuracion;
- un chequeo por hiperparametro / compatibilidad;
- un reporte breve con ranking y recomendacion final;
- un chequeo final en `test` solo para la configuracion ganadora.

## Fuente machine-readable

La especificacion ejecutable de esta mini-revalidacion vive en:

- [config/f7_mlp_baseline_revalidation_v1.yaml](../config/f7_mlp_baseline_revalidation_v1.yaml)
