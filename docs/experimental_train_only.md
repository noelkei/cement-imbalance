# Experimental Train-Only Flow Line

Este documento define la rama paralela `train-only` para `FlowPre` y `FlowGen`.
No reabre el cierre oficial de `FlowPre` ni el cierre oficial de `FlowGen`.

## Rol metodologico

La rama `train-only` existe para estudiar una hipotesis downstream concreta:
un generador ajustado al dominio de `train` podria ser mas util para balancear
`train` antes de entrenar el `MLP`, aunque no sea el mejor generador bajo
validacion temporal.

Por tanto, esta rama:

- no sustituye el canon vigente;
- no busca un nuevo winner oficial de `FlowGen`;
- no reabre `F6a` ni `F6b`;
- no invalida `outputs/models/official/flowgen_finalist/`;
- debe leerse como experimental, paralela y complementaria.

## Politica de monitorizacion

La capacidad se activa mediante:

```text
monitoring_policy = "train_only"
```

El comportamiento por defecto sigue siendo:

```text
monitoring_policy = "official_val"
```

En `official_val`, la clave de resultados `"val"` conserva su semantica canonica:
`val_selection` temporal.

En `train_only`, la clave `"val"` se conserva solo por compatibilidad tecnica con
los trainers, flatteners y artefactos existentes. Su rol real pasa a ser:

```text
train_monitor_pseudo_val
```

Eso significa:

- la superficie de monitorizacion se deriva de `train`;
- no es holdout temporal;
- no se debe usar para seleccionar un winner canonico;
- sus metricas no son comparables directamente con el `val_selection` oficial.

Los artefactos generados por esta rama deben incluir metadata de monitorizacion
con `monitor_is_holdout=false` y `canonical_selection_eligible=false`.

## Temporal realism

`temporal_realism` queda desactivado en modo `train_only`.

Esa capa esta disenada para contrastar generacion contra un holdout temporal real.
Si el monitor viene de `train`, escribir `val.temporal_realism` crearia una
semantica falsa. La rama puede conservar metricas generales de reconstruccion y
realismo sobre la pseudo-superficie, pero no debe fingir temporalidad de holdout.

## Namespace

Los runs de esta rama deben ir fuera de `official/`:

```text
outputs/models/experimental/train_only/flow_pre/
outputs/models/experimental/train_only/flowgen/
```

El namespace `official/` se mantiene reservado para el canon vigente y sus
artefactos de cierre ya materializados.

## Estado materializado actual

La rama ya no esta solo definida por politica. Hoy ya tiene cierres locales
materializados tanto para `FlowPre` como para `FlowGen`.

### `FlowPre train-only`

Los priors finales locales de `FlowPre train-only` viven en:

```text
outputs/models/experimental/train_only/flowpre_finalists/
```

Picks vigentes:

- `candidate_trainonly_1`
- `candidate_trainonly_2`

Su rol no es cerrar `FlowPre` de forma canonica, sino dejar listas las dos
bases reales de trabajo para `FlowGen train-only`.

### `FlowGen train-only`

La exploracion, confirmaciones, reseed y seleccion final local de `FlowGen
train-only` ya quedaron cerrados en:

```text
outputs/models/experimental/train_only/flowgen_finalist/
```

Winner local vigente:

- `flowgen_trainonly_tpv1_ct1_reseedfinal_r3a2_t06_clip125_seed15427_v1`

Interpretacion operativa:

- es el finalista local unico de la rama `train-only`;
- sirve como candidato experimental aguas abajo;
- no sustituye a `outputs/models/official/flowgen_finalist/`;
- no debe describirse como winner temporal oficial del proyecto.

## Uso downstream esperado

El valor de esta rama se decide despues, no por ranking generativo canonico. El
criterio real sera si los datasets balanceados con generadores `train-only`
ayudan en la comparacion downstream con `MLP`, siempre respetando:

- seleccion en `val` oficial para el cierre predictivo;
- `test` bloqueado por defecto;
- confirmacion final unica en `test` solo cuando corresponda.

## Regla de cierre de fase

La fase activa de esta rama ya no es explorar mas `FlowGen train-only`.

La decision abierta real pasa a ser otra:

- si el finalista local `train-only` entra o no en la shortlist downstream de
  `F7`;
- y, si entra, bajo que comparacion justa con el winner `official` y el resto
  de variantes candidatas para `MLP`.
