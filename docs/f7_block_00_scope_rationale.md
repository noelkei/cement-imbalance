# F7 Block 00 Scope Rationale

## Proposito

Este documento concentra el rationale del bloque `0` de preparacion de `F7`:

- reconfirmar el alcance de la campaña;
- dejar fijado el perimetro metodológico;
- y evitar que los bloques posteriores reabran supuestos base ya cerrados.

## Decision

Queda reconfirmado como alcance vigente de `F7`:

- campaña completa objetivo de `17400` runs;
- `MLP` como familia principal;
- `XGBoost` como baseline acotada;
- `96` datasets `MLP`;
- `4` datasets `XGBoost`;
- `30` seeds como parte del diseño base;
- cap sintético del `50%` aplicable por igual a:
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`

## Por que era importante cerrarlo primero

El bloque `0` no resolvía una duda técnica pequeña, sino el perímetro de toda la campaña.

Sin este cierre previo, los bloques siguientes podían acabar reabriendo:

- el tamaño real de la campaña;
- el papel de `XGBoost`;
- la semántica de las `30` seeds;
- y la comparabilidad entre policies sintéticas.

Eso habría contaminado:

- el baseline de `MLP`;
- el baseline de `XGBoost`;
- el inventario de datasets;
- el plan de runners;
- y el benchmark real de coste.

## Qué se decidió no reabrir

Dentro de este bloque se fijó que no se reabre, salvo decisión nueva explícita:

- convertir `XGBoost` en un segundo subestudio con grid propio;
- tratar las `30` seeds como ampliación opcional;
- reinterpretar el cap del `50%` de forma desigual entre ramas sintéticas;
- rediseñar el alcance hacia una campaña más pequeña por defecto.

## Qué se aplazó deliberadamente

No se quiso mezclar aquí la posible existencia de un piloto previo.

La decisión fue:

- fijar ya la campaña completa objetivo;
- y dejar la operativa de un eventual piloto para bloques posteriores.

## Impacto sobre el resto del plan

Este cierre obliga a que los siguientes bloques hereden el mismo perímetro:

- `1` y `2` congelan baselines para la campaña completa, no para una fase reducida;
- `4` y `5` deben construir datasets respetando el cap sintético ya fijado;
- `6` debe cerrar un panel real de `30` seeds;
- `12` y `13` deben materializar una campaña de `17400` runs, no una versión reinterpretada.

## Rationale resumido

- primero había que cerrar qué campaña estábamos preparando realmente;
- la campaña sigue siendo la completa de `17400` runs;
- `MLP` sigue siendo la familia principal;
- `XGBoost` sigue siendo baseline acotada;
- las `30` seeds siguen siendo parte del diseño base;
- y el cap sintético del `50%` sigue siendo homogéneo.
