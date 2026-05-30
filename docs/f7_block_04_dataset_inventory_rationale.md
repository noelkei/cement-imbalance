# F7 Block 04 Dataset Inventory Rationale

## Decision

El bloque `4` queda cerrado en su parte lógica y de identidad de inventario.

Queda fijado que:

- `MLP` usa exactamente `96` datasets:
  - `6 x_base`
  - `4 y_transform`
  - `4 synthetic_policy`
- `XGBoost` usa exactamente `4` datasets:
  - una sola base raw fija
  - `4 synthetic_policy`
- el inventario machine-readable canónico queda en:
  - [config/f7_dataset_inventory_v1.yaml](../config/f7_dataset_inventory_v1.yaml)
  - [config/f7_dataset_inventory_v1.csv](../config/f7_dataset_inventory_v1.csv)

## Qué queda fijado

### 1. Grid exacto de `MLP`

Se confirma sin excepciones:

- `x_base`:
  - `candidate_1`
  - `candidate_2`
  - `standard`
  - `robust`
  - `quantile`
  - `minmax`
- `y_transform`:
  - `standard`
  - `robust`
  - `quantile`
  - `minmax`
- `synthetic_policy`:
  - `none`
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`

### 2. Base fija de `XGBoost`

Se confirma sin reabrir:

- dataset oficial raw/no-scale;
- `feature_policy = raw_numeric_plus_type_onehot`;
- `synthetic_policy`:
  - `none`
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`

### 3. Política de materialización

La materialización final de datasets no se hace todavía en este bloque.

Se fija que:

- primero se congela el inventario lógico;
- luego se alinea el cap sintético del `50%` en el bloque `5`;
- después se materializa por adelantado;
- y finalmente se hace freeze estricto del inventario ya validado.

### 4. Política de mutación

Si cambia una receta validada:

- no se toca silenciosamente el inventario existente;
- se crea una versión nueva.

La política es:

- inventario inmutable y versionado.

### 5. Naming

Se fija un naming composicional corto y estable:

- `mlp__x-<xbase>__y-<ytransform>__syn-<policy>`
- `xgb__x-raw-base-v1__y-raw__syn-<policy>`

Esto prioriza:

- legibilidad humana;
- estabilidad;
- y fácil composición posterior en manifests, runners y reporting.

## Por que los datasets sintéticos no dependen de seed

Se consideró la posibilidad de generar una realización sintética distinta por seed de `MLP`.

La decisión fue no incorporarlo a `F7` principal.

La razón es metodológica:

- mezclaría variabilidad del generador con variabilidad de entrenamiento;
- cambiaría la semántica actual de las `30` seeds;
- rompería la interpretación de `synthetic_policy` como dataset fijo;
- y obligaría a redefinir conteos, metadatos e identidad de dataset.

Por tanto, en `F7`:

- cada `synthetic_policy` se trata como dataset fijo y congelable;
- la variabilidad del generador, si se estudia, debe ir en un subestudio aparte.

## Cierre posterior de `4B`

Tras cerrar el bloque `5`, la fase `4B` quedó completada el `2026-05-18`.

El batch canónico de materialización final es:

- `outputs/reports/f7_dataset_materialization/f7_dataset_materialization_20260518T132351821306Z/` (`local-only`)

Resultado:

- `96` datasets `MLP`
- `4` datasets `XGBoost`
- `2` pools compartidos `FlowGen`
- `0` fallos

Ubicación canónica de los datasets materializados:

- `data/sets/official/init_temporal_processed_v1/` (`local-only`, no subible)

Subárboles relevantes:

- `raw/`
- `scaled/`
- `synthetic_pools/`
- `augmented_scaled/`
- `xgboost/`
- `meta/`

Separación posterior aplicada en el árbol local:

- `legacy_pre_f7/` contiene artefactos históricos o de smoke que no deben consumirse en campaña;
- `meta/f7_canonical_materialized_inventory_v1.csv` fija localmente los ids y manifests exactos del batch canónico.

Por tanto, este bloque ya no debe leerse como `4A` cerrado y `4B` pendiente, sino como:

- `4A`:
  - inventario lógico
  - naming
  - tabla machine-readable
- `4B`:
  - materialización final
  - freeze de manifests
  - fingerprints estables
  - batch report canónico

## Rationale resumido

- primero había que fijar la tabla exacta de datasets;
- el inventario principal de `F7` queda ya cerrado en `96 + 4`;
- la materialización final se hará por adelantado, no bajo demanda;
- si cambia una receta, se versiona un inventario nuevo;
- y los datasets sintéticos siguen siendo datasets fijos, no realizaciones por seed.
