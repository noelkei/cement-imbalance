# F7 Run Plan 17400

## Proposito y estatus

Este documento fija la parrilla operativa actualmente prevista para `F7`.

No es:

- el contrato final de ejecucion automatizada;
- la shortlist final post-analisis;
- un reporte de resultados.

Si es:

- el plan de runs que queremos preservar por escrito;
- la referencia de conteo y combinatoria para no perder la decision tomada;
- la base para formalizar despues manifests, runners y campaign ids.

Este documento se apoya en:

- [docs/f7_experimental_space_rationale.md](f7_experimental_space_rationale.md)
- [docs/f7_trial_traceability_rationale.md](f7_trial_traceability_rationale.md)

## Decisiones ya fijadas

- split oficial unico: `init_temporal_processed_v1`
- comparacion downstream principal con `MLP`
- baseline adicional con `XGBoost`
- `30` seeds por run
- las `synthetic_policy` se tratan como datasets fijos, no como sub-sweeps abiertos
- la masa sintetica queda capada a un maximo del `50%` del numero real de cada clase minoritaria
- `MLP` se ejecuta con dos familias de training behavior:
  - `plain`
  - `imbalance_aware`
- `MLP` usa exactamente tres `loss_policy`
- `XGBoost` no barre escalados ni losses de entrenamiento como sub-grid

## Regla fija de masa sintetica

Para cada clase minoritaria `c`:

- `n_synth(c) <= floor(0.5 * n_real(c))`

Lectura practica:

- el dataset final de esa clase queda, como maximo, con composicion aproximada:
  - `2/3` real
  - `1/3` sintetico
- la policy evita que los sinteticos dominen la clase
- la policy se interpreta como guardrail comun para:
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`

### Implicacion con la distribucion actual de `train`

En la distribucion real actual:

- existe una clase mayoritaria que no necesita sintéticos;
- las clases minoritarias reciben un cap de hasta `50%` de su masa real;
- el desbalance se reduce, pero no desaparece.

Ejemplo conceptual:

- si una clase minoritaria tiene `n_real(c)` filas reales, su tope sintético pasa a ser `floor(0.5 * n_real(c))`;
- su total máximo tras augmentación pasa a `n_real(c) + floor(0.5 * n_real(c))`.

Esto implica que el desbalance sigue existiendo incluso despues de la augmentacion, y por tanto sigue teniendo sentido probar un regimen `imbalance_aware`.

## Dataset-level plan para `MLP`

### Nivel `x/base`

| eje | valores |
| --- | --- |
| `x_base` | `candidate_1`, `candidate_2`, `standard`, `robust`, `quantile`, `minmax` |

Total:

- `6`

### Nivel `y_transform`

| eje | valores |
| --- | --- |
| `y_transform` | `standard`, `robust`, `quantile`, `minmax` |

Total:

- `4`

### Nivel `synthetic_policy`

| eje | valores |
| --- | --- |
| `synthetic_policy` | `none`, `flowgen_official`, `flowgen_train_only`, `kmeans_smote` |

Total:

- `4`

### Conteo dataset-level `MLP`

Formula:

- `6 x 4 x 4 = 96`

Total:

- `96` datasets candidatos para `MLP`

## Dataset-level plan para `XGBoost`

### Regla principal

`XGBoost` no se barre sobre todo el espacio de escalados de `X` e `y`.

Se usara:

- una sola representacion canonica fija, preferiblemente la variante mas raw/no-scale utilizable;
- sobre esa base fija, solo se compara `synthetic_policy`.

### Nivel `synthetic_policy`

| eje | valores |
| --- | --- |
| `synthetic_policy` | `none`, `flowgen_official`, `flowgen_train_only`, `kmeans_smote` |

### Conteo dataset-level `XGBoost`

Total:

- `4` datasets candidatos para `XGBoost`

## Run-level plan para `MLP`

### Familias `batch/cycling`

Se prueban en todos los datasets `MLP`, incluyendo `synthetic_policy = none`.

| family | `batch_policy` | `cycling_policy` |
| --- | --- | --- |
| `plain` | `baseline` | `false` |
| `imbalance_aware` | `balanced` | `true` |

Total:

- `2`

### `loss_policy`

| loss id | descripcion |
| --- | --- |
| `overall_rmse` | agregacion global + `rmse` |
| `per_class_equal_rmse` | agregacion `per_class_equal` + `rmse` |
| `per_class_equal_rrmse` | agregacion `per_class_equal` + `rrmse` |

Total:

- `3`

### Conteo run-level `MLP`

Por dataset:

- `2 x 3 = 6`

Sobre todos los datasets `MLP`:

- `96 x 6 = 576`

Total estructural:

- `576` combinaciones `MLP` por seed

## Run-level plan para `XGBoost`

### Regla principal

`XGBoost` entra como baseline fuerte y acotada:

- una sola config de entrenamiento;
- una sola politica de features;
- sin barrer variantes `per_class_equal` ni losses de entrenamiento alternativas como sub-grid propio.

### Conteo run-level `XGBoost`

Por dataset:

- `1`

Sobre todos los datasets `XGBoost`:

- `4 x 1 = 4`

Total estructural:

- `4` combinaciones `XGBoost` por seed

## Seeds

Panel fijado:

- `30` seeds por run

Lectura:

- el panel es unico y compartido por las comparaciones que quieran ser estadisticamente homologas;
- cualquier agregado o ANOVA futuro debera declarar explicitamente este `seed_set_id`.

## Conteo total de runs

### `MLP`

- datasets: `96`
- configs por dataset: `6`
- seeds: `30`

Formula:

- `96 x 6 x 30 = 17280`

Total:

- `17280` runs `MLP`

### `XGBoost`

- datasets: `4`
- configs por dataset: `1`
- seeds: `30`

Formula:

- `4 x 1 x 30 = 120`

Total:

- `120` runs `XGBoost`

### Total global

Formula:

- `17280 + 120 = 17400`

Total:

- `17400` runs

## Resumen final por niveles

| nivel | `MLP` | `XGBoost` |
| --- | --- | --- |
| dataset `x/base` | `6` | `1` fijo |
| dataset `y_transform` | `4` | no se barre |
| dataset `synthetic_policy` | `4` | `4` |
| datasets totales | `96` | `4` |
| familias `batch/cycling` | `2` | no aplica |
| `loss_policy` / training config | `3` | `1` |
| combinaciones por seed | `576` | `4` |
| seeds | `30` | `30` |
| total runs | `17280` | `120` |

## Lo que este plan deja fuera

- multiples configs de `XGBoost`;
- variantes `per_class_equal` para `XGBoost` como grid de entrenamiento;
- multiples caps de masa sintetica;
- multiples `synthetic_seed` por policy;
- nuevas `synthetic_policy` adicionales;
- reapertura de finalistas `FlowPre` descriptivos como bases completas extra;
- paneles alternativos de seeds dentro de la misma campaña.

## Uso recomendado de este documento

Este documento debe servir como referencia para:

- nombrar la campaña de `F7`;
- construir manifests de trial;
- preparar runners y campaign configs;
- comprobar conteos antes de lanzar la ejecucion.
