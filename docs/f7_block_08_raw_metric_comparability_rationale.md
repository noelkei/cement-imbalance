# F7 Block 08 Raw Metric Comparability Rationale

## Decision

El bloque `8` congela la política de comparabilidad de métricas en `raw space` para `F7`.

El artefacto canónico es:

- [config/f7_raw_metric_contract_v1.yaml](../config/f7_raw_metric_contract_v1.yaml)

## Qué queda fijado

Toda run `campaign-valid` de:

- `MLP`
- `XGBoost`

debe persistir en `raw space`, para:

- `train`
- `val`

y además:

- `test` solo es obligatorio cuando la run se declara como `holdout_run`
- en la operativa por defecto, una `selection_run` solo exige `train` y `val`

y para los tres scopes:

- `overall`
- `macro`
- `per_class`

el siguiente paquete mínimo de métricas:

- `r2`
- `mse`
- `rmse`
- `rrmse`
- `mape`

## Por qué no se fija una única métrica soberana

Este bloque no modela un search tipo `Optuna`.

La campaña no necesita una sola métrica para explorar automáticamente el espacio. Lo que necesita es:

- un paquete completo y comparable de métricas finales en `raw`;
- y una forma estable de leer y ordenar resultados por defecto.

Por eso se distingue entre:

- una `anchor metric` por defecto;
- y la política real de análisis, que podrá usar varias métricas raw en paralelo.

## Anchor metric

Se congela como anchor por defecto:

- `raw_real.macro.rrmse`

Su papel es:

- ranking por defecto;
- orden por defecto en tablas y resúmenes;
- lectura rápida coherente entre bloques.

No significa que sea la única métrica relevante del análisis final.

## Macro frente a overall

La campaña debe guardar siempre:

- `overall`
- `macro`
- `per_class`

`macro` no queda como algo opcional o “solo cuando aplique”.

Se exige siempre porque:

- evita que las clases grandes dominen toda la lectura;
- mantiene comparabilidad más limpia en un problema con desbalance;
- y permite comparar familias sin perder la señal de clases minoritarias.

`overall` sigue siendo obligatorio como vista complementaria, pero no reemplaza la necesidad de `macro`.

## Modos de run y política anti-leakage

El contrato no obliga por defecto a calcular `test` en toda run.

Se congelan dos modos explícitos:

- `selection_run`
  - requiere `train` y `val` en `raw`
- `holdout_run`
  - requiere `train`, `val` y `test` en `raw`

La resolución del modo se hace desde el flag de holdout explícito del runner.

Esto mantiene dos propiedades a la vez:

- comparabilidad fuerte en la superficie `raw`
- cumplimiento de la política anti-leakage del proyecto, donde `test` no debe usarse por defecto en la operativa canónica

## Regla de validez de campaña

Si una run no puede invertir correctamente a `raw` y cerrar su paquete mínimo de métricas en `raw`, entonces:

- no es `campaign-valid`.

Esto aplica especialmente a `MLP` cuando haya `y_transform`, pero la regla vale para toda la campaña:

- no basta con una métrica nativa de entrenamiento;
- no basta con métricas solo en espacio transformado;
- y no basta con una inversión parcial o ambigua.

## Artefactos necesarios

Para que la inversión a `raw` sea reproducible, deben persistirse explícitamente:

- `y_transform`
- `value_space`
- scaler del target cuando aplique

Esto es parte del contrato de comparabilidad, no un extra opcional.

## Comparabilidad entre familias

`MLP` y `XGBoost` deben compararse sobre la misma superficie final:

- métricas `raw`
- sobre los mismos splits
- con los mismos scopes exigidos

No deben compararse usando:

- la loss nativa de entrenamiento;
- ni la `eval_metric` de `XGBoost` como si fuese ya el resultado final suficiente;
- ni métricas en espacio transformado como criterio principal de campaña.

## Política de análisis posterior

El análisis estadístico principal no queda reducido a una sola métrica.

Las vistas recomendadas como núcleo del análisis son:

- `raw_real.macro.r2`
- `raw_real.macro.rrmse`
- `raw_real.per_class.r2`
- `raw_real.per_class.rrmse`

Eso deja espacio para:

- efectos individuales;
- comparaciones por clase;
- y análisis de sinergias más adelante,

sin romper la política base de comparabilidad.

## Rationale resumido

- toda run debe cerrar con un paquete completo de métricas en `raw`;
- ese paquete es obligatorio para `train` y `val` siempre, y para `test` solo en runs de holdout explícitas;
- los scopes `overall`, `macro` y `per_class` son siempre obligatorios;
- las métricas transformadas pueden existir, pero no gobiernan la comparación principal;
- la `anchor metric` por defecto es `raw_real.macro.rrmse`;
- y una run que no pueda producir métricas raw invertidas correctamente no es válida para campaña.
