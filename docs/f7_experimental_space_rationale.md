# F7 Experimental Space Rationale

## Proposito y estatus

Este documento fija la gramatica experimental que conviene contemplar antes de:

- formalizar del todo el contrato de `F7`;
- implementar nuevas `synthetic_policy`;
- cerrar la shortlist final de combinaciones a ejecutar.

No es:

- un contrato congelado;
- una parrilla final;
- una obligacion de ejecutar toda la combinatoria posible.

Si es:

- un rationale de trabajo;
- una taxonomia de ejes y compatibilidades;
- un puente entre el cierre de `FlowGen` y la fase downstream con `MLP` y baselines comparables.

Estado aguas arriba asumido:

- `FlowPre official`: cerrado;
- `FlowGen official`: cerrado con winner unico;
- `FlowPre train_only`: cerrado localmente;
- `FlowGen train_only`: cerrado localmente como rama experimental;
- `F7`: sigue abierto como fase de comparacion downstream, no como reapertura de `FlowGen`.

## Principios metodologicos que gobiernan F7

- split oficial unico: `init_temporal_processed_v1`;
- todo lo que aprende parametros se ajusta solo con `train`;
- `val` se usa para seleccion;
- `test` permanece bloqueado salvo confirmacion final opt-in;
- comparaciones justas bajo mismo `base config`, mismo `seed set` y misma politica de seleccion;
- separar de forma estricta:
  - `dataset construction`
  - `training behavior`
- distinguir siempre:
  - rama canonica `official`
  - rama experimental local `train_only`

Notas operativas:

- las politicas sinteticas son siempre `train-only`;
- `val/test` no deben mutarse para acomodar sinteticos;
- la rama `train_only` puede alimentar `F7`, pero no sustituye el winner temporal oficial ni reabre el canon.

## Vista general de la gramatica experimental

La superficie experimental de `F7` se organiza en dos capas:

1. `dataset-level axes`
2. `run-level axes`

Regla principal:

- una misma variante `dataset-level` puede ejecutarse bajo varias decisiones `run-level`;
- no todos los ejes aplican con el mismo sentido a todas las familias de modelo.

### Resumen de capas

| capa | que controla | persiste dataset nuevo | ejemplos |
| --- | --- | --- | --- |
| `dataset-level` | identidad del bundle que entra a downstream | si, cuando cambia filas o representacion persistida | `x_transform`, `y_transform`, `synthetic_policy` |
| `run-level` | comportamiento de entrenamiento/seleccion sin redefinir el bundle | no | `batch_policy`, `cycling_policy`, `loss_policy`, `objective_metric`, `allow_synth` |

## Ejes dataset-level

Los ejes `dataset-level` son los que definen la identidad canonica del dataset que entra en `F7`.

### Tabla resumen

| eje | estado principal | etiquetas utiles | nota |
| --- | --- | --- | --- |
| `x_transform` | `implemented_now` y `supported_space` | `materialized_now`, `supported_space` | incluye escalados clasicos y variantes `FlowPre` |
| `y_transform` | `implemented_now` | `materialized_now` | eje independiente de `x_transform` |
| `synthetic_policy` | `implemented_now` para `none` y `kmeans_smote` | `blocked_by_upstream`, `local-only candidate`, `future_optional` | muta solo `train` |

### `x_transform`

Espacio soportado hoy:

- clasicos:
  - `standard`
  - `robust`
  - `minmax`
  - `quantile`
- derivados de `FlowPre`:
  - `flowpre_rrmse`
  - `flowpre_mvn`
  - `flowpre_fair`
  - `flowpre_candidate_1`
  - `flowpre_candidate_2`

Lectura recomendada:

- `materialized_now` para la base clasica oficial: los `16` bundles deterministas `X x Y` ya materializados;
- `supported_space` para transforms `FlowPre` cuyo uso downstream tiene sentido aunque no toda la superficie este materializada del mismo modo;
- `flowpre_candidate_1` y `flowpre_candidate_2` deben leerse sobre todo como bases de trabajo para `FlowGen`, no como "mejores transforms" universales por si mismas.

Papel de cada familia `FlowPre`:

- `flowpre_rrmse`: upstream especializado en realism bajo un criterio centrado en error relativo;
- `flowpre_mvn`: upstream mas alineado con normalidad/gaussianidad del espacio transformado;
- `flowpre_fair`: upstream orientado a equilibrio entre clases;
- `flowpre_candidate_1` y `flowpre_candidate_2`: work bases promovidas para la fase `FlowGen`, y por extension candidatas naturales cuando downstream quiera heredar esa rama.

### `y_transform`

Espacio soportado hoy:

- `standard`
- `robust`
- `minmax`
- `quantile`

Reglas:

- `y_transform` es un eje independiente de `x_transform`;
- no debe asumirse que el mejor `x_transform` arrastra automaticamente el mejor `y_transform`;
- el contrato de `F5` soporta una superficie mas amplia que la shortlist final que acabe corriendose en `F7`.

### `synthetic_policy`

#### Valores a contemplar en el rationale

| valor | estatus en este documento | estado hoy en repo | nota |
| --- | --- | --- | --- |
| `none` | `implemented_now` | implementado y usable | sin sinteticos persistidos |
| `flowgen_official` | `blocked_by_upstream` a nivel contractual, pero semanticamente cerrado aguas arriba | existe winner oficial ya cerrado | downstream debera decidir si lo incluye y bajo que policy id final |
| `flowgen_train_only` | `local-only candidate` | existe finalist local ya cerrado | util para comparar downstream; no canoniciza la rama |
| `kmeans_smote_joint` | `implemented_now` | implementado como `synthetic_policy = kmeans_smote` sobre bundles canonicos no sinteticos | baseline sintetico no entrenado pendiente de decision de shortlist |
| `random_oversample` | `future_optional` | no implementado aun como policy dataset-level | baseline simple de suelo |

#### Nota importante sobre naming

El contrato y parte del codigo actual hablan de una familia `flowgen` mas general. Para `F7`, conviene distinguir semanticamente:

- `flowgen_official`
- `flowgen_train_only`

aunque la resolucion final de nombres/ids pueda hacerse mas adelante en el contrato definitivo o en los manifests concretos.

#### Que significa que `synthetic_policy` sea dataset-level

Una `synthetic_policy` dataset-level:

- muta solo `train`;
- no toca `val/test`;
- persiste filas nuevas o reequilibra filas persistidas de `train`;
- requiere `synthetic_policy_id`;
- requiere `is_synth` cuando haya filas sinteticas nuevas;
- requiere manifiestos de procedencia;
- requiere conteos por split;
- requiere conteos por clase.

#### `random_oversample` no equivale a `cycling`

No conviene mezclar estas dos ideas:

| concepto | capa | que hace |
| --- | --- | --- |
| `random_oversample` | `dataset-level` | crea un `train` persistido aumentado o reequilibrado |
| `cycling` | `run-level` | repite reales dentro del dataloader de `MLP` sin crear un nuevo dataset persistido |

Consecuencia metodologica:

- `random_oversample` podria servir tanto a `MLP` como a `XGBoost`;
- `cycling` pertenece al comportamiento de `MLP` y no sustituye una policy sintetica dataset-level.

## Ejes run-level

Los ejes `run-level` cambian el comportamiento de entrenamiento o de seleccion, pero no redefinen la identidad del dataset.

### Tabla resumen

| eje | estado | aplica directamente a | nota |
| --- | --- | --- | --- |
| `model_family` | parcialmente formalizado en scripts, no en el contrato `closure_contract_v1` | `MLP`, `XGBoost` | util como capa de organizacion conceptual de `F7` |
| `batch_policy` | `implemented_now` | sobre todo `MLP` | `baseline` o `balanced` |
| `cycling_policy` | `implemented_now` | `MLP` | `cycle_reals=True/False` |
| `loss_policy` | `implemented_now` | `MLP` | combina `loss_reduction` y `regression_group_metric` |
| `objective_metric` | `implemented_now_post_run_only` | comparacion/seleccion | no es aun objective trainer-native |
| `allow_synth` | `implemented_now` | `MLP` | filtra o permite filas sinteticas |

### `model_family`

Familias a contemplar:

- `mlp`
- `xgboost`

Lectura recomendada:

- `MLP` es la ruta principal canonicamente prevista para `F7`;
- `XGBoost` puede contemplarse como baseline opcional si ayuda a responder si el beneficio proviene de la politica de datos o del tipo de modelo.

### `batch_policy`

Valores actuales:

- `baseline`
- `balanced`

Semantica real en `MLP`:

- `baseline`
  - usa un dataloader estandar;
  - si existe columna `is_synth`, se elimina para que no entre como feature;
  - no impone batches balanceados.
- `balanced`
  - usa preparacion especifica por clase;
  - se relaciona con `cycle_reals`;
  - permite construir entrenamiento mas balanceado a nivel de batches.

### `cycling_policy`

Valores actuales:

- `cycle_reals=True`
- `cycle_reals=False`

Semantica real en `MLP`:

- `cycle_reals=True`
  - cicla solo reales;
  - no cicla sinteticos;
  - balancea usando el total `real + synth` como referencia por clase;
  - si `batch_policy=balanced`, exige `batch_size` divisible por numero de clases.
- `cycle_reals=False`
  - deja el dataset final en proporcion mas natural;
  - no introduce repeticion extra de reales en el dataloader balanceado.

### `loss_policy`

El abanico actual ya implementado combina dos subejes.

#### `loss_reduction`

- `overall`
- `per_class_equal`
- `per_class_weighted`

#### `regression_group_metric`

- `mse`
- `rmse`
- `rrmse`

Lectura metodologica:

- `loss_reduction` controla como agregamos el error entre clases;
- `regression_group_metric` controla con que magnitud lo medimos;
- para el TFG, estas decisiones importan porque codifican que entendemos por comparacion justa entre clases mayoritarias y minoritarias.

### `objective_metric`

Hoy debe leerse como eje de seleccion y comparacion post-run, no como objective trainer-native real.

Estado actual:

- `implemented_now_scope`: `post_run_selection`
- `trainer_early_stopping_metric`: `val_loss`
- `default_id`: `raw_real.macro.rrmse`
- tie-breakers actuales:
  - `raw_real.worst_class.rrmse`
  - `raw_real.overall.rrmse`

Consecuencia:

- el documento puede contemplar `objective_metric` como eje de `F7`;
- pero no debe afirmar que hoy el trainer optimiza directamente esa metrica.

### `allow_synth`

Rol real:

- `allow_synth=True`
  - mantiene filas sinteticas si el dataset las trae;
- `allow_synth=False`
  - filtra filas sinteticas de `train/val/test` antes del entrenamiento.

Importancia:

- permite usar el mismo dataset materializado para comparar una corrida que consume sinteticos y otra que los ignora;
- no sustituye `synthetic_policy`, porque no crea ni redefine datasets.

## Compatibilidades y restricciones por familia de modelo

### Resumen corto

| eje / decision | `MLP` | `XGBoost` |
| --- | --- | --- |
| `x_transform` | si | opcional, no camino primario por defecto |
| `y_transform` | si | normalmente irrelevante como transform de entrada; se contemplaria solo si downstream lo requiere explicitamente |
| `synthetic_policy` | si | si, si existe bundle persistido compatible |
| `batch_policy` | si | no en el mismo sentido |
| `cycling_policy` | si | no |
| `loss_policy` | si | no en el mismo sentido |
| `objective_metric` | si, como seleccion post-run | si, como criterio comparativo externo |
| `allow_synth` | si | no como eje trainer-native equivalente |

### `MLP`

Para `MLP` tiene sentido contemplar:

- transforms en `X`;
- transforms en `Y`;
- `synthetic_policy`;
- `batch_policy`;
- `cycling_policy`;
- `loss_policy`;
- `objective_metric`;
- `allow_synth`.

Restricciones operativas relevantes:

- `balanced + cycle_reals=True` exige `batch_size` divisible por numero de clases;
- `allow_synth=False` filtra filas sinteticas si existen;
- `baseline` y `balanced` no son equivalentes y no deben mezclarse como si fueran solo un detalle cosmético.

### `XGBoost`

Ruta principal esperada:

- datos `raw`;
- opcionalmente con `synthetic_policy` dataset-level si existe bundle persistido comparable.

Lectura recomendada:

- no usar `XGBoost` como excusa para heredar todo el espacio de transforms como si fueran igual de naturales;
- si `XGBoost` entra, deberia entrar primero como baseline sencillo sobre `raw + synthetic_policy`, no como reexpresion total de toda la gramática de `MLP`.

### Combinaciones a evitar o no priorizar

- tratar `cycling` como si sustituyera `random_oversample`;
- asumir que `XGBoost` debe recorrer la misma superficie de transforms que `MLP`;
- mezclar ramas `official` y `train_only` sin dejar trazabilidad explicita de procedencia;
- tratar una `synthetic_policy` no implementada todavia como si ya fuese parte del contrato congelado;
- definir shortlist final antes de tener cerrada la semantica de las nuevas policies.

## Diseno conceptual de la nueva rama no entrenada

### `kmeans_smote_joint`

`kmeans_smote_joint` ya esta implementado como politica concreta bajo `synthetic_policy = kmeans_smote`, pero todavia no esta promovido automaticamente a la shortlist final de `F7`.

Objetivo:

- introducir una rama sintetica no entrenada;
- comparar si una heuristica geometrica local basta para competir con `FlowGen`;
- mantener la comparacion dentro del split temporal y sin leakage.

### Semantica conceptual propuesta

- opera solo en `train`;
- condicionada por `type`;
- trabaja en el espacio transformado de la variante, no en raw;
- usa geometria conjunta `[X, y]`;
- clusteriza localmente;
- interpola solo dentro del mismo cluster;
- genera filas persistidas con metadata canonica.

### Por que en espacio transformado y no raw

Para `FlowGen` tiene sentido pensar en una fuente sintetica en raw que luego se reexpresa por variante, porque es un generador aprendido.

Para una heuristica tipo `KMeans-SMOTE`, la geometria:

- define los vecinos;
- define los clusters;
- define las interpolaciones validas.

Por eso, en esta rama, la variante transformada no es un paso cosmetico posterior sino parte de la propia definicion del algoritmo.

### Comparacion conceptual entre ramas sinteticas

| rama | aprende modelo generativo | espacio conceptual natural | persistencia esperada |
| --- | --- | --- | --- |
| `flowgen_official` | si | raw del flujo generativo, luego reexpresion por variante | dataset augmentado persistido |
| `flowgen_train_only` | si | raw del flujo generativo local, luego reexpresion por variante | dataset augmentado persistido local |
| `kmeans_smote_joint` | no | variante transformada, por `type` y sobre `[X, y]` | dataset augmentado persistido |
| `random_oversample` | no | no depende de geometria sofisticada | dataset reequilibrado persistido |

### Baseline de suelo: `random_oversample`

Tiene sentido contemplarlo como baseline simple porque responde a otra pregunta:

- si ya mejoraramos solo repitiendo `train` minoritario, cuanto del beneficio viene de rebalancear y cuanto de sintetizar de verdad.

Recomendacion:

- mantenerlo como `future_optional` o baseline simple;
- no confundirlo con `cycling`;
- no exigirle la misma semantica generativa que a `kmeans_smote_joint`.

## Que decisiones cierra este documento y cuales deja abiertas

### Lo que este documento si debe cerrar

- la separacion `dataset-level` vs `run-level`;
- la pertenencia de cada eje a una capa concreta;
- la semantica general de `MLP` frente a `XGBoost`;
- la distincion entre:
  - `synthetic_policy`
  - `cycling`
- el papel conceptual de:
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote_joint`
  - `random_oversample`

### Lo que este documento no debe cerrar

- la shortlist exacta de `F7`;
- si `XGBoost` entra o no definitivamente;
- si `train_only` entra o no en la shortlist final;
- el numero final de combinaciones a ejecutar;
- los hiperparametros concretos de `kmeans_smote_joint`;
- los nombres definitivos de policy ids si requieren un ajuste fino del contrato.

## Recommended workflow despues de este documento

Secuencia recomendada:

1. cerrar este rationale;
2. decidir que `synthetic_policy` nuevas se aprueban para implementacion;
3. materializar datasets de prueba con manifests y metadata canonica;
4. despues cerrar la shortlist final de `F7`.

Regla practica:

- no saltar directamente a la parrilla final mientras la semantica de una nueva `synthetic_policy` siga ambigua;
- no formalizar un contrato definitivo de `F7` sin haber aclarado antes que ejes pertenecen a dataset y cuales a entrenamiento.

## Fuentes del repo que anclan este rationale

- `docs/project_context.md`
- `docs/implementation_status.md`
- `docs/experimental_train_only.md`
- `docs/target_architecture.md`
- `docs/phase_map.md`
- `config/closure_contract_v1.yaml`
- `config/mlp_closure_base_v1.yaml`
- `data/dataset_contract.py`
- `data/sets.py`
- `training/train_mlp.py`
- `scripts/run_xgboost_temporal_vs_legacy.py`

## Criterio de lectura final

Este documento debe leerse como:

- mas concreto que un brainstorming;
- menos rigido que un contrato final;
- suficiente para implementar la siguiente capa de trabajo sin confundir:
  - estado actual;
  - espacio soportado;
  - ramas candidatas aun no implementadas.
