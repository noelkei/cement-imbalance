# F7 Run Plan 17400 Rationale

## Proposito y estatus

Este documento explica el rationale metodologico detras de la parrilla de `17400` runs fijada en [docs/f7_run_plan_17400.md](f7_run_plan_17400.md).

No es:

- un resultado experimental;
- una defensa estadistica final basada en evidencia ya observada;
- un contrato rigido de launcher.

Si es:

- la justificacion escrita de por que esta parrilla si y otras no;
- el rationale por niveles para poder defender el plan en el TFG;
- una guia para futuras reducciones, ampliaciones o shortlists sin perder la logica original.

## Principio general

La parrilla se ha construido para separar con claridad:

1. efecto de `dataset construction`
2. efecto de `training behavior`
3. efecto de `model family`
4. variabilidad debida a seeds

Regla central:

- no queremos que una conclusion sobre una `synthetic_policy` dependa de un unico espacio de representacion, de un unico regimen de entrenamiento o de una unica seed.

## Rationale del nivel `x/base` para `MLP`

### Opciones incluidas

- `candidate_1`
- `candidate_2`
- `standard`
- `robust`
- `quantile`
- `minmax`

### Por que entran `candidate_1` y `candidate_2`

- son las dos bases reales downstream heredadas del cierre de `FlowPre` / `FlowGen`;
- representan las dos work bases relevantes del camino generativo ya cerrado;
- si no entran, `F7` deja fuera el espacio downstream mas importante del proyecto.

### Por que entran `standard`, `robust`, `quantile` y `minmax`

- sirven como baseline clasica fuera del ecosistema `FlowPre` / `FlowGen`;
- permiten comprobar si el valor de una `synthetic_policy` es general o depende de una base flow-derived;
- añaden replicacion estructural para hablar de efectos medios de policy, no solo de casos puntuales.

### Por que no entran otras bases

Se dejan fuera como bases completas:

- `rrmse`
- `mvn`
- `fair`
- `candidate_trainonly_1`
- `candidate_trainonly_2`

Razones:

- `rrmse`, `mvn` y `fair` ya cumplen un rol descriptivo / de upstream especializado y reabrirlos como bases completas reabriria `FlowPre`;
- `candidate_trainonly_1` y `candidate_trainonly_2` duplicarian la dimension de bases y volverian la parrilla menos interpretable.

## Rationale del nivel `y_transform`

### Opciones incluidas

- `standard`
- `robust`
- `quantile`
- `minmax`

### Por que entran las cuatro

- `y_transform` no es cosmético: cambia la geometria del target;
- puede afectar la estabilidad del entrenamiento;
- puede afectar especialmente a losses relativas como `rrmse`;
- si dejamos solo una transform de `y`, parte de la conclusion sobre una `synthetic_policy` podria ser en realidad un artefacto de representacion del target.

### Por que no entran transforms adicionales

- abrir mas transforms de `y` convertiria `F7` en un estudio de target engineering;
- la parrilla ya es grande y necesita mantener un espacio clasico, defendible y cerrado.

## Rationale del nivel `synthetic_policy`

### Opciones incluidas

- `none`
- `flowgen_official`
- `flowgen_train_only`
- `kmeans_smote`

### Por que entra `none`

- es el control imprescindible;
- sin `none`, no se puede separar mejora debida a sinteticos de mejora debida a base/transform/modelo.

### Por que entra `flowgen_official`

- representa la rama oficial ya cerrada y promovida;
- si no entra, la parrilla ya no cubre el camino canonico del proyecto.

### Por que entra `flowgen_train_only`

- representa la mejor rama experimental local cerrada;
- permite responder si el camino `train_only` aporta valor downstream real o no.

### Por que entra `kmeans_smote`

- representa la alternativa sintetica no generativa, mas simple e interpretable;
- sirve como control frente a `FlowGen`;
- permite responder si hace falta un generador entrenado o si una policy sintetica estructurada mas simple ya compra señal.

### Por que no entran mas policies

Se dejan fuera:

- `random_oversample`
- multiples variantes de `kmeans_smote`
- multiples variantes de `flowgen_official` o `flowgen_train_only`

Razones:

- queremos que cada `synthetic_policy` sea una identidad fija de dataset;
- abrir sub-configs por policy convertiria `F7` en varios sub-estudios simultaneos y haria mucho mas dificil atribuir efectos.

## Rationale del cap sintetico del `50%`

### Regla

Para cada clase minoritaria:

- `n_synth <= 0.5 * n_real`

### Por que esta regla si

- impide que los sinteticos dominen una clase;
- deja una composicion final aproximada de `2/3` real y `1/3` sintetico;
- mantiene una señal real fuerte dentro de cada clase;
- hace comparables entre si las tres policies sinteticas;
- evita introducir otro eje oculto de comparacion basado en cuanta masa sintetica se mete.

### Por que no igualar a la mayoritaria

- igualar por completo llevaria a una generacion demasiado agresiva, especialmente en la clase mas pequeña;
- parte de la ventaja observada podria deberse a haber “inundado” la clase con sinteticos y no a la calidad de la policy.

### Por que no caps mas bajos

- caps como `10%` o `30%` reducen mucho la capacidad de corregir el desbalance;
- harian menos visible la diferencia entre `none` y las ramas sinteticas.

## Rationale de `batch/cycling` para `MLP`

### Familias incluidas

- `plain`
  - `batch_policy = baseline`
  - `cycling = false`
- `imbalance_aware`
  - `batch_policy = balanced`
  - `cycling = true`

### Por que se prueban en todos los datasets, incluso `none`

- permite separar mejor el efecto de `dataset construction` del efecto de `training behavior`;
- permite responder si un regimen imbalance-aware ya mejora algo incluso sin sinteticos;
- evita que la conclusion “los sinteticos ayudan” este contaminada por haber cambiado a la vez el regimen de entrenamiento solo en una parte de la parrilla.

### Por que no se deja `cycling` solo para datasets sinteticos

- aunque conceptualmente podria parecer suficiente, esa decision romperia la simetria del diseño;
- la comparacion entre `none` y sinteticos quedaria mas dificil de interpretar;
- al mantener ambas familias en todos los datasets, la parrilla conserva una gramatica mas limpia.

### Por que no se incluye `balanced + no cycling`

- la parrilla ya captura dos extremos claros:
  - control puro
  - regimen imbalance-aware fuerte
- una opcion intermedia añade combinatoria, pero reduce la nitidez interpretativa del diseno.

## Rationale de `loss_policy` para `MLP`

### Losses incluidas

- `overall + rmse`
- `per_class_equal + rmse`
- `per_class_equal + rrmse`

### Por que entra `overall + rmse`

- es la baseline natural y mas interpretable;
- deja un punto de comparacion claro contra el que medir cualquier regimen imbalance-aware.

### Por que entra `per_class_equal + rmse`

- aísla el efecto de igualar clases sin cambiar la unidad del error;
- permite distinguir entre mejora por fairness de clases y mejora por metrica relativa.

### Por que entra `per_class_equal + rrmse`

- representa la hipotesis mas fuerte de entrenamiento sensible a desbalance y escala relativa;
- es la opcion mas natural para preguntar si la mejor combinacion downstream es la mas “imbalance-aware”.

### Por que no entran otras losses

Se dejan fuera:

- `mse`
- `overall + rrmse`
- `per_class_weighted + *`

Razones:

- `mse` aporta poco frente a `rmse` y es menos interpretable;
- `overall + rrmse` es menos clara conceptualmente que las tres elegidas;
- `per_class_weighted` queda en una zona intermedia mas ambigua y complica mucho la lectura del estudio.

## Rationale de `XGBoost`

### Decision principal

`XGBoost` entra como baseline fuerte, pero no replica la parrilla completa de representaciones.

### Por que no se barre sobre todos los escalados

- `XGBoost` no necesita normalidad;
- los arboles suelen ser mucho menos sensibles al escalado monotónico de features;
- abrir un grid completo de `x_base` e `y_transform` para `XGBoost` generaria coste extra sin una hipotesis metodologica igual de fuerte que en `MLP`.

### Por que si se barre `synthetic_policy`

- esa si es la pregunta relevante para `XGBoost`:
  - si una policy sintetica mejora tambien una baseline tree-based fuerte;
- ayuda a responder si el beneficio es general o muy especifico de `MLP`.

### Por que no se abren variantes `per_class_equal` o varias losses de entrenamiento

- eso convertiría `XGBoost` en un sub-estudio aparte;
- el objetivo aqui no es tunear a fondo la baseline, sino tener una referencia fuerte y sobria;
- la comparabilidad con `MLP` es mas limpia si `XGBoost` mantiene una sola config de entrenamiento y multiples metricas de evaluacion downstream.

## Rationale del panel de `30` seeds

### Por que `30`

- el panel deja de ser una comprobacion rapida y pasa a ofrecer una base estadistica mucho mas fuerte;
- permite estimar medias y dispersion con bastante menos dependencia de una seed concreta;
- hace mas defendibles comparaciones family-level, ANOVA y post hoc posteriores.

### Coste asumido

- `30` seeds convierten la campaña en una ejecucion grande;
- el coste se acepta porque la parrilla esta pensada como plan serio de cierre comparativo, no como smoke test.

## Rationale del conteo total

### `MLP`

- `96` datasets
- `2` familias `batch/cycling`
- `3` losses
- `30` seeds

Formula:

- `96 x 2 x 3 x 30 = 17280`

### `XGBoost`

- `4` datasets
- `1` config
- `30` seeds

Formula:

- `4 x 1 x 30 = 120`

### Total

- `17280 + 120 = 17400`

### Lectura importante

El numero `17400` no es un objetivo arbitrario ni una cifra redonda forzada.

Es la consecuencia de:

- mantener una gramatica amplia y simetrica en `MLP`;
- acotar `XGBoost` a la pregunta que realmente importa;
- y exigir robustez estadistica fuerte con `30` seeds.

## Que comparaciones habilita bien esta parrilla

Este diseno permite defender preguntas como:

- si una `synthetic_policy` mejora de media en distintos espacios de `X` e `y`;
- si una `synthetic_policy` mejora solo en ciertas bases y no en otras;
- si un regimen `imbalance_aware` mejora incluso sin sinteticos;
- si las mejoras se sostienen tambien frente a una baseline `XGBoost`;
- si el mejor comportamiento downstream viene de dataset construction, de training behavior o de una combinacion de ambos.

## Que deja fuera conscientemente

- apertura de nuevas familias de modelo;
- tuning exhaustivo de `XGBoost`;
- variantes multiples de masa sintetica;
- sub-grids propios por `synthetic_policy`;
- reapertura de finalistas descriptivos de `FlowPre` como bases completas;
- cambios de panel de seeds dentro de la misma campaña.

## Conclusión

La parrilla de `17400` runs se justifica porque:

- es amplia, pero sigue teniendo una gramatica interpretable;
- favorece comparaciones justas entre construccion de dataset y comportamiento de entrenamiento;
- evita que `XGBoost` infle la campaña sin una hipotesis fuerte;
- y deja una base estadistica suficientemente fuerte como para sostener conclusiones medias y comparaciones family-level con mucha mas seriedad que un panel corto.
