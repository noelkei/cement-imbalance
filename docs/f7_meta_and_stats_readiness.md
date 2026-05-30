# F7 Meta and Statistical Readiness

## Proposito y estatus

Status note:

- este documento debe leerse ya como antecedente de trabajo;
- las decisiones estructurales del bloque `14` han quedado congeladas después en:
  - [docs/f7_block_14_analysis_grammar_rationale.md](f7_block_14_analysis_grammar_rationale.md)
  - [docs/f7_statistical_analysis_plan_structured.md](f7_statistical_analysis_plan_structured.md)
  - [docs/f7_statistical_analysis_spec.yaml](f7_statistical_analysis_spec.yaml)
- por tanto, cuando este documento hable de decisiones estadísticas todavía abiertas, debe interpretarse como contexto histórico de preparación, no como estado vigente final del bloque `14`.

Este documento fija que falta y que conviene hacer a corto plazo para que la campaña `F7` tenga:

1. una capa meta de identidad, trazabilidad y comparabilidad suficientemente robusta;
2. una capa de resultados suficientemente preparada para agregacion por seeds, ANOVA y post hoc posteriores.

No es:

- el contrato final de campaña;
- la implementacion final de runners;
- la decision final sobre la metrica principal de analisis;
- la decision final sobre el test estadistico concreto.

Si es:

- un documento de trabajo orientado a futuro proximo;
- la hoja de ruta para cerrar la parte meta y estadistica antes de lanzar la campaña completa;
- la referencia que debemos seguir para no quedarnos a medio camino con una campaña grande pero poco trazable.

Documentos relacionados:

- [docs/f7_run_plan_17400.md](f7_run_plan_17400.md)
- [docs/f7_run_plan_17400_rationale.md](f7_run_plan_17400_rationale.md)
- [docs/f7_trial_traceability_rationale.md](f7_trial_traceability_rationale.md)
- [docs/f7_campaign_readiness_checklist.md](f7_campaign_readiness_checklist.md)

## Objetivo de este frente

La campaña `F7` no necesita solo:

- datasets bien definidos;
- configs bien congeladas;
- runners que ejecuten.

Tambien necesita poder responder, sin ambiguedad:

- que dataset exacto se corrio;
- con que regimen exacto se corrio;
- que replica por seed es;
- a que campaña pertenece;
- con que otras runs es estadisticamente comparable;
- y como deben agregarse sus resultados para seleccion y analisis posterior.

En una campaña pequeña se pueden reconstruir muchas cosas “a mano”. En `F7`, con `17400` runs previstas, eso deja de ser aceptable.

## Lectura del estado actual

La situacion hoy no es “no tenemos nada”, pero tampoco “ya esta listo”.

### Resumen corto

- la base meta actual es **util y relativamente buena**, pero incompleta para una campaña grande;
- la base de resultados canónicos es **util y bastante prometedora**, pero incompleta para analisis estadistico serio y repetible a gran escala.

La lectura correcta es:

- ya existen cimientos reales;
- faltan piezas clave para que esos cimientos sean suficientes.

---

## 1. Estado actual de la capa meta

## 1.1. Que ya existe

### A. `dataset_level_axes` ya existen y ya son reales

En la capa de datasets ya existe soporte para:

- `x_transform`
- `y_transform`
- `synthetic_policy`

Esto ya vive en:

- [data/dataset_contract.py](../data/dataset_contract.py)
- manifests de datasets materializados

Eso es importante porque significa que la identidad dataset-level no parte de cero.

### B. `comparison_contract` ya existe en `train_mlp`

En [training/train_mlp.py](../training/train_mlp.py) ya se construye y persiste un `comparison_contract` con:

- `contract_id`
- `comparison_group_id`
- `seed_set_id`
- `base_config_id`
- `objective_metric_id`
- `dataset_level_axes`
- `run_level_axes`

Esto ya es una base clara para comparabilidad entre runs.

### C. `run_context` canónico ya existe

En [evaluation/results.py](../evaluation/results.py), `build_run_context(...)` ya normaliza y persiste:

- `run_id`
- `variant_fingerprint`
- `model_family`
- `contract_id`
- `comparison_group_id`
- `seed_set_id`
- `base_config_id`
- `objective_metric_id`
- `dataset_name`
- `dataset_manifest_path`
- `dataset_level_axes`
- `run_level_axes`
- `split_id`
- `split_manifest_path`
- `seed`
- `config_path`
- `config_sha256`
- `test_enabled`

Esto ya es bastante mas de lo que tendriamos en un repo desordenado tipico.

### D. `run_manifest.json` canónico ya existe

`save_canonical_run_artifacts(...)` ya escribe:

- `run_manifest.json`
- `metrics_long.csv`

Eso significa que la idea de run canonica ya esta parcialmente materializada.

### E. Hay precedentes reales de `evaluation_context`

En distintos scripts y familias del repo ya aparece la idea de:

- `campaign_id`
- `seed_set_id`
- `objective_metric_id`
- `dataset_level_axes`
- `run_level_axes`

Eso es útil porque confirma que esta capa no es un desideratum abstracto: ya existe un patrón a consolidar.

## 1.2. Que falta para que la capa meta sea suficientemente robusta

### A. `dataset_candidate_id`

Este hueco ya quedó resuelto en los bloques `7` y `12`.

Hoy tenemos:

- `dataset_level_axes`
- `dataset_name`
- `dataset_manifest_path`
- `dataset_candidate_id` formal y estable pensado para `F7`

#### Por que hace falta

- la campaña necesita hablar de datasets candidatos como unidades experimentales explícitas;
- no basta con reconstruirlos indirectamente desde axes o manifests;
- ayuda a agrupar, resumir y auditar sin depender de parsing ad hoc.

### B. `run_spec_id`

Este hueco ya quedó resuelto en los bloques `7` y `12`.

Hoy tenemos:

- `base_config_id`
- `run_level_axes`
- `run_spec_id` canónico, explícito y materializado en el `run_spec inventory`

#### Por que hace falta

- en `F7` queremos distinguir con limpieza:
  - mismo dataset
  - mismo run spec
  - distinta seed
- ese patrón debe ser trivial de reconstruir, no inferido artesanalmente.

### C. `trial_id`

Este hueco ya quedó resuelto en los bloques `7` y `12`.

Hoy ya existe una noción cerrada de:

- `trial_id = dataset identity + run config identity + replication identity`

#### Por que hace falta

- `run_id` nombra una instancia material, pero no necesariamente expresa bien la semántica de campaña;
- `trial_id` ayuda a tratar la run como unidad experimental explícita.

### D. `campaign_id` formal para `F7`

Este hueco ya quedó resuelto en los bloques `7` y `12`.

`campaign_id = f7_campaign_v1` ya forma parte de la gramática canónica de `F7` y de la `campaign spec`.

#### Por que hace falta

- la campaña de `17400` runs necesita pertenencia clara a una campaña;
- esa pertenencia debe ser visible tanto en manifests como en tablas de resultados.

### E. `statistical_family_id`

Hoy esta reconocido en los docs como faltante.

#### Por que hace falta

- ANOVA y post hoc posteriores necesitan saber que runs/familias pertenecen al mismo universo estadístico comparable;
- si eso no queda explícito, la comparabilidad acaba dependiendo de convenciones implícitas.

### F. `seed_panel_version`, `replication_index` y semántica de replica

Este hueco quedó resuelto en lo mínimo necesario para `F7`.

Hoy tenemos:

- `seed`
- `seed_set_id`
- `replication_index`
- `seed_panel_version`

La única semántica que no está todavía cerrada como pieza canónica de campaña es `seed_role` como campo explícito de resultados/runners, aunque su significado ya queda documentado en el panel de seeds.

#### Por que hace falta

- con `30` seeds, la replica deja de ser un detalle informal;
- conviene que el panel pueda evolucionar sin romper trazabilidad;
- conviene distinguir mejor una réplica dentro del panel.

### G. `campaign spec` formal

Este hueco ya quedó resuelto en el bloque `12`.

Hoy existe una especificación operativa machine-readable de campaña `F7` en:

- [config/f7_campaign_spec_v1.yaml](../config/f7_campaign_spec_v1.yaml)

Y además ya existen sus inventarios derivados materializados:

- `outputs/reports/f7_campaign_spec/f7_campaign_dataset_candidates_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_run_specs_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_trials_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_expansion_manifest_v1.json`

#### Por que hace falta

- sin eso, la identidad meta de la campaña se queda repartida entre varios docs;
- el runner tendría que reconstruir demasiada semántica desde texto.

---

## 2. Estado actual de la capa de resultados

## 2.1. Que ya existe

### A. `metrics_long.csv` canónico

En [evaluation/results.py](../evaluation/results.py) ya existe una capa canónica que aplana resultados a formato largo.

Las filas recogen, entre otras cosas:

- identidad de run
- identidad de dataset
- identidad de split
- `metric_group`
- `metric_name`
- `metric_scope`
- `component`
- `class_id`
- `value_space`
- `metric_value`

Esto es una base muy buena.

### B. Soporta varias granularidades útiles

La capa ya baja métricas a niveles como:

- `overall`
- `macro`
- `worst_class`
- `per_class`

según la familia de modelo y el tipo de resultado.

Esto es valioso porque evita quedarnos solo con un resumen demasiado grosero.

### C. `split_role` ya esta modelado

La existencia de:

- `train_diagnostic`
- `val_selection`
- `test_holdout`

ayuda mucho a evitar mezclar superficies con usos metodológicos distintos.

### D. Ya existe agregacion descriptiva básica por seeds

En [evaluation/aggregation.py](../evaluation/aggregation.py) ya existen funciones como:

- `filter_metrics_table(...)`
- `aggregate_metrics_by_seed(...)`
- `compare_family_variants(...)`

Esto significa que la capa de resultados no es solo almacenamiento, sino que ya tiene algo de analítica reproducible encima.

## 2.2. Que falta para que la capa de resultados este lista para ANOVA/post hoc serio

### A. Integrar `XGBoost` en la misma capa canónica de resultados

Hoy la canalización canónica de [evaluation/results.py](../evaluation/results.py) soporta claramente:

- `mlp`
- `flowpre`
- `flowgen`

pero no aparece una rama equivalente cerrada para `xgboost`.

#### Por que hace falta

- si `XGBoost` no entra en la misma gramática de `run_manifest + metrics_long`, la comparación con `MLP` queda menos homogénea;
- la capa canónica de resultados debe soportar ambas familias que entran en `F7`.

### B. Añadir ids canonicos a la tabla de resultados

Hoy `metrics_long` puede reconstruir bastante usando:

- `variant_fingerprint`
- `dataset_level_axes`
- `run_level_axes`

pero para una campaña grande conviene que las filas traigan explícitamente:

- `dataset_candidate_id`
- `run_spec_id`
- `trial_id`
- `campaign_id`
- `statistical_family_id`

#### Por que hace falta

- simplifica agrupacion y filtrado;
- hace el analisis mas robusto;
- reduce dependencia de inferencias posteriores.

### C. Congelar la unidad observacional y la unidad agregada

Hoy la infraestructura permite agregar, pero no esta completamente cerrado:

- que fila observacional exacta se usa para ANOVA;
- que subset de métricas gobierna la comparación principal;
- que subset es solo descriptivo.

#### Por que hace falta

- una campaña grande necesita que estas decisiones queden escritas antes del analisis, aunque el test estadistico exacto aún no se fije.

### D. Congelar la métrica principal de selección y contraste

La capa actual soporta muchas métricas, lo cual es una ventaja.  
Pero justamente por eso hace falta fijar:

- qué `metric_name`
- qué `metric_scope`
- qué `value_space`
- qué `split_role`

gobiernan la selección principal y la comparación principal.

#### Importante

Este documento no decide hoy cuál debe ser esa métrica. Solo deja claro que esa decisión falta y es bloqueante para el cierre estadístico serio.

### E. Construir una capa estadística general reusable

Hoy no existe todavía una capa general de:

- ANOVA
- post hoc
- tests alternativos cuando ANOVA no sea apropiado

reutilizable para `F7`.

Sí hay precedentes puntuales e históricos, pero no una capa general de campaña.

#### Por que hace falta

- la campaña no debería depender de scripts históricos ad hoc para su análisis estadístico final;
- conviene tener una superficie reusable y coherente con la gramática de `F7`.

### F. Definir grupos comparables y no comparables

Aunque el rationale de trazabilidad ya lo discute, falta que la capa de resultados lo pueda usar de forma explícita para:

- filtrar familias comparables;
- evitar mezclar líneas incompatibles;
- dejar claro el alcance de cada ANOVA/post hoc.

#### Por que hace falta

- la capa meta sola no basta si luego los resultados no tienen una forma cómoda de expresar ese alcance.

---

## 3. Que hacer a corto plazo

Esta sección no toma decisiones que todavía no han sido fijadas por el proyecto. Lo que hace es secuenciar el trabajo necesario.

## 3.1. Cerrar primero la identidad meta mínima

### Objetivo

Que cada run de `F7` pueda responder de forma explícita:

- qué dataset candidato es;
- qué run spec es;
- qué réplica es;
- a qué campaña pertenece.

### Trabajo pendiente

1. definir `dataset_candidate_id`
2. definir `run_spec_id`
3. definir `trial_id`
4. definir `campaign_id`
5. definir `statistical_family_id`
6. definir `seed_panel_version`
7. definir `replication_index`

### Resultado esperado

Una campaña donde la identidad experimental no dependa de recomponer semántica desde `run_id` o desde blobs JSON.

## 3.2. Formalizar `campaign spec`

### Objetivo

Tener una pieza machine-readable que unifique el plan de campaña.

### Trabajo pendiente

Crear una especificación de campaña que recoja al menos:

- datasets `MLP`
- datasets `XGBoost`
- definicion de `run_spec`
- panel de seeds
- ids de campaña
- grupos comparables

### Resultado esperado

El runner deja de depender de interpretación manual de varios docs.

## 3.3. Extender la capa canónica de resultados para `XGBoost`

### Objetivo

Que `XGBoost` pase por la misma lógica canónica de resultados que `MLP`.

### Trabajo pendiente

- construir `run_context` coherente también para `XGBoost`;
- producir `run_manifest.json` y `metrics_long.csv` con la misma gramática;
- asegurar que las columnas clave de comparabilidad estén presentes.

### Resultado esperado

Comparaciones `MLP` vs `XGBoost` sin dos pipelines de reporting distintos.

## 3.4. Añadir ids canonicos a `metrics_long`

### Objetivo

Que la tabla de resultados ya venga preparada para agregación y análisis.

### Trabajo pendiente

Añadir a la capa canónica columnas como:

- `dataset_candidate_id`
- `run_spec_id`
- `trial_id`
- `campaign_id`
- `statistical_family_id`

### Resultado esperado

Menos trabajo artesanal posterior y menor riesgo de errores en agregación.

## 3.5. Fijar la gramática del análisis principal

### Objetivo

Cerrar cómo leeremos la campaña, aunque todavía no se decida el test estadístico final.

### Trabajo pendiente

Decidir y documentar:

- métrica principal de selección/comparación;
- `split_role` principal;
- `value_space` principal;
- `metric_scope` principal;
- unidad observacional;
- unidad agregada por seeds.

### Resultado esperado

La fase de análisis deja de depender de decisiones improvisadas al final de la campaña.

## 3.6. Diseñar la capa estadística reutilizable

### Objetivo

No depender de scripts históricos puntuales para el cierre estadístico de `F7`.

### Trabajo pendiente

Preparar una capa reusable que, cuando llegue el momento, pueda soportar:

- ANOVA si las condiciones del diseño y de los residuos lo permiten;
- alternativas no paramétricas o de medidas repetidas si ANOVA no es adecuado;
- post hoc consistentes con el diseño finalmente fijado.

### Importante

Este documento no decide ahora qué test concreto hay que usar. Solo deja fijado que esta capa debe existir y debe apoyarse en una gramática meta cerrada.

## 3.7. Congelar reglas de comparabilidad

### Objetivo

Dejar claro qué se puede mezclar y qué no, no solo en docs sino en la propia capa de campaña/resultados.

### Trabajo pendiente

Formalizar reglas para no mezclar, sin declararlo, por ejemplo:

- familias con distinto `seed_set_id`
- líneas `official` y `train_only`
- `MLP` y `XGBoost`
- datasets con distintas `synthetic_policy`
- campañas distintas

### Resultado esperado

El análisis posterior se vuelve más seguro y más defendible.

---

## 4. Criterios de “suficientemente completo”

## 4.1. La capa meta estará suficientemente completa cuando

- exista `campaign spec` de `F7`;
- cada run tenga `dataset_candidate_id`, `run_spec_id`, `trial_id`, `campaign_id` y `seed_set_id`;
- exista semántica explícita de réplica;
- los manifests puedan reconstruir sin ambigüedad la identidad completa del trial.

Nota de estado actual:

- la `campaign spec` y el `trial inventory` ya existen;
- `dataset_candidate_id`, `run_spec_id`, `trial_id`, `campaign_id`, `comparison_group_id` y `replication_index` ya tienen gramática y materialización explícita;
- el bloque `13` ya consume esa tabla como fuente operativa única mediante:
  - [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
  - [evaluation/f7_campaign_runner.py](../evaluation/f7_campaign_runner.py)

Matiz importante:

- el runner añade una validación operativa más estricta que la del bloque `12`;
- en `MLP`, si un dataset no puede reconstruir `raw_real` por falta de `target scaler`, el `preflight` del runner ya lo marca como no ejecutable aunque la fila del `trial inventory` sea estructuralmente válida.

## 4.2. La capa de resultados estará suficientemente completa cuando

- `MLP` y `XGBoost` generen ambos `metrics_long.csv` compatibles;
- la tabla de resultados incluya ids canónicos explícitos;
- esté fijada la gramática de la comparación principal;
- la capa de agregación por seeds ya pueda trabajar con esa gramática;
- exista una superficie estadística reusable diseñada para la campaña.

---

## 5. Que no decide este documento

Para evitar cerrar cosas que todavía no se han decidido, este documento no fija:

- la métrica principal exacta de selección o ANOVA;
- el test estadístico final concreto;
- si el análisis principal será paramétrico o no paramétrico;
- la gramática del análisis estadístico principal del bloque `14`.

Lo que sí fija es que todas esas decisiones faltan y deben resolverse explícitamente antes de considerar cerrada la robustez meta y estadística de `F7`.

---

## 6. Uso recomendado de este documento

Este documento debe usarse como referencia para:

- planear la siguiente tanda de implementación;
- revisar readiness antes de lanzar pilotos y antes de lanzar campaña completa;
- comprobar si la robustez meta/estadística ya está al nivel que exige una campaña de `17400` runs.
