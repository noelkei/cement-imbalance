# F7 Campaign Readiness Checklist

## Proposito y estatus

Este documento recoge la checklist de cosas que faltan antes de poder ejecutar la campaña `F7` tal como hoy esta planteada.

No es:

- el plan de runs en si;
- el rationale metodologico del plan;
- el contrato final de ejecucion automatizada.

Si es:

- una lista operativa de bloqueantes y tareas previas;
- la referencia que iremos usando para comprobar que la campaña esta realmente lista;
- un puente entre el plan de `17400` runs y la implementacion concreta de runners, configs y manifests.

Documentos relacionados:

- [docs/f7_run_plan_17400.md](f7_run_plan_17400.md)
- [docs/f7_run_plan_17400_rationale.md](f7_run_plan_17400_rationale.md)
- [docs/f7_experimental_space_rationale.md](f7_experimental_space_rationale.md)
- [docs/f7_trial_traceability_rationale.md](f7_trial_traceability_rationale.md)
- [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md)
- [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md)
- [docs/f7_campaign_preparation_workplan.md](f7_campaign_preparation_workplan.md)

## Resumen de la campaña objetivo

Estado objetivo actualmente fijado:

- `MLP`:
  - `96` datasets
  - `2` familias `batch/cycling`
  - `3` `loss_policy`
  - `30` seeds
  - total `17280` runs
- `XGBoost`:
  - `4` datasets
  - `1` config fija
  - `30` seeds
  - total `120` runs
- total global:
  - `17400` runs

Guardrails ya decididos:

- cap sintetico maximo del `50%` del numero real de cada clase minoritaria;
- `XGBoost` no barre escalados ni variantes de training loss;
- `MLP` si compara:
  - `plain = baseline + no cycling`
  - `imbalance_aware = balanced + cycling`
- `MLP` usa exactamente:
  - `overall + rmse`
  - `per_class_equal + rmse`
  - `per_class_equal + rrmse`

Alcance reconfirmado el `2026-05-16` en el bloque `0` del workplan:

- la preparacion actual sigue orientada a la campaña completa de `17400` runs;
- `XGBoost` se mantiene como baseline acotada;
- las `30` seeds siguen formando parte del diseño base;
- el cap sintetico del `50%` sigue aplicando por igual a las tres policies sinteticas.

## Criterio de lectura de esta checklist

La checklist se organiza en tres niveles:

1. `must_have_before_first_run`
2. `must_have_before_full_campaign`
3. `nice_to_have_but_not_blocking`
4. `final_preflight_before_launch`

Regla practica:

- no lanzar ni siquiera un piloto serio si quedan huecos criticos en `must_have_before_first_run`;
- no lanzar la campaña completa mientras queden huecos en `must_have_before_full_campaign`.
- usar [docs/f7_campaign_preparation_workplan.md](f7_campaign_preparation_workplan.md) como orden principal de trabajo y esta checklist como criterio de cierre/readiness.
- interpretar ese orden segun la seccion `Interpretacion operativa del orden` del workplan:
  - algunos bloques son prerequisitos estrictos;
  - otros se trabajan en paralelo;
  - y algunos deben cerrarse por fases o como paquete para evitar retrodependencias practicas.

Convención de rationale para memoria:

- cada bloque de la precampaña debe acabar con un documento `docs/f7_block_XX_<tema>_rationale.md`;
- si el rationale completo vive de forma natural en otro artefacto de cierre, ese archivo `rationale` puede actuar como índice corto y punto de entrada estable;
- la intención es poder localizar rápidamente todas las decisiones metodológicas de la precampaña con una búsqueda por `f7_block_` y `rationale`.

---

## 1. `must_have_before_first_run`

Estas piezas deben quedar resueltas antes de correr siquiera un bloque piloto representativo de la campaña.

### 1.1. `train_mlp` debe usar el device canónico de campaña

Estado:

- cerrado el `2026-05-16`

Por que bloquea:

- la campaña es demasiado grande para depender de selección implícita de device;
- había que cerrar tanto el soporte canónico como la decisión operativa real para esta máquina.

Hecho cuando:

- `train_mlp.py` resuelve device mediante la utilidad canonica;
- `mps` puede pedirse explicitamente;
- el device efectivo queda persistido en los artefactos de run;
- existe benchmark real de `cpu` frente a `mps` en esta máquina;
- queda fijado que para `F7` las runs de `MLP` deben forzar `cpu`;
- la política no depende de `auto`.

### 1.2. Congelar baseline final de `MLP`

Estado:

- cerrado el `2026-05-16`

Por que bloquea:

- no podemos lanzar una campaña masiva sin saber que arquitectura / hiperparametros estamos fijando como baseline unica.

Hecho cuando:

- quedan congelados y trazados:
  - `hidden_dim`
  - `num_layers`
  - `embedding_dim`
  - `batch_size`
  - `learning_rate`
  - `num_epochs`
  - `early_stopping_patience`
  - `lr_scheduler`
  - `optimizer`
- existe un `config_id` canonico de campaña para esa base.
- la revalidacion de apoyo activa definida en:
  - [docs/f7_mlp_baseline_revalidation_v3.md](f7_mlp_baseline_revalidation_v3.md)
  - [config/f7_mlp_baseline_revalidation_v3.yaml](../config/f7_mlp_baseline_revalidation_v3.yaml)
  queda ejecutada y resumida.
- la decision final queda aterrizada en:
  - [docs/f7_mlp_baseline_final_v1.md](f7_mlp_baseline_final_v1.md)
  - [config/f7_mlp_base_v1.yaml](../config/f7_mlp_base_v1.yaml)

### 1.3. Congelar baseline final de `XGBoost`

Estado:

- cerrado el `2026-05-16`

Por que bloquea:

- `XGBoost` debe entrar como baseline fuerte y acotada, no como script semi-historico.

Hecho cuando:

- queda fijada una sola config de entrenamiento;
- queda congelado:
  - `objective`
  - `eval_metric`
  - hiperparametros principales del booster;
- existe un `model_config_id` canonico.
- existe una revalidacion de apoyo ejecutable definida en:
  - [docs/f7_xgb_baseline_revalidation_v1.md](f7_xgb_baseline_revalidation_v1.md)
  - [config/f7_xgb_baseline_revalidation_v1.yaml](../config/f7_xgb_baseline_revalidation_v1.yaml)
- existe una micro-revalidacion final ejecutada y documentada en:
  - [docs/f7_xgb_baseline_revalidation_v2.md](f7_xgb_baseline_revalidation_v2.md)
  - [config/f7_xgb_baseline_revalidation_v2.yaml](../config/f7_xgb_baseline_revalidation_v2.yaml)
- existe un baseline final congelado en:
  - [docs/f7_xgb_baseline_final_v1.md](f7_xgb_baseline_final_v1.md)
  - [config/f7_xgb_base_v1.yaml](../config/f7_xgb_base_v1.yaml)

Nota:

- el panel final de `30` seeds y su semantica comun entre `MLP` y `XGBoost` se congela mas adelante en el bloque especifico de seeds;
- este bloque `1.3` solo exige que el baseline de `XGBoost` quede definido de forma canonica y listo para entrar en esa futura gramática comun.

### 1.4. Fijar la representacion unica de dataset para `XGBoost`

Estado:

- cerrado el `2026-05-16`

Por que bloquea:

- ya se decidio que `XGBoost` no barre escalados;
- pero aun falta declarar cual es la base unica fija que usara.

Hecho cuando:

- se elige explicitamente la variante base;
- queda documentada como `feature_policy` o dataset base fija de `XGBoost`;
- esa decision se refleja en el contrato de campaña.
- la decision fija:
  - dataset oficial raw/no-scale
  - `feature_policy = raw_numeric_plus_type_onehot`

Nota:

- el bloque `2` deja ademas fijado que la futura capa SHAP de `XGBoost` debe persistir artefactos lo bastante completos como para permitir analisis agregados posteriores por feature, no solo un smoke test local.

### 1.5. Alinear la masa sintetica al cap del `50%`

Estado:

- cerrado el `2026-05-17`

Por que bloquea:

- sin esto, comparariamos policies que difieren tanto en algoritmo como en cantidad de sinteticos;
- eso ensucia la interpretacion de resultados.

Hecho cuando:

- `flowgen_official`
- `flowgen_train_only`
- `kmeans_smote`

quedan materializados o re-materializados bajo la misma regla:

- `n_synth <= 0.5 * n_real` por clase minoritaria.

Nota:

- la política final de `F7` queda refinada y ya no es solo ese guardrail resumido;
- además exige:
  - `n_real(c) + n_synth(c) <= max_real_train`
  - `n_synth = 0` en clases mayoritarias de referencia
  - y validación central obligatoria antes de aceptar un dataset como `campaign-ready`.
- artefactos canónicos:
  - [config/f7_synthetic_cap_policy_v1.yaml](../config/f7_synthetic_cap_policy_v1.yaml)
  - [data/f7_synthetic_cap_policy.py](../data/f7_synthetic_cap_policy.py)

### 1.6. Definir y congelar el panel de `30` seeds

Estado:

- cerrado el `2026-05-18`

Por que bloquea:

- el plan actual ya no usa el `closure_5x_v1`;
- sin el panel real no existe campaña comparable.

Hecho cuando:

- existe `seed_set_id` nuevo;
- las `30` seeds concretas estan fijadas;
- la semantica del panel queda documentada en config o contract.
- artefactos canónicos:
  - [config/f7_seed_panel_v1.yaml](../config/f7_seed_panel_v1.yaml)
  - [config/f7_seed_panel_v1.csv](../config/f7_seed_panel_v1.csv)
  - [docs/f7_block_06_seed_panel_rationale.md](f7_block_06_seed_panel_rationale.md)

### 1.7. Verificar que `MLP` soporta bien los dos regimenes `batch/cycling`

Estado:

- pendiente

Por que bloquea:

- el plan exige dos familias:
  - `plain`
  - `imbalance_aware`
- y ahora ambas aplican tambien a `synthetic_policy = none`.

Hecho cuando:

- `plain = baseline + no cycling` funciona;
- `imbalance_aware = balanced + cycling` funciona;
- no hay fallos de sampler, batching o divisibilidad por clases.

### 1.8. Confirmar que `allow_synth` / `is_synth` se comportan bien

Estado:

- pendiente

Por que bloquea:

- la campaña mezcla datasets con y sin sinteticos;
- hace falta asegurar que:
  - `is_synth` no entra como feature;
  - los sinteticos se filtran correctamente si corresponde;
  - train/val/test siguen limpios.

Hecho cuando:

- hay smoke tests o verificaciones claras de ese comportamiento en datasets:
  - `none`
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`

### 1.9. Verificar comparabilidad de métricas en `raw space`

Estado:

- cerrado el `2026-05-18`

Por que bloquea:

- distintas runs usarán distintos `y_transform`;
- si las métricas principales no se calculan en espacio `raw`, la comparabilidad downstream queda comprometida.

Hecho cuando:

- se verifica explícitamente qué funciones calculan las métricas de `MLP`;
- la inversión al espacio `raw` queda garantizada para la comparación principal;
- el `value_space` principal queda claro y trazado;
- el contrato distingue correctamente:
  - `selection_run = train + val`
  - `holdout_run = train + val + test`
- la política descrita en [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md) queda satisfecha.
- artefactos canónicos:
  - [config/f7_raw_metric_contract_v1.yaml](../config/f7_raw_metric_contract_v1.yaml)
  - [docs/f7_block_08_raw_metric_comparability_rationale.md](f7_block_08_raw_metric_comparability_rationale.md)

### 1.10. Medir tiempos reales con un piloto corto

Estado:

- pendiente

Por que bloquea:

- `17400` runs ya no permite trabajar con estimaciones teoricas solamente.

Hecho cuando:

- se corren al menos:
  - `10-20` runs representativas de `MLP`
  - `2-4` runs de `XGBoost`
- se registra:
  - media
  - mediana
  - `p90`
  - tiempo wall-clock total
- se recalcula el coste esperado de la campaña.

---

## 2. `must_have_before_full_campaign`

Estas piezas pueden no bloquear un piloto pequeño, pero si bloquean el lanzamiento serio de toda la campaña.

### 2.1. Formalizar `campaign spec` de `F7`

Estado:

- hecho

Por que bloquea:

- con `17400` runs no basta un plan narrativo;
- hace falta una especificacion operativa de campaña.

Hecho cuando:

- existe un config o contract de campaña con:
  - datasets `MLP`
  - datasets `XGBoost`
  - regímenes `MLP`
  - config `XGBoost`
  - `seed_set_id`
  - `comparison_group_id`
  - `campaign_id`
- esta `campaign spec` ya es coherente con la hoja de ruta meta/estadistica de [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md)
- y con el orden y preguntas de decisión de [docs/f7_campaign_preparation_workplan.md](f7_campaign_preparation_workplan.md)
- y la expansión materializada deja:
  - `100` dataset candidates
  - `7` run specs
  - `17400` trials

### 2.2. Formalizar ids canonicos de trial/campaign

Estado:

- hecho

Por que bloquea:

- luego no podremos agregar ni auditar bien si cada run no cae en una gramatica estable.

Hecho cuando:

- quedan claros y usables al menos:
  - `dataset_candidate_id`
  - `run_spec_id`
  - `seed_set_id`
  - `trial_id`
  - `campaign_id`
  - `comparison_group_id`
- y esa gramatica queda integrada tanto en manifests como en la capa canónica de resultados, siguiendo [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md)

Nota de estado:

- la gramática base ya queda congelada en:
  - [config/f7_meta_grammar_v1.yaml](../config/f7_meta_grammar_v1.yaml)
  - [docs/f7_block_07_meta_grammar_rationale.md](f7_block_07_meta_grammar_rationale.md)
- la integración runner-side ya existe en:
  - [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
  - [evaluation/results.py](../evaluation/results.py)
  - [evaluation/f7_campaign_state.py](../evaluation/f7_campaign_state.py)
- sigue pendiente la parte estrictamente estadística/analítica del bloque `14`, no la semántica básica de ids.

### 2.3. Cerrar la capa meta y estadistica minima de campaña

Estado:

- pendiente

Por que bloquea:

- una campaña de `17400` runs necesita algo mas que manifests sueltos y agregacion descriptiva basica;
- hace falta una capa meta y una capa de resultados que soporten comparabilidad y analisis posterior de forma robusta.

Hecho cuando:

- los huecos descritos en [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md) quedan resueltos al nivel minimo necesario para `F7`;
- `MLP` y `XGBoost` pueden entrar en la misma gramática canónica de resultados;
- la unidad de analisis principal queda documentada aunque el test estadistico final aun no se haya fijado.

### 2.4. Cerrar politica de persistencia e interpretabilidad

Estado:

- cerrado el `2026-05-18`

Por que bloquea:

- una campaña de `17400` runs no puede lanzarse sin una política clara de qué artefactos guardar y qué interpretabilidad mínima exigir;
- no queremos depender de guardar todos los pesos para poder defender resultados e insights.

Hecho cuando:

- queda cerrada la separación entre artefactos obligatorios, artefactos pesados y piezas recomputables;
- existe una capa mínima de interpretabilidad implementada para `MLP`;
- existe una capa mínima de interpretabilidad implementada para `XGBoost`;
- los requisitos descritos en [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md) quedan cubiertos.
- artefactos canónicos:
  - [config/f7_artifact_persistence_contract_v1.yaml](../config/f7_artifact_persistence_contract_v1.yaml)
  - [docs/f7_block_09_artifact_persistence_rationale.md](f7_block_09_artifact_persistence_rationale.md)
  - [config/f7_mlp_interpretability_contract_v1.yaml](../config/f7_mlp_interpretability_contract_v1.yaml)
  - [docs/f7_block_10_mlp_interpretability_rationale.md](f7_block_10_mlp_interpretability_rationale.md)
  - [config/f7_xgb_interpretability_contract_v1.yaml](../config/f7_xgb_interpretability_contract_v1.yaml)
  - [docs/f7_block_11_xgb_interpretability_rationale.md](f7_block_11_xgb_interpretability_rationale.md)

Nota de estado:

- el bloque `9` deja cerrada la política y la superficie canónica de persistencia por run;
- la interpretabilidad específica de `MLP` queda ya cerrada en el bloque `10`;
- el ajuste `10B` deja la superficie primaria final de `MLP` en `chem_*`, `phase_*` y features directas, sin nombres `ilr_*` en los artefactos canónicos finales;
- la interpretabilidad `MLP` persiste también estadísticas de dispersión y masa relativa (`share_abs_importance`) para análisis estadístico posterior;
- el coste observado en validación es bajo para `scaled` y alto para `flowpre_candidate_*`, por lo que la campaña completa debe presupuestarse con ese runtime;
- la interpretabilidad específica de `XGBoost` queda ya cerrada en el bloque `11`;
- `XGBoost` dispone ahora de una capa nativa `SHAP` y de una capa puente de perturbación ligera alineada con `MLP`;
- la comparación cruzada entre `MLP` y `XGBoost` puede apoyarse en la capa puente común sin reabrir el contrato de persistencia.

### 2.5. Alinear naming contractual de `synthetic_policy`

Estado:

- pendiente

Por que bloquea:

- hoy el contrato general aun mezcla la familia `flowgen` con la distincion semantica que queremos hacer entre:
  - `flowgen_official`
  - `flowgen_train_only`

Hecho cuando:

- queda resuelto si la distincion vive en:
  - `synthetic_policy`
  - o en `synthetic_policy_id`
- y esa decision es consistente en manifests, runners y reporting.

### 2.6. Materializar o verificar todos los datasets necesarios

Estado:

- pendiente

Por que bloquea:

- la campaña depende de que existan de verdad:
  - `96` datasets `MLP`
  - `4` datasets `XGBoost`

Hecho cuando:

- los datasets existen o pueden regenerarse de forma determinista;
- todos tienen manifests coherentes;
- todos registran bien:
  - `dataset_level_axes`
  - `synthetic_policy_id`
  - `counts_by_class`
  - procedencia upstream

### 2.7. Crear runner de campaña para `MLP`

Estado:

- cerrado el `2026-05-20`

Por que bloquea:

- `17280` runs `MLP` no son manejables con lanzamientos manuales o scripts ad hoc.

Hecho cuando:

- existe un runner que:
  - itera datasets
  - itera `run_spec`
  - itera seeds
  - soporta resume/restart
  - evita duplicados
  - escribe manifests y logs consistentes

Estado actual:

- ya existe un runner canónico de campaña común a `MLP` y `XGBoost`:
  - [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
- `MLP` ya tiene paridad explícita de `run_id` y `run_dir` con la capa de campaña;
- la ejecución queda namespaced por `campaign_id`;
- el estado se persiste por trial y por intento.

### 2.8. Crear runner de campaña para `XGBoost`

Estado:

- cerrado el `2026-05-20`

Por que bloquea:

- el script actual es util como precedente, pero no como runner canonico de campaña `F7`.

Hecho cuando:

- existe un runner alineado con:
  - los `4` datasets `XGBoost`
  - las `30` seeds
  - la misma gramatica de trial/campaign

Estado actual:

- el mismo runner canónico ya ejecuta `XGBoost` sobre el `trial inventory` congelado;
- `XGBoost` conserva su ruta de entrenamiento actual, pero ahora bajo control de campaña con:
  - ledger
  - registry
  - resume/rerun
  - lineage de campaña

### 2.9. Congelar metrica de seleccion y agregacion por seeds

Estado:

- pendiente

Por que bloquea:

- sin esto no sabremos como seleccionar shortlist o finalistas de forma coherente.

Hecho cuando:

- queda fijada la metrica principal de seleccion;
- queda fijado como agregamos sobre seeds;
- queda claro que metricas secundarias se reportan pero no gobiernan ranking.
- la decision es coherente con la capa de resultados y de analisis descrita en [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md);
- y con la política de comparabilidad en `raw space` descrita en [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md)

### 2.10. Definir estrategia de ejecucion

Estado:

- pendiente

Por que bloquea:

- `17400` runs requieren una politica clara de ejecucion y reanudacion.

Hecho cuando:

- se decide:
  - serie vs paralelo
  - granularidad por bloque
  - orden de lanzamiento
  - politica de resume
  - politica de retries

### 2.11. Verificar footprint de disco y outputs

Estado:

- pendiente

Por que bloquea:

- tantas runs pueden generar demasiado peso en manifests, logs y reportes si no se controla bien.

Hecho cuando:

- existe una estimacion de footprint;
- se decide que artefactos se guardan por run;
- se revisa que no guardamos demasiado por defecto.

---

## 3. `nice_to_have_but_not_blocking`

Estas piezas no deberian frenar un piloto ni necesariamente la campaña completa, pero ayudarian mucho a orden, reporting o defensa posterior.

### 3.1. Reportes agregados predefinidos por family

Ejemplos:

- resumen por `synthetic_policy`
- resumen por `x_base`
- resumen por `y_transform`
- resumen por `model_family`

### 3.2. Plantilla de analisis estadistico final

No hace falta implementarlo del todo ya, pero si tener claro:

- que ANOVA/post hoc queremos;
- que tablas resumen esperamos;
- como se agrupan familias comparables.
- ver tambien [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md)

### 3.3. Tabla contractual exacta de combinaciones

Version machine-readable del plan:

- tabla de los `96` datasets `MLP`
- tabla de los `4` datasets `XGBoost`
- tabla de `run_spec`
- cruce final de `trial_id`

### 3.4. Politica de checkpoint / pruning de artefactos

Para no inflar `outputs/` sin necesidad:

- decidir que guardar siempre;
- decidir que puede regenerarse;
- decidir que se puede resumir en vez de persistirse completo.
- ver tambien [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md)

### 3.5. Dashboard o reporte ligero de progreso de campaña

Util para:

- ver cuantos runs completadas van;
- detectar fallos por bloque;
- seguir cobertura por seeds y por familias.

### 3.6. Documento de trabajo ordenado por fases

Ya existe una pieza más completa para guiar la preparación de campaña:

- [docs/f7_campaign_preparation_workplan.md](f7_campaign_preparation_workplan.md)

Su papel no es sustituir esta checklist, sino ampliar:

- el orden de trabajo;
- las preguntas de decisión;
- y el detalle de cada bloque antes de implementarlo.

---

## 4. `final_preflight_before_launch`

Estas piezas son la última barrera antes de decir de verdad:

- "sí, ya podemos lanzar la campaña completa"

No sustituyen los bloques anteriores. Sirven para comprobar que todo lo ya decidido funciona junto y que el lanzamiento no depende de supuestos tácitos.

### 4.1. Dry run end-to-end real

Estado:

- cerrado el `2026-05-21`

Por que bloquea:

- una cosa es que las piezas funcionen por separado;
- otra distinta es que el circuito real de campaña funcione entero.

Hecho cuando:

- se corre al menos:
  - `1` caso `MLP`
  - `1` caso `XGBoost`
- ambos pasan por la misma infraestructura final de campaña;
- se resuelve dataset desde la `campaign spec`;
- se construye el `trial_id`;
- se entrena;
- se guardan manifests y resultados;
- se genera `metrics_long`;
- se genera la interpretabilidad mínima;
- y todo eso entra en la capa de agregación sin romperse.

Nota de cierre:

- la validación real suficiente queda cubierta por la cadena canónica pequeña `104 + 52 + 52` ya ejecutada con:
  - runner
  - lineage
  - reporting
  - interpretabilidad
  - y surfaces estadísticas endurecidas.

### 4.2. Enumeracion exacta de trials antes del lanzamiento

Estado:

- cerrado el `2026-05-21`

Por que bloquea:

- con `17400` runs no se puede lanzar sobre una combinatoria implícita;
- hace falta saber exactamente qué runs existen en la campaña.

Hecho cuando:

- existe una tabla machine-readable con todos los trials;
- el conteo exacto cuadra con el plan:
  - `17280` runs `MLP`
  - `120` runs `XGBoost`
  - `17400` total
- no hay duplicados;
- no hay huecos;
- no hay combinaciones fuera de contrato.

Nota de cierre:

- la readiness pass grande valida:
  - `17400` trials en `primary`
  - `17400` en cada una de las `3` extensiones
  - `69600` en total
  - y `120` seeds en la cadena completa.

### 4.3. Politica de fallo, rerun e idempotencia

Estado:

- cerrado el `2026-05-21`

Por que bloquea:

- en una campaña grande no basta con “si falla, ya veremos”;
- hace falta saber cuándo una run cuenta, cuándo no, y cómo se recupera.

Hecho cuando:

- está definido qué archivos mínimos debe dejar una run para considerarse válida;
- está definido cuándo una run incompleta debe:
  - retomarse
  - marcarse como fallida
  - rehacerse desde cero
- el runner evita relanzar runs ya válidas;
- la política de resume es consistente con la trazabilidad de campaña.

Estado actual:

- ya existe estado explícito por trial:
  - `pending`
  - `running`
  - `completed`
  - `failed`
  - `blocked`
- `completed` requiere `campaign_valid_f7=true`;
- fallos y bloqueos conservan `failure_reason_code` y detalle;
- las campañas de extensión por seeds se modelan como campañas nuevas con linaje explícito.

### 4.4. Freeze de entorno y software

Estado:

- cerrado el `2026-05-21`

Por que bloquea:

- una campaña larga no debe quedar mezclada entre varias versiones del stack sin control.

Hecho cuando:

- queda fijado el entorno de ejecución de campaña;
- quedan registradas versiones relevantes de:
  - Python
  - PyTorch
  - XGBoost
  - dependencias críticas
- la campaña registra el commit o versión de código base sobre la que corre.

Nota de cierre:

- el readiness report canónico ya registra:
  - Python
  - PyTorch
  - XGBoost
  - plataforma
  - `git_commit`

### 4.5. Freeze estricto de datasets y manifests

Estado:

- cerrado el `2026-05-18`

Por que bloquea:

- no se puede lanzar una campaña larga si los datasets pueden cambiar silenciosamente a mitad de ejecución.

Hecho cuando:

- el inventario exacto de datasets queda fijado antes del lanzamiento;
- cada dataset de campaña tiene manifest estable;
- cada dataset tiene fingerprint o identidad estable;
- no se rematerializan datasets silenciosamente bajo el mismo id durante la campaña.

Nota de estado actual:

- el inventario lógico exacto ya quedó fijado en:
  - [config/f7_dataset_inventory_v1.yaml](../config/f7_dataset_inventory_v1.yaml)
  - [config/f7_dataset_inventory_v1.csv](../config/f7_dataset_inventory_v1.csv)
- la materialización final `4B` quedó cerrada el `2026-05-18` en:
  - [outputs/reports/f7_dataset_materialization/f7_dataset_materialization_20260518T132351821306Z](../outputs/reports/f7_dataset_materialization/f7_dataset_materialization_20260518T132351821306Z)
- el batch final deja:
  - `96` datasets `MLP`
  - `4` datasets `XGBoost`
  - `2` pools compartidos `FlowGen`
  - `102` artefactos `ok`
  - `0` fallos
- el árbol canónico materializado queda en:
  - [data/sets/official/init_temporal_processed_v1](../data/sets/official/init_temporal_processed_v1)
- dentro de esa raíz:
  - `raw/`
  - `scaled/`
  - `synthetic_pools/`
  - `augmented_scaled/`
  - `xgboost/`
  - `meta/`
- el material histórico o de smoke queda separado en:
  - `legacy_pre_f7/`
- la referencia local exacta de qué ids y rutas usar en campaña queda copiada en:
  - [data/sets/official/init_temporal_processed_v1/meta/f7_canonical_materialized_inventory_v1.csv](../data/sets/official/init_temporal_processed_v1/meta/f7_canonical_materialized_inventory_v1.csv)

### 4.6. Outputs mínimos para run válida

Estado:

- cerrado el `2026-05-21`

Por que bloquea:

- el runner necesita un criterio objetivo para decidir si una run cuenta o no.

Hecho cuando:

- está definido el conjunto mínimo de outputs obligatorios por run válida;
- típicamente incluye:
  - `run_manifest.json`
  - config snapshot
  - `results.yaml`
- `metrics_long.csv`
- interpretabilidad mínima
- estado final consistente

Nota de cierre:

- este criterio queda ya capturado operativamente por:
  - `campaign_valid_f7`
  - `analysis_ready_comparable`
  - `analysis_ready_blockers`
  - y la persistencia canónica validada en el bloque `13`.

### 4.7. Verificacion explicita de bloqueo de `test`

Estado:

- cerrado el `2026-05-21`

Por que bloquea:

- el lanzamiento de campaña no debe iterar sobre `test` por accidente.

Hecho cuando:

- el runner trabaja por defecto en `val_selection`;
- `test` permanece bloqueado salvo opt-in explícito;
- la agregación principal y la selección principal no usan `test` durante la campaña.

Nota de cierre:

- para `F7`, `test` permanece habilitado como output de holdout;
- pero la selección y narrowing pre-freeze siguen gobernadas por `val`, en coherencia con el bloque `14`.

---

## Orden recomendado de trabajo

### Fase 1: dejar el motor listo

1. soporte canonico de `MPS` en `train_mlp`
2. congelar baseline `MLP`
3. congelar baseline `XGBoost`
4. fijar base unica de `XGBoost`
5. fijar panel de `30` seeds

### Fase 2: cerrar la identidad de datasets

6. alinear cap sintetico al `50%`
7. resolver naming contractual de `synthetic_policy`
8. materializar o verificar datasets necesarios
9. validar `allow_synth` / `is_synth`

### Fase 3: cerrar la capa de campaña

10. formalizar `campaign spec`
11. formalizar ids canonicos
12. cerrar la capa meta/estadistica minima
13. cerrar politica de persistencia e interpretabilidad
14. congelar metrica de seleccion y agregacion

### Fase 4: dejar la ejecucion lista

15. runner `MLP`
16. runner `XGBoost`
17. benchmark real de tiempos
18. estrategia de ejecucion / resume
19. chequeo de footprint

### Fase 5: preflight final de lanzamiento

20. dry run end-to-end real
21. enumeración exacta de trials
22. política de fallo / rerun / idempotencia
23. freeze de entorno y software
24. freeze estricto de datasets y manifests
25. definición de outputs mínimos por run válida
26. verificación explícita de bloqueo de `test`

---

## Criterio final de readiness

La campaña `F7` solo debe considerarse lista para lanzamiento completo cuando:

- no queda nada pendiente en `must_have_before_full_campaign`;
- existe `campaign spec` formal;
- la capa meta y de resultados cumple el minimo descrito en [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md);
- la política de persistencia, interpretabilidad y comparabilidad en `raw space` cumple el minimo descrito en [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md);
- existen o se regeneran de forma determinista todos los datasets;
- los runners `MLP` y `XGBoost` funcionan en smoke test real;
- se ha medido el tiempo real de ejecucion con un piloto representativo;
- el preflight final de lanzamiento está completo;
- la trazabilidad por `dataset_candidate_id`, `run_spec_id`, `seed_set_id` y `trial_id` queda garantizada.

Estado observado al cierre de esta checklist:

- el preflight final de lanzamiento queda satisfecho por:
  - [f7_launch_readiness_v1.json](../outputs/reports/f7_launch_readiness/f7_launch_readiness_v1.json)
  - [f7_launch_readiness_v1.md](../outputs/reports/f7_launch_readiness/f7_launch_readiness_v1.md)
- resultado:
  - `go_no_go = go`
  - `blockers = []`
