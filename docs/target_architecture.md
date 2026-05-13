# Target Architecture

## Propósito del documento

Este documento define la **arquitectura objetivo implementable** del proyecto para cerrar bien el TFG y dejar el repositorio en una forma publicable, reproducible y defendible.

Su relación con los otros documentos canónicos es esta:

- `docs/project_context.md` es la fuente canónica del **estado actual** del proyecto.
- `docs/target_architecture_north_star.md` es la fuente canónica del **estado deseado / north star**.
- Este documento aterriza ambas cosas en una **arquitectura objetivo realista**, alineada con el repo actual y con el tiempo restante del TFG.

Regla de interpretación:

- `estado actual` = lo que ya existe u opera hoy;
- `pendiente` = trabajo necesario para cerrar bien el TFG;
- `objetivo` = cómo debe quedar el proyecto cuando la fase de cierre esté bien resuelta.

## 1. Resumen ejecutivo

El objetivo no es rehacer el proyecto desde cero ni imponer una estructura perfecta en abstracto. El objetivo es dejar un repo claro, defendible y publicable, donde la lógica central viva en código reutilizable, el split oficial sea temporal, el leakage esté controlado y los resultados finales puedan reproducirse sin depender de notebooks operativas.

La arquitectura objetivo debe **apoyarse en la estructura raíz actual** del repositorio. `data/`, `models/`, `losses/`, `training/`, `config/`, `training_scripts/`, `notebooks/` y `outputs/` ya contienen lógica y valor reales. La prioridad es separar mejor responsabilidades y formalizar piezas que hoy faltan, no mover artificialmente todo dentro de `cement_imbalance/`.

Metodológicamente, `MLP` sigue siendo la línea predictiva principal; `FlowPre` sigue siendo un componente central; `FlowGen` sigue siendo una línea importante pero subordinada al objetivo predictivo; `CVAE-CNF` queda como línea histórica o experimental, no como eje del cierre.

El repo final debe distinguir con claridad entre:

- material público reproducible;
- material privado o sensible que debe quedarse local o ignorado por git;
- artefactos canónicos;
- artefactos regenerables;
- material histórico.

## 2. Target architecture

### 2.1 Objetivo final del proyecto

El proyecto final debe permitir, de forma reproducible y defendible:

1. cargar y validar los datos fuente permitidos;
2. aplicar anonimización y preprocesado coherentes con las restricciones reales del proyecto;
3. construir un split temporal oficial y reutilizable;
4. limpiar, transformar y construir datasets sin leakage;
5. entrenar y comparar variantes metodológicas relevantes;
6. agregar resultados por seeds y particiones;
7. producir tablas, figuras y notebooks finales del TFG;
8. dejar un repositorio público serio, con frontera explícita entre lo reproducible y lo sensible.

### 2.2 Alcance imprescindible para cerrar bien el TFG

Estas piezas son `must-have`:

- split oficial `train/val/test` de naturaleza temporal;
- cuantificación mínima del drift entre particiones;
- política explícita anti-leakage;
- pipeline reproducible de preparación de datos, dataset building, entrenamiento, evaluación y reporting;
- comparación de un conjunto acotado de variantes relevantes;
- resultados agregados por seeds;
- selección de finalistas en `val`;
- confirmación única en `test`;
- notebooks finales limpias para el TFG;
- repo publicable sin fuga de información sensible.

### 2.3 Deseable pero opcional

Estas piezas suman valor, pero no deben bloquear el cierre:

- baseline adicional como `XGBoost`;
- comparación con `SMOTE` o `KMeans-SMOTE`;
- bootstrap o intervalos más completos;
- reporting estadístico más sofisticado;
- notebook o apéndice extra centrado en realismo sintético.

### 2.4 Fuera de alcance

Estas líneas no deben dominar la arquitectura objetivo:

- convertir `CVAE-CNF` en la línea principal;
- ampliar el cierre a targets distintos de `init`;
- empaquetar todo dentro de `cement_imbalance/` por estética;
- rehacer el proyecto entero desde cero;
- despliegue productivo;
- refactor estructural grande sin impacto claro en reproducibilidad o claridad metodológica.

### 2.5 Principios de diseño no negociables

- `Reproducibilidad`: cada experimento importante debe quedar trazado por config, split, seed, métricas y artefactos.
- `Anti-leakage`: todo lo que aprende se ajusta solo en `train`.
- `Split-first`: el split oficial se fija antes de tuning, selección o comparación final.
- `Val-first`: la selección se hace en `val`; `test` queda bloqueado hasta el cierre.
- `Comparación justa del cierre`: la comparación final entre combinaciones de datasets con `MLP` debe usar el mismo split oficial, la misma `closure seed set` y el mismo `base config` congelado.
- `Retune acotado desde priors`: `FlowPre` y `FlowGen` deben revalidarse para el cierre temporal partiendo de los mejores configs actuales como priors heredados del split aleatorio y con una búsqueda local pequeña, no con una reoptimización masiva.
- `Responsabilidades separadas`: datos, transformaciones, entrenamiento, evaluación y reporting no deben vivir mezclados.
- `Notebooks no operativas`: las notebooks finales pueden explicar y visualizar; no deben ser el mecanismo principal de entrenamiento.
- `Conservación de lógica valiosa`: una mala ubicación actual no implica que la lógica deba eliminarse.
- `Minimalismo estructural`: solo se añaden capas nuevas si resuelven una responsabilidad real.
- `Frontera público/privado`: todo lo sensible debe quedar fuera del repo público o en rutas locales ignoradas por git.

### 2.6 Decisiones metodológicas que siguen abiertas

La arquitectura debe **soportar** estas decisiones, no cerrarlas artificialmente:

- fórmula final del criterio `fair`;
- peso final de `FlowGen` en el cierre del TFG;
- inclusión definitiva de `XGBoost`;
- inclusión definitiva de `SMOTE` o `KMeans-SMOTE`;
- shortlist final de combinaciones a comparar;
- vecindario y presupuesto final de la búsqueda local de `FlowPre` y `FlowGen` bajo split temporal;
- `closure seed set` y `MLP base config` finales para la comparación de cierre;
- promoción o no de ciertos datasets derivados a artefactos oficiales.

### 2.7 Pipeline objetivo a nivel conceptual

La pipeline objetivo del proyecto debe ser esta:

1. `Carga y validación`
   Lectura del dataset fuente permitido y validación mínima de esquema, columnas obligatorias y tipos.
2. `Anonimización y transformaciones seguras`
   Aplicación de mappings y transformaciones necesarias para cumplir las restricciones del proyecto.
3. `Limpieza determinista`
   Reglas físicas o imposibles aplicables sin depender del split, incluyendo quality filters de dominio aceptados como el control de `sum_chem` y `sum_phase`.
4. `Split oficial`
   Construcción y congelación del split temporal `train/val/test` con metadatos reutilizables.
5. `Limpieza estadística y transforms`
   Ajuste en `train` de outlier workflows, escalados y `FlowPre`, con aplicación posterior a `val/test`; en la política canónica de cierre, el holdout learned cleaning debe ser `flag-only`.
6. `Dataset building`
   Construcción de variantes de dataset: originales, escaladas, latentes con `FlowPre`, y versiones aumentadas si se usan.
   Aquí deben vivir solo los ejes `dataset-level`: `x_transform`, `y_transform` y `synthetic_policy`.
7. `Entrenamiento`
   Entrenamiento de `MLP` como línea principal y de componentes auxiliares como `FlowGen` cuando aplique.
   Aquí deben vivir los ejes `run-level`, como `batch_policy`, `cycling_policy`, `loss_policy` y `objective_metric`.
8. `Evaluación`
   Métricas predictivas, métricas por clase, métricas de drift, métricas de realismo sintético cuando sean necesarias y agregación por seeds.
9. `Reporting`
   Tablas, figuras, estadísticas finales y notebooks de cierre.

## 3. Estructura objetivo del repo

### 3.1 Criterio general

La estructura objetivo debe ser **mínima y realista**. La opción preferida es mantener la estructura raíz actual y mejorar la separación de responsabilidades dentro de ella.

No se propone una reestructuración agresiva ni una paquetización artificial. `reports/` y `scripts/` solo se justifican como capas ligeras. Si alguna de sus responsabilidades puede vivir inicialmente dentro de `evaluation/` sin empeorar la claridad, eso es aceptable.

### 3.2 Estructura objetivo sugerida

```text
docs/
config/
data/
models/
losses/
training/
evaluation/
notebooks/
training_scripts/
outputs/
reports/   # opcional, ligera
scripts/   # opcional, ligera
cement_imbalance/
```

### 3.3 Responsabilidad objetivo por bloque

- `docs/`
  Debe contener contexto, arquitectura objetivo, contrato de datos público, política de split, política de artefactos y criterios metodológicos.

- `config/`
  Debe contener solo configs canónicas y mantenidas a mano.
  Los YAMLs generados por Optuna, runners masivos o re-seeds no deben acabar como canon aquí.

- `data/`
  Debe ser la casa de:
  carga,
  validación,
  preprocessing,
  limpieza,
  split,
  transforms,
  dataset building.

  La separación puede ser inicialmente conceptual dentro de `data/`, sin necesidad de crear muchas carpetas nuevas desde el primer día.

- `models/`
  Debe contener definiciones puras de modelos y wrappers arquitectónicos.

- `losses/`
  Debe contener funciones de pérdida y criterios matemáticos puros.

- `training/`
  Debe contener loops de entrenamiento, dataloaders, tuning y utilidades estrictamente ligadas al entrenamiento.

- `evaluation/`
  Debe pasar a ser una capa real y canónica para:
  drift,
  métricas predictivas,
  métricas de realismo,
  agregación de resultados,
  selección de finalistas,
  estadística mínima de comparación.

- `reports/`
  Si se introduce, debe ser una capa **muy pequeña** para tablas, figuras y artefactos finales.
  No debe convertirse en un subsistema grande.

- `scripts/`
  Si se introduce, debe ser una capa **muy pequeña** de entrypoints canónicos.
  No debe duplicar lógica de `training/` ni reemplazar innecesariamente a `training_scripts/`.

- `notebooks/`
  El estado objetivo debe distinguir entre dos tipos de notebooks, aunque al principio sigan físicamente en la misma carpeta:
  notebooks finales del TFG,
  notebooks históricas o de workbench o archive.

  Las primeras son parte del producto final.
  Las segundas son referencia histórica, apoyo exploratorio o material de rescate.

- `training_scripts/`
  Debe quedar como capa histórica o semioperativa de runners versionados.
  Puede seguir existiendo, pero no debe ser la API canónica del repo público final.

- `outputs/`
  Debe tratarse siempre como generado y regenerable.
  Los nuevos runs canónicos deberían concentrarse en convenciones claras dentro de `outputs/`.
  Para la línea temporal de cierre ya queda implementado un namespace separado del histórico bajo `outputs/models/official/<family>/`.

- `cement_imbalance/`
  No es el eje de la arquitectura.
  No se propone mover todo ahí.
  Si no aporta valor real, puede terminar quedándose como shim mínimo o incluso desaparecer en una fase posterior.

## 4. Tratamiento de componentes actuales

### 4.1 Núcleo activo a preservar

Estas piezas forman parte del núcleo real del proyecto y deben preservarse:

- `data/preprocess.py`
- `models/flow_pre.py`
- `models/flowgen.py`
- `models/mlp.py`
- `losses/flow_pre_loss.py`
- `losses/flowgen_loss.py`
- `losses/mlp_loss.py`
- `training/train_flow_pre.py`
- `training/train_flowgen.py`
- `training/train_mlp.py`
- `training/optuna_mlp.py`

`FlowPre` y `FlowGen` siguen siendo componentes canónicos del cierre, pero sus mejores hiperparámetros actuales no deben tratarse automáticamente como configuración final canónica para el split temporal. Deben entenderse como priors operativos heredados del split aleatorio histórico hasta que se revaliden bajo la metodología temporal oficial. En el estado real actual, `FlowPre` ya acumula `162` runs oficiales completas en `outputs/models/official/flow_pre/` (`22` de `revalidate_v1`, `20` de `explore_v2`, `30` de `explore_v3`, `11` de `explore_v4` y `79` del reseed final). Esa evidencia ya no corresponde a una fase abierta: el reseed quedó cerrado y los finalistas operativos viven en `outputs/models/official/flowpre_finalists/`. El repo conserva además la capa evaluativa ampliada y operativa de `FlowPre` (`rrmse`, `mvn`, `fair`, `flowgencandidate`, `rrmse_primary`) y la capa global de sanidad de reconstrucción (`global_reconstruction_status`), pero deben leerse ya como la base metodológica usada para cerrar la fase, no como soporte provisional de una fase abierta.

Los picks finales por lente ya están congelados y trazados en `outputs/models/official/flowpre_finalists/README.md`; este documento no reabre esa selección, solo la incorpora como estado real de partida para la fase siguiente.

Interpretación vigente de esos finalistas:

- `rrmse`, `mvn` y `fair` deben leerse como scalers/upstreams especializados de `FlowPre` para escalar dataset y derivar variantes de dataset. No son las bases principales de entrenamiento de `FlowGen`.
- `candidate_1` y `candidate_2` son las dos bases reales de trabajo seleccionadas para arrancar y desarrollar la fase `FlowGen`.
- si alguna implementación legacy/provisional de `FlowGen` todavía referencia un promotion manifest `rrmse`, eso debe leerse solo como compatibilidad técnica heredada, no como la base conceptual ni operativa de la fase generativa.
- cuando `FlowGen` quede cerrado y existan sus outputs promovidos, `candidate_1` y `candidate_2` pasarán a ser artefactos históricos de trazabilidad y dejarán de ser artefactos activos/canónicos de uso.

### 4.2 Piezas a preservar, pero dividir o reubicar conceptualmente

- `data/cleaning.py`
  Debe preservarse, pero separar dos cosas que hoy están mezcladas:
  carga concreta del raw y decisiones privadas de negocio,
  limpieza estadística y detección de outliers reutilizable.

  El nombre hardcodeado del raw, filtros privados de producto o proceso y detalles sensibles no deben seguir incrustados como canon público.

- `data/splits.py`
  Debe evolucionar hacia la casa del split oficial y de sus metadatos.
  El split aleatorio/estratificado actual puede sobrevivir como baseline o referencia histórica, pero no como verdad canónica final.

- `data/sets.py`
  Debe preservarse porque contiene la lógica operativa real para construir datasets.
  Debe separarse mejor entre:
  construcción de datasets,
  persistencia de artefactos,
  checks o plots de QA.

  `data/sets/` debe seguir tratándose como almacenamiento de derivados regenerables, no como fuente primaria de verdad.

- `training/utils.py`
  Hoy funciona como punto de acoplamiento transversal.
  Debe mantenerse inicialmente como compatibilidad, pero su responsabilidad debería partirse en:
  paths/config,
  logging de runs,
  IO de datasets derivados.

- `training/eda.py`
  Debe tratarse como módulo exploratorio.
  De ahí conviene rescatar solo lo reusable para comparativas latentes, UMAP, resúmenes y plots finales si realmente se usan.
  No debe quedar como pilar del producto final.

- `evaluation/`
  Debe convertirse en capa real.
  Parte de la evaluación hoy embebida en `training/train_flowgen.py`, `training/train_mlp.py`, notebooks y runners debe migrar aquí de forma selectiva y de bajo riesgo.

### 4.3 Piezas a mantener como históricas o secundarias

- `CVAE-CNF`
  Se preserva como línea histórica o experimental.
  No se elimina por defecto, pero tampoco se considera parte del target principal del cierre.

- `training_scripts/`
  Los scripts versionados, especialmente las últimas `1-2` versiones por familia, deben preservarse como referencia operativa e histórica.
  No deben convertirse en la capa canónica del repo público final.

- `notebooks` actuales
  No deben descartarse automáticamente.
  Muchas contienen decisiones útiles, checks y exploración recuperable.
  Su papel objetivo pasa a ser de referencia histórica, no de entrypoint principal.

- `FlowGen temperature tuning`
  Debe preservarse como capacidad experimental o histórica ya implementada.
  No forma parte del camino canónico del cierre salvo promoción explícita posterior.

- `outputs/` acumulados
  Deben mantenerse como archivo local útil, pero no como base arquitectónica.
  Los outputs antiguos de sweeps, retraining y exploración son históricos.

- directorios derivados adicionales como `data/augmented/` o `data/flow_pre_normalized/`
  Deben tratarse como material derivado o histórico salvo promoción explícita posterior.

### 4.4 Tratamiento específico por carpeta principal

- `data/`
  Preservar y ordenar.
  No moverla fuera del eje del proyecto.

- `models/`
  Preservar tal como está a nivel conceptual.

- `training/`
  Preservar como capa de entrenamiento.

- `evaluation/`
  Activar como capa canónica real.

- `losses/`
  Preservar como capa matemática pura.

- `training_scripts/`
  Mantener como runners históricos/versionados.

- `notebooks/`
  Distinguir claramente entre finales y de referencia histórica.

- `outputs/`
  Mantener como generado/local.

- `config/`
  Mantener como fuente de configs base, pero separar lo público de lo privado y expulsar lo generado del canon.

- `cement_imbalance/`
  Mantener como pieza menor o prescindible.
  No reorganizar el repo alrededor de ella.

## 5. Piezas nuevas necesarias

La arquitectura objetivo necesita pocas piezas nuevas, pero sí algunas imprescindibles.

- `Manifest del split oficial`
  Una pieza mínima que describa versión de split, criterio, fechas o ventanas, conteos por clase y política de bloqueo de `test`.

- `Capa de drift`
  Un módulo o script pequeño para cuantificar drift entre particiones y producir 2-3 salidas claras para el TFG.

- `Registro de resultados`
  Una tabla agregada y consistente para dejar de depender de YAMLs dispersos en `outputs/`.

- `Provenance de datasets derivados`
  Un manifest mínimo por dataset derivado con split, `cleaning_policy_id`, transformaciones, origen del modelo si aplica y conteos por clase.

- `Capa ligera de reporting`
  Puede vivir en `evaluation/` o en un `reports/` pequeño.
  Su función es generar tablas, figuras y estadísticas finales, no crear otro subsistema.

- `Entry points canónicos ligeros`
  Puede vivir en `scripts/` o equivalente.
  Su función es exponer el flujo final de manera clara y estable, no reemplazar toda la operativa histórica.

- `Notebooks finales`
  El estado objetivo debe incluir un conjunto pequeño de notebooks limpias de cierre, por ejemplo:
  `01_data_constraints_and_eda`,
  `02_temporal_split_and_drift`,
  `03_variant_comparison`,
  `04_final_results_and_error_analysis`.

## 6. Artefactos y fuentes de verdad

### 6.1 Fuente de verdad canónica

Deben actuar como fuente de verdad:

- código fuente mantenido;
- documentación canónica en `docs/`;
- configs base canónicas;
- contrato de datos público;
- metadata del split oficial;
- resultados finales agregados que formen parte del cierre;
- criterio explícito para distinguir artefactos oficiales de históricos.

### 6.2 Regenerables

Deben considerarse regenerables:

- `data/processed/`;
- `data/cleaned/`;
- `data/splits/` como materialización de datasets;
- `data/sets/`;
- `data/sets/scaled_sets/` como namespace legacy/histórico;
- `data/sets/augmented_scaled_sets/`;
- checkpoints;
- logs;
- outputs intermedios;
- muestras sintéticas;
- tablas auxiliares o figures temporales;
- configs generadas por Optuna o runners masivos.

Los derivados promovidos al canon físico deben vivir bajo el árbol oficial versionado, por ejemplo:

- `data/sets/official/<split_id>/raw/<dataset_name>/`;
- `data/sets/official/<split_id>/scaled/<dataset_storage_name>/`.

### 6.3 Históricos

Deben considerarse históricos:

- notebooks operativas antiguas;
- scripts versionados antiguos;
- sweeps y retrainings acumulados;
- outputs top-level heredados sin papel en el cierre final;
- modelos o datasets que no entren en la shortlist final del TFG.

### 6.4 Frontera entre material público y material privado/local

El repo final se quiere hacer público. Por tanto, la arquitectura objetivo debe distinguir explícitamente entre:

- `versión pública anonimizada`;
- `versión privada local`.

Esto aplica, como mínimo, a estas piezas:

- `dataset fuente`
  versión pública: no incluida, o solo muestra sintética o contrato de esquema;
  versión privada local: raw real y cualquier export industrial.

- `mappings de columnas y clases`
  versión pública anonimizada: nombres anonimizados, familias de columnas y clases abstractas o ya anonimizadas;
  versión privada local: nombres reales, productos reales, mappings NDA.

- `documentación metodológica sensible`
  versión pública: justificación metodológica sin revelar información privada;
  versión privada local: notas con contexto industrial, decisiones sensibles o interpretación de variables no publicable.

- `configuración sensible`
  versión pública: configs base reproducibles sin paths, nombres ni detalles privados;
  versión privada local: rutas, nombres reales, overlays privados y cualquier configuración no publicable.

- `metadatos temporales`
  versión pública: política de split y estadísticas agregadas;
  versión privada local: detalles temporales exactos si fuesen sensibles.

Regla operativa:

- todo lo sensible debe vivir solo en local, fuera del repo público o en rutas ignoradas por git;
- cuando una pieza necesite existir en ambos mundos, debe concebirse como `pública anonimizada` y `privada local`;
- la versión privada no debe convertirse en dependencia estructural del repo público.

### 6.5 Política de git y almacenamiento local

Parte de la configuración y documentación sensible deberá vivir:

- solo en local;
- o en rutas bajo `.gitignore`;
- o fuera del árbol publicable del repositorio.

Esto incluye, en principio:

- raw data;
- mappings reales;
- notas privadas;
- overlays de config privada;
- outputs pesados;
- artefactos masivos de sweeps;
- ficheros temporales de tuning;
- bases de datos locales de estudios;
- cualquier documento que revele nombres, escalas o transformaciones confidenciales.

## 7. Plan de migración por fases

La migración debe ser de bajo riesgo y sin sobrerreacción estructural.

### Fase 1. Cerrar el canon documental

- fijar `project_context.md`, `target_architecture.md` y la política de artefactos;
- dejar explícita la frontera público/privado;
- definir la taxonomía `actual / pendiente / objetivo`;
- fijar por escrito el estatus `canon actual / pendiente / experimental-histórico`;
- dejar explícito que los mejores hiperparámetros heredados de `FlowPre` y `FlowGen` son priors operativos del split aleatorio, no canon final para el cierre temporal;
- dejar explícita la política de comparación final justa con `MLP` usando mismas seeds y mismo `base config` congelado;
- dejar explícita la exclusión canónica actual del `FlowGen temperature tuning`.

### Fase 2. Formalizar split y contrato

- fijar el split temporal oficial;
- crear su manifest;
- fijar un contrato de datos público y, si hace falta, un overlay privado local.

### Fase 3. Separar responsabilidades sin romper la operativa

- estado actual: cerrada en la capa de `data/`;
- el cleaning estadístico split-aware ya vive sobre el split oficial y su política canónica actual se versiona bajo `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/`;
- el raw dataset building canónico ya vive bajo `data/sets/official/init_temporal_processed_v1/raw/df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1/`;
- `data.sets.load_or_create_raw_splits()` ya consume por defecto el bundle oficial y mantiene compatibilidad con downstream;
- el canon operativo llega de forma segura hasta ese raw bundle oficial versionado; los datasets derivados siguen fuera del canon hasta regeneración y manifest canónico;
- F4 parte desde aquí para activar evaluación y reporting, no para rehacer esta separación.

### Fase 4. Introducir capas ligeras faltantes

- estado actual: cerrada en `evaluation/` como capa ligera y canónica para métricas, normalización de resultados, agregación por seeds y comparación básica por familia;
- estado real adicional: sobre esa base ya existe una superficie operativa de evaluación expandida para `FlowPre` bajo `outputs/reports/f6_explore_v2_results/`; sus lentes adicionales (`flowgencandidate`, `rrmse_primary`) y la capa global de sanidad de reconstrucción ya se usaron para cerrar `FlowPre`, pero ese reporting debe leerse hoy como snapshot histórico frente a la referencia vigente de cierre en `outputs/models/official/flowpre_finalists/`;
- activar `evaluation/` como capa real;
- introducir, solo si compensa, una capa ligera de `reports/` y otra de `scripts/`;
- distinguir notebooks finales frente a históricas.

### Fase 5. Cierre experimental

- estado actual: la subfase `FlowPre` está cerrada. Existen `162` runs oficiales completas bajo el split temporal, el reseed final quedó en `79/79` y los finalistas quedaron materializados.
- estado real adicional: `FlowGen` ya no está pendiente de arranque ni abierto. Existen `50` runs oficiales comparables bajo `outputs/models/official/flowgen/` entre exploración y reseed final, una capa histórica de ranking exploratorio en `scripts/f6_flowgen_rank_official_v2.py`, una capa canónica de agregación post-reseed en `outputs/models/official/flowgen/campaign_summaries/post_reseed/`, y un finalista único materializado en `outputs/models/official/flowgen_finalist/` con winner final `flowgen_tpv1_c2_train_e03_seed2468_v1`.
- estado real adicional: existe también una rama local `train_only` ya cerrada bajo `outputs/models/experimental/train_only/`, con dos bases finales de `FlowPre train_only` y un finalista local único de `FlowGen train_only`; esa rama sirve como input downstream experimental y no como reapertura del canon.
- foco inmediato: usar el winner ya cerrado de `FlowGen` como input aguas abajo del cierre experimental, sin reabrir reseed ni búsqueda local
- decisión abierta asociada: determinar si el finalista local `train_only` entra o no en la shortlist downstream de `F7`
- congelar `MLP base config` y `closure seed set`;
- ejecutar la shortlist final de comparaciones;
- comparar combinaciones finales con `MLP` bajo mismas seeds y mismo `base config`;
- agregar resultados por seeds;
- hacer la selección en `val`;
- confirmar una vez en `test`;
- mantener `FlowGen temperature tuning` fuera del camino canónico salvo rescate explícito.

### Fase 6. Cierre público del repo

- decidir qué se publica y qué queda local;
- limpiar lo que deba ignorarse por git;
- dejar README, notebooks finales y artefactos ligeros de cierre.

## 8. Riesgos, tensiones y decisiones abiertas

- `fair`
  La fórmula final sigue abierta. La arquitectura debe soportar varias variantes sin reescribir el repo.

- `integración del split oficial`
  El contrato del split temporal oficial ya está fijado. Lo que sigue abierto es cómo integrar aguas abajo cleaning, transforms y dataset building sobre ese contrato sin leakage.

- `peso final de FlowGen`
  Sigue abierto cuánto protagonismo tendrá en el cierre downstream y si bastará con el winner `official` o convendrá contrastarlo además con la rama local `train_only`.
  La arquitectura debe permitir tanto un papel fuerte como uno de apoyo o apéndice.

- `estado formal de F6 FlowPre`
  La subfase `FlowPre` ya está cerrada: hay finalistas materializados. `rrmse`, `mvn` y `fair` deben seguir leyéndose como scalers/upstreams especializados; `candidate_1` y `candidate_2` como bases históricas de procedencia de la exploración generativa; y cualquier referencia residual a un `promotion_manifest` `rrmse` solo como compatibilidad técnica heredada. `FlowGen` ya quedó también cerrado con un finalista único promovido, por lo que la tensión principal del proyecto ya no es cerrar `FlowPre` ni `FlowGen`, sino el cierre downstream con `MLP`.

- `baselines externos`
  `XGBoost` y `SMOTE/KMeans-SMOTE` siguen siendo decisiones abiertas.

- `promoción de derivados`
  No todos los datasets materializados merecen elevarse a artefacto oficial.

- `frontera público/privado`
  Requiere disciplina explícita; el repo actual todavía no la refleja suficientemente.

- `riesgo de romper lógica útil`
  Parte de la lógica valiosa está escondida en notebooks y scripts históricos.
  El refactor debe extraer con criterio, no podar por ubicación.

- `tiempo del TFG`
  El mayor riesgo no es técnico sino de foco.
  La arquitectura debe ayudar a cerrar el TFG, no abrir un proyecto de limpieza infinito.

## Criterio de aceptación final

Se considerará que la arquitectura objetivo está bien cerrada cuando el proyecto permita:

- reproducir el flujo principal sin depender de notebooks operativas;
- justificar el split temporal con evidencia mínima de drift;
- controlar explícitamente el leakage;
- comparar variantes bajo una política clara de validación y test;
- distinguir con nitidez entre canónico, regenerable e histórico;
- distinguir con nitidez entre público y privado;
- sostener un cierre del TFG técnicamente defendible y publicable.
