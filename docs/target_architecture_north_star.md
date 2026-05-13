# Documento aspiracional del proyecto (Fase C / Target Architecture)

## 1. Propósito del documento

Este documento define el **estado objetivo** del proyecto para el cierre del TFG y para dejar el repositorio en una forma sólida, reproducible y presentable en GitHub.

Tiene dos metas simultáneas:

1. convertir el proyecto en un **repo sano, modular y reproducible**, donde la lógica principal viva en código y no en notebooks operativas;
2. cerrar el TFG con una narrativa y metodología de **alta señal estadística**, centrada en evaluación honesta bajo `distribution shift`, control de leakage y evidencia defendible.

Este documento **no sustituye** a `docs/project_context.md`.

- `docs/project_context.md` describe el **estado actual**, la historia canónica y el trabajo pendiente.
- Este documento describe el **estado objetivo**: cómo debería quedar el proyecto para cerrar bien el TFG.
- Nota de alineación con el estado real actual: el repo ya dispone de una capa evaluativa operativa ampliada para `FlowPre`, incluyendo `rrmse_primary`, `flowgencandidate` y una capa global de sanidad de reconstrucción. Esa superficie ya se ha usado para cerrar `FlowPre`: existen `162` runs oficiales completas y finalistas materializados en `outputs/models/official/flowpre_finalists/`. Si sigue existiendo un manifest `rrmse`, debe leerse solo como compatibilidad técnica heredada, no como base operativa principal de `FlowGen`.
- Los picks finales por lente ya están congelados y trazados en `outputs/models/official/flowpre_finalists/README.md`; este documento no los reabre, sino que asume ese cierre como punto de partida del siguiente tramo.
- Interpretación vigente de esos finalistas: `rrmse`, `mvn` y `fair` son scalers/upstreams especializados de `FlowPre` para escalar dataset y derivar variantes; `candidate_1` y `candidate_2` son las dos bases reales de trabajo para arrancar y desarrollar `FlowGen`; y, una vez `FlowGen` quede cerrado, `candidate_1` y `candidate_2` pasarán a ser trazabilidad histórica y dejarán de ser artefactos activos/canónicos.
- Nota adicional de alineación: además del cierre `official`, hoy existe una rama local `train_only` ya cerrada con finalistas propios bajo `outputs/models/experimental/train_only/`; esta rama se interpreta como soporte downstream experimental y no como reapertura del cierre canónico.

---

## 2. Objetivo final del proyecto

El proyecto final debe permitir, de forma reproducible y defendible:

- cargar y validar el dataset fuente;
- aplicar anonimización y preprocesado coherentes con las restricciones reales del proyecto;
- construir un split temporal correcto;
- limpiar y transformar datos evitando leakage;
- generar variantes relevantes de dataset;
- entrenar y comparar modelos principales;
- evaluar resultados con criterios estadísticos claros;
- producir tablas, figuras y notebooks finales del TFG;
- y dejar una estructura de repo que sirva como portfolio técnico serio.

El resultado final no debe ser solo “un conjunto de experimentos que funcionaron”, sino una **pipeline defendible** que responda con claridad:

- qué decisiones metodológicas mejoran realmente la predicción;
- cuánto mejoran;
- bajo qué condiciones;
- y qué pipeline se recomendaría en producción para predecir el futuro cercano.

---

## 3. Alcance: must-have, nice-to-have y fuera de alcance

### 3.1 Must-have para cerrar bien el TFG

Estas piezas deben entrar sí o sí en la arquitectura objetivo:

- split cronológico `train/val/test` como fuente de verdad;
- cuantificación mínima del drift entre particiones;
- control explícito de leakage;
- pipeline reproducible para:
  - preparación de datos,
  - construcción de datasets,
  - entrenamiento,
  - evaluación,
  - reporting;
- comparación defendible de variantes relevantes;
- resultados agregados por seeds;
- tablas/figuras finales reproducibles;
- notebooks finales limpias para el TFG;
- cierre estadístico con análisis de efectos y conclusiones no-humo.

### 3.2 Nice-to-have si entra por tiempo

Estas piezas suman valor, pero no deben bloquear el cierre:

- baseline adicional tipo `XGBoost`;
- comparación con `SMOTE` / `KMeans-SMOTE`;
- intervalos bootstrap más completos;
- mejoras extra de interpretabilidad;
- formalización más sofisticada del análisis estadístico.

### 3.3 Fuera de alcance por ahora

Estas cosas no deben dominar el diseño ni desviar tiempo:

- ampliar el proyecto a targets distintos de `init`;
- rescatar `CVAE-CNF` como línea principal;
- despliegue o producto de producción;
- refactor puramente estético sin impacto en reproducibilidad o claridad;
- rehacer todo el proyecto desde cero.

---

## 4. Principios de diseño no negociables

La arquitectura objetivo debe respetar estos principios.

### 4.1 Reproducibilidad

- todo experimento debe quedar trazado por `config + seed + split + métricas + artefactos`;
- las tablas/figuras finales deben poder regenerarse;
- `test` debe permanecer bloqueado hasta el cierre.

### 4.2 Anti-leakage

- todo lo que “aprende” debe ajustarse solo en `train`;
- el split debe decidirse antes del tuning;
- los sintéticos deben generarse solo desde `train`;
- la selección de finalistas se hace solo en `val`.

### 4.3 Separación de responsabilidades

- preparación de datos, entrenamiento, evaluación y reporting no deben vivir mezclados;
- notebooks no deben ser el lugar principal de la lógica de entrenamiento;
- outputs y datasets derivados no deben ser la fuente de verdad.

### 4.4 Conservación de lógica valiosa

- no eliminar scripts/notebooks solo por estar mal ubicados;
- preservar lógica útil aunque luego se extraiga, divida o reubique.

### 4.5 Realismo metodológico

- la evaluación debe reflejar el caso de uso real bajo `distribution shift`;
- las comparaciones deben ser justas y explícitas;
- el cierre debe priorizar evidencia defendible frente a complejidad innecesaria.

---

## 5. Pipeline objetivo a nivel conceptual

La arquitectura final debe soportar una pipeline conceptual como esta.

### 5.1 Carga y validación de datos fuente

- lectura del dataset raw;
- comprobaciones mínimas de esquema y columnas obligatorias.

### 5.2 Anonimización y transformaciones seguras

- aplicación de transformaciones monotónicas requeridas por las restricciones de privacidad del proyecto;
- documentación de su papel dentro del pipeline.

### 5.3 Reglas de limpieza

- eliminación o marcado de valores físicamente imposibles;
- política explícita de outliers:
  - imposibles con reglas globales;
  - outliers estadísticos ajustados solo en `train`.

### 5.4 Construcción del split oficial

- split temporal por fecha;
- generación de metadatos de partición;
- cuantificación de drift `train→val` y `train→test`.

### 5.5 Construcción de variantes de dataset

- variantes de escalado para `X`;
- variantes de escalado para `Y`;
- variantes `FlowPre`;
- posibles variantes con o sin sintéticos;
- estrategias de balancing acordadas.

### 5.6 Entrenamiento de componentes intermedios

- entrenamiento de `FlowPre` como transformación multivariante invertible;
- entrenamiento de `FlowGen` si se usa la línea sintética.

### 5.7 Entrenamiento predictivo

- `MLP` como línea principal;
- baseline opcional como `XGBoost` si se incluye.

### 5.8 Evaluación y comparación

- métricas por clase y agregadas;
- estabilidad por seed;
- tablas de resultados consolidadas;
- selección de finalistas en `val`;
- confirmación única en `test`.

### 5.9 Reporting final

- tablas del paper;
- figuras;
- notebooks finales;
- análisis estadístico de efectos;
- análisis breve de errores / interpretabilidad.

---

## 6. Espacio experimental que la arquitectura debe soportar

La arquitectura final no tiene que fijar todavía todas las combinaciones, pero sí debe ser capaz de soportar un espacio experimental reducido y defendible.

### 6.1 Ejes principales de comparación

#### Scaling / transformación

- `StandardScaler`
- `RobustScaler`
- `QuantileTransformer`
- `MinMaxScaler`
- variantes `FlowPre` sobre `X`
- escalado independiente de `Y`

#### Balancing / dataset strategy

- sin balancing;
- balancing por weighting;
- balancing vía batches;
- balancing con sintéticos.

#### Synthetic data

- none;
- `FlowGen`;
- comparador opcional tipo `SMOTE` / `KMeans-SMOTE` si entra por tiempo.

#### Batching / sampling

- proporción original;
- batches equilibrados;
- cycling / oversampling de minoritarias.

#### Loss weighting

- proporcional al dataset original;
- balanceada por proporciones;
- inversamente proporcional a tamaños por clase.

#### Criterio de selección

- `R2`;
- `RRMSE`;
- global vs macro / media por clase.

### 6.2 Regla importante

Este espacio debe aparecer en la arquitectura como **capacidad a soportar**, no como promesa de ejecutar toda la combinatoria.

La selección final de configuraciones debe quedar acotada por:

- relevancia metodológica;
- tiempo disponible;
- claridad del mensaje final del TFG.

---

## 7. Arquitectura objetivo del repositorio

No hace falta clavar un layout exacto, pero sí separar responsabilidades. Además, la propuesta debe estar alineada con el árbol real actual del proyecto.

### 7.1 Restricción práctica importante

El repo actual ya tiene bloques útiles en:

- `data/`
- `evaluation/`
- `losses/`
- `models/`
- `training/`
- `training_scripts/`
- `notebooks/`
- `config/`

La arquitectura objetivo debe intentar **reutilizar y reorganizar** estas piezas antes que reinventarlas.

### 7.2 Aclaración importante sobre `cement_imbalance/`

Actualmente existe una carpeta `cement_imbalance/` al mismo nivel que el resto de carpetas principales, pero solo contiene un `__init__.py` y no actúa hoy como paquete real del proyecto.

Por tanto:

- **no** se desea forzar una migración donde todas las carpetas pasen a vivir dentro de `cement_imbalance/`, porque eso equivaldría a “meter el proyecto dentro de otro proyecto” sin una ganancia clara;
- si esa carpeta no aporta valor estructural real, debe considerarse **prescindible** o, como mínimo, no debe condicionar la arquitectura objetivo;
- la prioridad es una estructura clara y realista, no una paquetización artificial solo por estética.

### 7.3 Estructura objetivo sugerida

Hay dos caminos válidos:

- **A.** mantener carpetas raíz actuales pero limpiando responsabilidades;
- **B.** migrar progresivamente parte de la lógica reusable a un paquete central solo si aporta claridad real.

Para este TFG, la opción más realista es:

- mantener la estructura raíz para no romper demasiado;
- evitar mover todo dentro de `cement_imbalance/`;
- y solo introducir una capa central adicional si aporta una mejora clara y de bajo coste.

### 7.4 Responsabilidades objetivo por bloque

#### `data/`

Debe quedar centrado en:

- carga;
- validación de schema;
- limpieza;
- split oficial;
- construcción de datasets derivados.

No debería mezclar:

- análisis exploratorio;
- reporting;
- lógica de artefactos de entrenamiento;
- ni evaluación final.

#### `models/`

Debe contener:

- definiciones de modelos;
- wrappers limpios;
- piezas arquitectónicas puras.

No debería contener lógica de orquestación de experimentos.

#### `training/`

Debe contener:

- loops de entrenamiento;
- preparación de dataloaders;
- training routines;
- selección / checkpointing básico.

No debería absorber reporting ni lógica de análisis ad hoc.

#### `evaluation/`

Debe volverse una capa explícita y útil para:

- métricas predictivas;
- métricas de drift;
- métricas de realismo;
- agregación de resultados;
- comparación entre corridas.

Nota de alineación con el estado actual: el repo ya dispone de una capa evaluativa ampliada y operativa para `FlowPre`, y esa capa ya se usó para cerrar la fase y seleccionar finalistas. Los reports `f6*` y `explore*` siguen siendo útiles como trazabilidad histórica, pero la referencia vigente de cierre es `outputs/models/official/flowpre_finalists/README.md` junto al promotion manifest oficial del ganador `rrmse`.

#### `reports/` o equivalente

Aunque ahora no exista, hace falta una capa o scripts equivalentes para:

- tablas finales;
- figuras;
- ANOVA / post-hoc;
- resumen de resultados.

Puede vivir como nueva carpeta o como scripts claramente agrupados.

#### `config/`

Debe actuar como fuente de verdad para:

- configuraciones de datasets;
- configuraciones de modelos;
- configuraciones de experimentos;
- seeds;
- criterios de selección.

#### `notebooks/`

Debe quedar limitado a:

- EDA final;
- drift justification;
- resultados finales;
- error analysis / interpretability.

No debe seguir siendo el sitio principal donde se entrena o se gestiona la lógica del proyecto.

#### `outputs/`

Debe tratarse siempre como regenerable:

- checkpoints;
- logs;
- figuras temporales;
- resultados intermedios;
- outputs de sweeps.

No debe ser fuente de verdad.

---

## 8. Tratamiento de los componentes actuales

### 8.1 Preservar lógica pero probablemente dividir / reubicar

- `data/cleaning.py`
- `data/sets.py`
- `training/utils.py`
- `training/eda.py` (al menos revisión)
- algunos notebooks con lógica o análisis no migrados aún

### 8.2 Preservar como núcleo activo

- `models/flow_pre.py`
- `models/flowgen.py`
- `models/mlp.py`
- `training/train_flow_pre.py`
- `training/train_flowgen.py`
- `training/train_mlp.py`
- partes valiosas de `evaluation/metrics.py`, `evaluation/generate.py`, `evaluation/invert_flow_pre.py`
- configs relevantes en `config/`

### 8.3 Tratar como secundarios / experimentales / legacy

- `CVAE-CNF` completo salvo que alguna utilidad puntual se reaproveche;
- scripts versionados antiguos de `training_scripts/`;
- notebooks operativas antiguas;
- outputs y retrainings acumulados.

### 8.4 Mantener como referencia histórica, no como producto

- notebooks actuales;
- scripts de tandas históricas;
- outputs antiguos de sweeps / retrained.

---

## 9. Piezas nuevas que faltan sí o sí

La arquitectura objetivo necesita formalizar piezas que ahora no están bien separadas o no existen como tales.

### 9.1 Split temporal oficial

Hace falta una pieza explícita para:

- construir el split cronológico;
- guardar metadata de split;
- reutilizarlo de forma consistente.

### 9.2 Drift analysis

Hace falta una capa o script para:

- calcular KS / PSI / Wasserstein entre particiones;
- producir 2–3 plots claros;
- justificar metodológicamente el split temporal.

### 9.3 Experiment runner

Hace falta una pieza para:

- iterar configs;
- correr seeds;
- consolidar métricas;
- guardar resultados tabulares.

Nota de alineación con el estado actual: en el repo ya existen versiones ligeras de este rol para `FlowPre`; lo que sigue faltando es el cierre canónico completo, no partir de cero.

### 9.4 Result aggregation

Hace falta una capa clara para:

- `results.csv` / `results.parquet`;
- tablas resumidas;
- resultados por clase;
- medias, desviaciones e intervalos si se incluyen.

Nota de alineación con el estado actual: ya existe agregación/reporting ligero reconstruible desde artefactos para la primera tanda oficial de `FlowPre`, pero todavía sin promoción formal ni cierre completo de F6. El siguiente paso operativo inmediato puede ser exploración adicional controlada de `FlowPre`, no necesariamente completar de forma automática el presupuesto teórico del runner original.

### 9.5 Statistical reporting

Hace falta una pieza para:

- ANOVA / permutation ANOVA;
- post-hoc con corrección;
- tablas de efectos;
- integración con el reporte final.

### 9.6 Notebooks finales

Hacen falta notebooks limpias de cierre, por ejemplo:

- `01_eda_final.ipynb`
- `02_drift_and_split_justification.ipynb`
- `03_results_comparison.ipynb`
- `04_error_analysis_and_interpretability.ipynb`

---

## 10. Artefactos canónicos vs regenerables

### 10.1 Fuente de verdad

Deben ser fuente de verdad:

- `data/raw`
- configs canónicas
- código fuente
- metadata de split oficial
- documentación del contrato de datos

### 10.2 Regenerables

Deben considerarse regenerables:

- `data/processed`
- `data/cleaned`
- `data/splits`
- `data/sets`
- outputs de entrenamiento
- figuras
- tablas derivadas
- synthetic sets

### 10.3 Históricos / no canónicos

Deben considerarse históricos:

- barridos antiguos;
- notebooks operativas anteriores;
- reentrenamientos acumulados;
- outputs intermedios de exploración.

---

## 11. Data contract y trazabilidad

La arquitectura final debe incluir explícitamente un mínimo `data contract`, ya sea en `.md`, `.yaml` o ambos, con:

- columnas obligatorias;
- target oficial;
- columna de fecha;
- columna de clase (`cement_type`);
- rangos plausibles;
- reglas de imposibles;
- transformaciones necesarias y su propósito funcional.

### 11.1 Aclaración importante sobre privacidad y repo público

El repositorio final se quiere hacer público. Por tanto, cualquier detalle que pueda producir leakage de nombres reales, mappings sensibles, escalas originales, transformaciones confidenciales o información cubierta por NDA debe **quedarse fuera del repo público**.

Eso implica que:

- parte de la documentación sensible deberá vivir **solo en local** o en almacenamiento privado;
- ciertos archivos deberán ir en `.gitignore`;
- la versión pública del proyecto solo debe contener la información mínima necesaria para reproducibilidad **sin exponer detalles confidenciales**;
- el diseño debe preservar las buenas prácticas metodológicas, pero separando claramente lo **reproducible en público** de lo **sensible y privado**.

Además, todo dataset derivado importante debería poder rastrearse con:

- `split_version`;
- config usada;
- scaler / flow usado;
- conteos por clase;
- seeds relevantes;
- y si hubo o no filtrado de sintéticos imposibles.

---

## 12. Política de evaluación final

La arquitectura debe forzar o facilitar esta política:

- tuning y selección en `val`;
- `test` bloqueado hasta final;
- semillas fijas por configuración;
- misma lógica de comparación entre variantes;
- métrica objetivo predefinida;
- finalistas seleccionados en `val`;
- confirmación única en `test`.

---

## 13. Estadística mínima que debe soportar el repo

El repositorio final debe facilitar:

- resultados agregados por seed;
- comparación global y por clase;
- medida de variabilidad;
- CI bootstrap si entra por tiempo;
- ANOVA / permutation ANOVA sobre `val`;
- post-hoc corregido;
- discusión honesta de discrepancias entre `val` y `test`.

---

## 14. Entregables finales esperados

### 14.1 Entregables del repo

- `README` útil con reproducción básica;
- configs limpias y mínimas;
- scripts o runners de reproducción;
- resultados consolidados;
- scripts de tablas / figuras;
- notebooks finales.

### 14.2 Entregables del paper / TFG

- tabla de drift;
- tabla de resultados principales;
- tabla de efectos;
- figuras clave;
- conclusión final con pipeline recomendada y limitaciones.

---

## 15. Plan de migración por fases

La reestructuración debe ejecutarse con bajo riesgo y en fases.

### Fase 1. Inventario y clasificación

- identificar núcleo activo;
- distinguir histórico vs reutilizable;
- etiquetar `preservar / reubicar / dividir / deprecar`.

### Fase 2. Formalizar documentación y fuente de verdad

- `project_context.md`
- `target_architecture.md`
- data contract
- definición de split oficial

### Fase 3. Reorganización de bajo riesgo

- mover o extraer lógica sin cambiar comportamiento;
- separar funciones monolíticas;
- centralizar configs relevantes.

### Fase 4. Piezas nuevas imprescindibles

- split temporal reproducible;
- drift analysis;
- experiment runner;
- result aggregation;
- statistical reporting.

### Fase 5. Cierre experimental

- ejecutar comparaciones relevantes;
- consolidar tablas;
- seleccionar finalistas;
- confirmar en `test`.

### Fase 6. Cierre narrativo y portfolio

- notebooks finales;
- `README`;
- limpieza de outputs;
- repo listo para GitHub.

---

## 16. Riesgos y restricciones

La arquitectura objetivo debe respetar:

- restricciones de privacidad y NDA;
- tiempo limitado hasta el cierre del TFG;
- necesidad de no romper lógica útil escondida en scripts/notebooks;
- necesidad de priorizar claridad metodológica sobre exceso de features;
- riesgo de explosión combinatoria en el espacio experimental;
- necesidad de separar explícitamente lo que puede vivir en un repo público de lo que debe quedarse local/privado.

---

## 17. Checklist de aceptación final

El proyecto estará “listo” si:

- se puede reproducir el flujo principal sin depender de notebooks operativas;
- el split temporal y el drift están documentados y justificados;
- no hay leakage metodológico no controlado;
- las comparaciones usan un diseño claro y justo;
- hay resultados agregados por seed;
- existe un análisis estadístico defendible;
- `test` se usa solo para confirmación final;
- el repo cuenta una historia clara:

`shift → metodología → comparación → evidencia → decisión final`
