# F7 Artifact Persistence and Interpretability

## Proposito y estatus

Estado actual:

- el bloque `9` de política de persistencia ya queda cerrado en:
  - [config/f7_artifact_persistence_contract_v1.yaml](../config/f7_artifact_persistence_contract_v1.yaml)
  - [docs/f7_block_09_artifact_persistence_rationale.md](f7_block_09_artifact_persistence_rationale.md)
- el bloque `10` de interpretabilidad mínima de `MLP` ya queda cerrado en:
  - [config/f7_mlp_interpretability_contract_v1.yaml](../config/f7_mlp_interpretability_contract_v1.yaml)
  - [docs/f7_block_10_mlp_interpretability_rationale.md](f7_block_10_mlp_interpretability_rationale.md)
- el bloque `11` de interpretabilidad mínima de `XGBoost` ya queda cerrado en:
  - [config/f7_xgb_interpretability_contract_v1.yaml](../config/f7_xgb_interpretability_contract_v1.yaml)
  - [docs/f7_block_11_xgb_interpretability_rationale.md](f7_block_11_xgb_interpretability_rationale.md)
- este documento sigue siendo la referencia amplia de contexto y decisiones conectadas;
- la política de persistencia y las capas mínimas de interpretabilidad por familia ya quedan cerradas a nivel contractual.

Este documento fija que debemos guardar, que puede ser recomputable, y que capas de interpretabilidad y comparabilidad de métricas faltan antes de considerar lista la campaña `F7`.

No es:

- el contrato final de outputs por run;
- el diseño final del reporte de insights del TFG;
- el detalle fino del análisis estadístico posterior de interpretabilidad entre familias.

Si es:

- una guía de trabajo inmediato para no lanzar la campaña sin política de artefactos;
- una separación explícita entre lo que depende del modelo entrenado y lo que se puede recomputar;
- una referencia para asegurar que la comparabilidad entre runs se hace en el espacio correcto, especialmente cuando cambia `y_transform`.

Documentos relacionados:

- [docs/f7_run_plan_17400.md](f7_run_plan_17400.md)
- [docs/f7_run_plan_17400_rationale.md](f7_run_plan_17400_rationale.md)
- [docs/f7_campaign_readiness_checklist.md](f7_campaign_readiness_checklist.md)
- [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md)

## Principio general

Una campaña de `17400` runs no puede depender de dos malas prácticas:

1. guardar demasiado y colapsar `outputs/` sin criterio;
2. guardar demasiado poco y luego no poder reconstruir ni defender lo importante.

La política correcta debe separar:

- artefactos que hay que persistir sí o sí;
- artefactos que solo hace falta persistir para shortlist / finalistas / anchors;
- artefactos que se pueden recomputar a partir de manifests y resultados canónicos.

Además, para `F7` queda congelado que:

- la superficie analítica sustantiva por run se conserva completa;
- la optimización de disco se hace eliminando solo duplicados físicos triviales;
- las nuevas runs canónicas `F7` usan una sola copia estable de:
  - `results.yaml`
  - `run_manifest.json`
  - `metrics_long.csv`
  - `config.yaml`
- la compatibilidad histórica se resuelve en lectores tolerantes, no emitiendo aliases duplicados en la escritura nueva.

Además, la campaña debe asegurar que las métricas comparativas se calculan en el espacio correcto:

- si distintas runs usan distintos `y_transform`, no son directamente comparables en el espacio transformado;
- para comparación real de performance, hay que volver al espacio `raw` del target antes de calcular las métricas principales.

---

## 1. Capas de artefactos por run

## 1.1. Artefactos que siempre deben guardarse

Estas piezas deben persistirse para toda run, aunque luego no se conserven los pesos del modelo.

### A. Identidad y contexto

- `run_manifest.json`
- config snapshot
- referencias al dataset consumido
- `dataset_level_axes`
- `run_level_axes`
- `seed`
- `seed_set_id`
- `comparison_group_id`
- `campaign_id` cuando exista formalmente
- hashes o fingerprints de config relevantes

### B. Resultados canónicos

- `results.yaml`
- `metrics_long.csv`

### C. Interpretabilidad completa por run

Para `F7`, la interpretabilidad ya no se trata como una capa opcional solo para shortlist.

La política vigente es:

- persistir `interpretability_summary.json` por run;
- persistir las tablas globales y por clase por run;
- persistir también las superficies familiares específicas implementadas:
  - `MLP`: bridge/input y, cuando aplique, latente/proyección;
  - `XGBoost`: SHAP y perturbación.

Esto se mantiene porque el análisis posterior entre seeds, por clase y entre familias necesita esa superficie completa.

## 1.2. Artefactos que no deberían ser obligatorios para todas las runs

Estas piezas son útiles, pero no necesariamente deben guardarse para toda la campaña completa.

### A. Pesos del modelo

- checkpoints completos
- archivos de weights finales

### B. Duplicados físicos triviales

Lo que sí debe evitarse en nuevas runs `F7` es guardar dos copias del mismo contenido cuando el directorio de run ya codifica la identidad.

Ejemplos que ya no deben duplicarse en nuevas runs `F7`:

- `results.yaml` y `<run_id>_results.yaml`
- `run_manifest.json` y `<run_id>_run_manifest.json`
- `metrics_long.csv` y `<run_id>_metrics_long.csv`
- `config.yaml` y `<run_id>.yaml`

La eliminación de estos aliases no reduce la reproducibilidad ni la capacidad del análisis posterior.

## 1.3. Artefactos que deben poder recomputarse

Estas piezas no necesitan persistirse como “verdad primaria” por run.

- agregaciones por seeds
- rankings por familia
- tablas comparativas
- ANOVA
- post hoc
- reportes globales de campaña
- resúmenes por `synthetic_policy`
- resúmenes por `x_base`
- resúmenes por `y_transform`

La fuente primaria para recomputarlas debe ser:

- manifests canónicos
- `results.yaml`
- `metrics_long.csv`

---

## 2. Interpretabilidad para `XGBoost`

## 2.1. Línea recomendada

Para `XGBoost`, la capa natural de interpretabilidad es:

- SHAP

## 2.2. Qué debemos guardar

En el estado final vigente de `F7`, por run se guarda:

- `interpretability_summary.json`
- superficie global de `SHAP`
- superficie por clase de `SHAP`
- superficie global de perturbación
- superficie por clase de perturbación
- top features globales y por clase

## 2.3. Estado actual

La parte de `XGBoost` ya no está pendiente de definición metodológica en `F7`.

Lo que queda para más adelante no es cerrar la persistencia base, sino explotar esta superficie en la capa analítica posterior.

---

## 3. Interpretabilidad para `MLP`

## 3.1. Necesidad

`MLP` no puede quedarse sin capa de interpretabilidad formal si queremos:

- enlazar insights con `FlowPre`;
- construir un mini reporte final de cómo afectan las variables a la predicción;
- defender mejor qué patrones está usando el modelo.

## 3.2. Punto de partida real

La capa de interpretabilidad de `MLP` para `F7` ya quedó cerrada y persistida por run.

## 3.3. Qué produce la capa canónica

La capa canónica de `MLP` produce, como mínimo:

- `feature_influence_global`
- `feature_influence_per_class`
- top features globales
- top features por clase
- resumen persistible por run

## 3.4. Qué formato tiene

La persistencia mínima canónica incluye:

- `interpretability_summary.json`
- `top_features_global.csv`
- `top_features_per_class.csv`
- `feature_influence_global.csv`
- `feature_influence_per_class.csv`
- `input_feature_influence_global.csv`
- `input_feature_influence_per_class.csv`

Y, cuando aplica, también:

- `latent_feature_influence_global.csv`
- `latent_feature_influence_per_class.csv`
- manifest/cache de proyección `FlowPre`

## 3.5. Relación con `FlowPre`

La interpretabilidad de `MLP` no debe diseñarse aislada.

Debe ser enlazable con los outputs de `FlowPre` para poder construir después un relato tipo:

- qué variables parecen estructuralmente importantes;
- qué variables mueven más la predicción final en `MLP`;
- si hay coherencia o tensión entre la lectura transformacional de `FlowPre` y la lectura predictiva de `MLP`.

## 3.6. Qué falta por hacer

- formalizar la capa de interpretabilidad mínima de `MLP`;
- decidir qué parte se calcula para todas las runs y qué parte solo para shortlist/finalistas;
- integrar esa salida en la política de persistencia canónica.

---

## 4. Comparabilidad de métricas en espacio `raw`

## 4.1. Problema

Si dos runs usan distintos `y_transform`, no es metodológicamente limpio comparar sus métricas principales en el espacio transformado.

Ejemplos:

- `y = standard`
- `y = robust`
- `y = quantile`
- `y = minmax`

Cada una define una geometría distinta del target.

## 4.2. Regla metodológica

Para la comparación principal de performance:

- el modelo puede entrenarse en el espacio transformado;
- pero las predicciones deben volver al espacio `raw`;
- y las métricas comparativas principales deben calcularse en espacio `raw`.

## 4.3. Por qué esto es importante

Porque si no:

- distintas runs serían comparadas en escalas no equivalentes;
- parte de la aparente mejora podría venir de la representación de `y`, no del modelo/dataset/policy;
- el análisis downstream perdería comparabilidad real.

## 4.4. Qué debemos verificar

Hace falta comprobar que:

- existe una ruta canónica de inversión al espacio `raw` para `MLP`;
- las métricas principales que se usen para selección/comparación se calculan realmente en ese espacio;
- esa lógica es consistente para todas las variantes de `y_transform`.

## 4.5. Qué no se debe asumir

No basta con que el repo guarde algunas métricas `native` o algunas métricas `raw_real` en ciertos casos.

Hay que verificar explícitamente:

- qué se calcula siempre;
- qué se calcula solo a veces;
- qué valor se usa como métrica principal de selección;
- y si la capa canónica garantiza comparabilidad homogénea para toda la campaña.

## 4.6. Implicación práctica

Antes de lanzar la campaña completa, necesitamos confirmar o reforzar que:

- `MLP` produce métricas comparables en espacio `raw`;
- `XGBoost`, si aplica target transform o alguna representación equivalente, también se reporta de forma coherente en ese mismo espacio de comparación;
- la capa de resultados deja explícito el `value_space` de cada métrica y cuál es el principal.

---

## 5. Qué hacer a corto plazo

## 5.1. Definir política de persistencia por run

Hace falta cerrar explícitamente:

### Guardar siempre

- `run_manifest.json`
- config snapshot
- `results.yaml`
- `metrics_long.csv`
- interpretabilidad ligera

### Guardar solo para shortlist/finalistas/anchors

- weights
- SHAP detallado
- interpretabilidad pesada

### Recomputable

- agregaciones por seeds
- rankings
- tablas de campaña
- ANOVA
- post hoc

## 5.2. Construir capa mínima de interpretabilidad `MLP`

Hace falta:

- definir outputs mínimos;
- decidir qué se persiste siempre;
- integrar esa capa en el flujo canónico de resultados.

## 5.3. Definir capa mínima de interpretabilidad `XGBoost`

Hace falta:

- decidir salida ligera basada en SHAP;
- decidir qué artefactos se guardan siempre y cuáles solo para shortlist/finalistas;
- integrar esa capa en la política de persistencia.

## 5.4. Verificar comparabilidad en `raw space`

Hace falta:

- revisar la función o funciones que calculan métricas para `MLP`;
- comprobar si la inversión a espacio `raw` ocurre siempre que debe ocurrir;
- confirmar que la métrica principal de comparación usa ese espacio;
- dejarlo documentado y trazado.

## 5.5. Reflejar esta política en la capa meta y de campaña

Hace falta que la campaign spec o el contrato operativo futuro puedan declarar:

- qué artefactos persistir por defecto;
- qué interpretabilidad mínima es obligatoria;
- qué artefactos pesados se reservan a shortlist/finalistas.

---

## 6. Criterios de “suficientemente completo”

## 6.1. Persistencia

Este frente estará suficientemente completo cuando:

- exista una política explícita de qué guardar siempre y qué no;
- esa política esté integrada en los runners y/o el contrato de campaña;
- no dependamos de guardar todos los weights para poder defender la campaña.

## 6.2. Interpretabilidad

Este frente estará suficientemente completo cuando:

- `MLP` tenga una capa mínima de interpretabilidad de campaña;
- `XGBoost` tenga una capa mínima de SHAP o equivalente;
- ambas puedan alimentar un reporte de insights posterior sin requerir rehacer toda la campaña.

## 6.3. Comparabilidad de métricas

Este frente estará suficientemente completo cuando:

- la campaña pueda demostrar que las métricas principales son comparables entre distintos `y_transform`;
- el espacio de cálculo principal quede claro;
- y la capa de resultados lo refleje de forma explícita.

---

## 7. Qué no decide este documento

Este documento no fija todavía:

- la métrica principal exacta de selección;
- el formato final del reporte de insights;
- el detalle exacto de la interpretabilidad pesada;
- si se guardarán todos los weights de shortlist o también algunos adicionales;
- la implementación concreta de la capa de interpretabilidad `MLP`.

Lo que sí deja claro es que todas esas piezas deben resolverse antes de considerar madura la política de artefactos, interpretabilidad y comparabilidad de `F7`.
