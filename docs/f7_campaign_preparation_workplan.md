# F7 Campaign Preparation Workplan

## Proposito y estatus

Este documento organiza, de principio a fin, todo lo que hoy sabemos que falta o conviene cerrar antes de lanzar la campaña `F7`.

No es:

- una checklist minima;
- un contrato final ya congelado;
- una decision cerrada de todos los puntos.

Si es:

- una hoja de ruta completa y ordenada;
- una guia de trabajo para ir resolviendo cada bloque sin perder el hilo;
- una lista de preguntas que debemos contestar antes de implementar cada parte, sin asumir decisiones que todavia no se hayan tomado.

La idea es separar dos usos distintos:

- [docs/f7_campaign_readiness_checklist.md](f7_campaign_readiness_checklist.md): para marcar bloqueantes y readiness;
- este documento: para ordenar el trabajo y explicitar las preguntas que deben responderse en cada paso.

Documentos relacionados:

- [docs/f7_run_plan_17400.md](f7_run_plan_17400.md)
- [docs/f7_run_plan_17400_rationale.md](f7_run_plan_17400_rationale.md)
- [docs/f7_meta_and_stats_readiness.md](f7_meta_and_stats_readiness.md)
- [docs/f7_artifact_persistence_and_interpretability.md](f7_artifact_persistence_and_interpretability.md)
- [docs/f7_campaign_readiness_checklist.md](f7_campaign_readiness_checklist.md)

## Decisiones ya fijadas que este documento no reabre por defecto

Salvo que decidamos cambiarlas explícitamente más adelante, este documento parte de estas decisiones ya tomadas:

- campaña objetivo de `17400` runs;
- `MLP` como familia principal;
- `XGBoost` como baseline acotada;
- `96` datasets `MLP`;
- `4` datasets `XGBoost`;
- `30` seeds por run;
- cap sintético del `50%` del tamaño real de cada clase minoritaria;
- dos familias `batch/cycling` en `MLP`:
  - `plain = baseline + no cycling`
  - `imbalance_aware = balanced + cycling`
- tres `loss_policy` en `MLP`:
  - `overall + rmse`
  - `per_class_equal + rmse`
  - `per_class_equal + rrmse`
- `XGBoost` no barre escalados ni abre un sub-grid propio de training losses.

Este documento sí puede formular preguntas sobre cómo implementar o formalizar esas decisiones, pero no las considera abiertas salvo que las volvamos a poner en discusión explícitamente.

## Como leer este documento

Cada bloque se organiza asi:

- `objetivo`: que queremos dejar resuelto;
- `por que va en este punto del orden`: por que conviene hacerlo aqui y no mas tarde;
- `entregables esperados`: que deberia existir al cerrar el bloque;
- `preguntas a responder contigo`: preguntas abiertas o de confirmacion que debemos contestar antes de implementar;
- `riesgo si se deja ambiguo`: que puede salir mal si no se cierra bien.

Importante:

- este documento no decide por su cuenta cosas que todavia no se han fijado;
- cuando una decision sigue abierta, aqui solo se formula la pregunta y el criterio para resolverla.

## Como usarlo de forma operativa

La forma recomendada de trabajar este documento es secuencial:

1. elegir un bloque;
2. responder las preguntas abiertas de ese bloque;
3. decidir qué se congela y qué sigue abierto;
4. implementar solo lo necesario para cerrar ese bloque;
5. validar el resultado;
6. actualizar la checklist y, si hace falta, este mismo documento.

Para que el trabajo no se nos quede a medias, cada bloque debería acabar dejando constancia de cinco cosas:

- `decision tomada o decision aplazada`:
  - qué se ha cerrado;
  - qué se ha dejado deliberadamente abierto;
- `artefacto o salida creada`:
  - config
  - doc
  - runner
  - manifest
  - validación
- `ficheros o superficies tocadas`;
- `smoke test o validación realizada`;
- `impacto sobre readiness`:
  - qué punto de la checklist queda ya cubierto o más cerca de quedar cubierto.

Si un bloque no deja estas cinco piezas razonablemente claras, conviene asumir que todavía no está realmente cerrado.

## Interpretacion operativa del orden

El orden de bloques de este documento es **la guia principal de trabajo**, pero debe leerse con una precision importante:

- sí expresa una prioridad y una secuencia recomendada;
- no significa que todos los bloques formen una cadena estrictamente lineal sin ningun acoplamiento;
- algunos bloques son prerequisitos limpios;
- otros pueden trabajarse en paralelo;
- y unos pocos conviene tratarlos como paquetes o en dos fases para evitar retrodependencias practicas.

Regla operativa:

- un bloque anterior no deberia requerir una decision sustantiva todavia abierta en un bloque posterior;
- si detectamos ese caso, no debemos “forzar” el cierre artificial del bloque anterior;
- debemos dejarlo explicitamente marcado como:
  - cierre parcial;
  - bloque acoplado;
  - o bloque dividido en subfases.

### Tipos de relacion entre bloques

- `strict_prerequisite`: conviene cerrarlo antes de pasar al siguiente porque fija contrato real.
- `parallelizable`: puede avanzarse en paralelo con otro bloque sin riesgo metodologico fuerte.
- `coupled`: el bloque tiene una dependencia practica compartida con otro y conviene tratarlos como paquete.
- `split_phase`: el bloque esta bien conceptualmente pero ejecutarlo bien requiere dividirlo en:
  - fase de definicion
  - fase de materializacion / freeze

### Lectura recomendada para `F7`

#### Bloques que si actuan como prerequisito fuerte

- `0. Reconfirmar el alcance de la campaña`
- `6. Fijar el panel de 30 seeds`
- `7. Cerrar la gramatica meta de campaña`
- `12. Formalizar campaign spec`
- `13. Construir runners de campaña`
- `15. Hacer benchmark real de coste`
- `16. Hacer preflight final de lanzamiento`

#### Bloques que pueden avanzarse en paralelo

- `1. Congelar el baseline de MLP`
- `2. Congelar el baseline de XGBoost`
- `3. Resolver device y runtime de MLP`

Nota:

- `1`, `2` y `3` deben leerse como tres frentes tempranos del mismo nivel practico;
- no hace falta esperar a cerrar uno de ellos para empezar a trabajar en los otros dos.

#### Bloques acoplados o que conviene tratar como paquete

- `4. Definir y materializar el inventario de datasets`
- `5. Alinear la politica de masa sintetica`

Regla:

- `4` no debe cerrarse de una sola vez si la materializacion final depende todavia de decisiones operativas de `5`;
- en la practica, `4` debe dividirse asi:
  - `4A`: definir inventario logico y naming
  - `5`: cerrar cap sintetico y regla de aceptacion
  - `4B`: materializar, validar y congelar inventario final

Tambien deben tratarse como paquete:

- `9. Definir politica de persistencia e interpretabilidad`
- `10. Diseñar la capa minima de interpretabilidad para MLP`
- `11. Diseñar la capa minima de interpretabilidad para XGBoost`

Regla:

- `9` fija la politica general;
- `10` y `11` concretan su realizacion por familia;
- si falta detalle real en `10` o `11`, `9` no debe declararse totalmente congelado.

Hay un acoplamiento menor adicional entre:

- `12. Formalizar campaign spec`
- `14. Congelar la gramatica del analisis principal`

Regla:

- `12` debe cerrarse con una spec ejecutable y coherente;
- `14` debe congelar la lectura estadistica principal antes del lanzamiento completo;
- si `14` obliga a ajustar ids, metricas o familias estadisticas, ese ajuste debe hacerse de forma controlada sobre `12`, no mediante convenciones informales en runners.

### Orden operativo recomendado a seguir a partir de ahora

Para evitar pisarnos mas adelante, la secuencia practica recomendada es esta:

1. `0` cerrar alcance y supuestos base.
2. avanzar `1`, `2` y `3` como bloque temprano coordinado;
3. hacer `4A` inventario logico de datasets;
4. hacer `5` politica de masa sintetica;
5. volver a `4B` para materializacion y freeze del inventario final;
6. hacer `6`, `7` y `8`;
7. tratar `9`, `10` y `11` como paquete;
8. hacer `12`;
9. hacer `13`;
10. hacer `14`;
11. hacer `15`;
12. hacer `16`.

### Regla para continuar en futuras sesiones

Cuando retomemos con prompts del tipo:

- “que toca ahora”
- “sigue con el siguiente bloque”
- “vamos con el siguiente punto”

la referencia correcta no debe ser solo el numero siguiente del documento, sino:

- el siguiente bloque no cerrado en esta secuencia operativa;
- respetando las reglas de acoplamiento y split-phase de esta seccion.

En particular:

- no debe asumirse que `4` queda totalmente cerrado antes de `5`;
- no debe asumirse que `9` queda totalmente congelado antes de bajar a `10` y `11`;
- no debe tratarse `12` como contrato estadistico final si `14` todavia no se ha congelado.

---

## 0. Reconfirmar el alcance de la campaña

### Estado

- cerrado el 2026-05-16

### Rationale canónico

- [docs/f7_block_00_scope_rationale.md](f7_block_00_scope_rationale.md)

### Decision tomada

Queda reconfirmado como alcance vigente de `F7`:

- campaña completa objetivo de `17400` runs;
- `MLP` como familia principal;
- `XGBoost` como baseline acotada;
- `96` datasets `MLP`;
- `4` datasets `XGBoost`;
- `30` seeds como parte del diseño base;
- cap sintetico del `50%` aplicable por igual a:
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`

Para este bloque no se reabre:

- convertir `XGBoost` en un segundo subestudio con grid propio;
- tratar las `30` seeds como ampliacion opcional posterior;
- reinterpretar el cap del `50%` como policy desigual entre ramas sinteticas.

### Decision aplazada deliberadamente

- la definicion operativa de un eventual piloto previo queda fuera de este bloque;
- en este punto solo se fija la campaña completa objetivo y sus supuestos base.

### Impacto sobre los bloques siguientes

Los bloques posteriores deben heredar este perimetro sin reabrirlo salvo decision explicita nueva.

En particular:

- `1`, `2` y `3` deben trabajar ya contra la campaña completa objetivo;
- `4` y `5` deben construir inventario y datasets bajo el cap sintetico ya fijado;
- `6` debe fijar un panel real de `30` seeds, no una version reducida;
- `12` y `13` deben tratar `F7` como campaña de `17400` runs.

### Objetivo

Dejar fijado, antes de tocar implementacion adicional, el alcance actual de `F7`:

- `MLP` como familia principal;
- `XGBoost` como baseline acotada;
- `96` datasets `MLP`;
- `4` datasets `XGBoost`;
- `30` seeds;
- cap sintetico del `50%`.

### Por que va primero

Todo lo demas depende de este perimetro:

- datasets a materializar;
- ids de campaña;
- runners;
- footprint;
- tiempos.

### Entregables esperados

- referencia explicita al plan vigente;
- confirmacion de que no estamos reabriendo el perimetro a mitad de la preparacion;
- lista de supuestos base que todos los bloques posteriores heredan.

### Preguntas a responder contigo

- ¿Confirmamos que la campaña objetivo sigue siendo la de `17400` runs y no una fase piloto reducida?
- ¿Confirmamos que `XGBoost` sigue entrando solo como baseline acotada y no como segundo subestudio?
- ¿Confirmamos que el cap sintetico del `50%` aplica a las tres policies sinteticas?
- ¿Confirmamos que las `30` seeds son parte del diseño base y no una ampliacion posterior?
- ¿Queremos distinguir ya entre “campaña completa objetivo” y “bloque piloto previo”, o de momento solo dejar descrita la campaña completa?

### Riesgo si se deja ambiguo

- rediseñar varias veces datasets, runners e ids;
- mezclar decisiones de piloto y campaña final;
- inflar el trabajo tecnico sin haber cerrado el alcance real.

---

## 1. Congelar el baseline de `MLP`

### Estado

- cerrado

### Rationale canónico

- [docs/f7_block_01_mlp_baseline_rationale.md](f7_block_01_mlp_baseline_rationale.md)

### Decision ya tomada dentro de este bloque

Para `F7`, [config/mlp_closure_base_v1.yaml](../config/mlp_closure_base_v1.yaml) se toma como:

- prior estructural fuerte;
- pero no como config literal final de campaña.

Tambien queda fijado que el baseline comun de este bloque debe congelar solo la parte compartida por toda la campaña:

- arquitectura;
- optimizer;
- scheduler;
- `learning_rate`;
- `batch_size`;
- `num_epochs`;
- early stopping.

Y no debe congelar dentro del baseline de campaña:

- panel de seeds;
- `loss_policy`;
- regimen `batch/cycling`.

### Decision ya tomada sobre la revalidacion

La revalidacion del baseline de `MLP` queda partida en dos pasos ya definidos:

- `v1`, ya ejecutada:
  - sirvio para fijar `batch_size = 198`;
  - sirvio para fijar `num_epochs = 300`;
  - dejo trazada la comparacion inicial de `loss` y `policy`.
- `v2`, ya ejecutada:
  - confirmo que la zona fuerte estaba bastante por encima del baseline pequeño;
  - dejo `512x6` como candidata estructural principal;
  - y `384x4` como candidata fuerte mas barata.
- `v3`, ya ejecutada y cerrada:
  - mantiene `batch_size = 198`;
  - mantiene `num_epochs = 300`;
  - explora la frontera final de capacidad:
    - `256`, `320`, `384`, `448`, `512`
    - con `3`, `4`, `5` y `6` capas;
  - comprueba explicitamente si `512x6` generaliza bien bajo cambios de `loss` y `policy`;
  - introduce el criterio practico de coste:
    - target alrededor de `5 s` de media por run.

### Cierre del bloque

Queda congelado como baseline estructural final de `MLP` para `F7`:

- `hidden_dim = 320`
- `num_layers = 4`
- `embedding_dim = 12`
- `batch_size = 198`
- `learning_rate = 1e-4`
- `num_epochs = 300`
- `optimizer = adam`
- `lr_scheduler = plateau`
- `early_stopping_patience = 20`

Queda documentado en:

- [config/f7_mlp_base_v1.yaml](../config/f7_mlp_base_v1.yaml)
- [docs/f7_mlp_baseline_final_v1.md](f7_mlp_baseline_final_v1.md)

El rationale metodologico completo del cierre de este bloque queda concentrado en:

- [docs/f7_mlp_baseline_final_v1.md](f7_mlp_baseline_final_v1.md)

Y queda explicitamente fuera del baseline estructural:

- panel de seeds;
- `loss_reduction`;
- `regression_group_metric`;
- `dataloader_mode`;
- `cycle_reals`.

Artefactos de apoyo:

- `v1` historica:
  - [config/f7_mlp_baseline_revalidation_v1.yaml](../config/f7_mlp_baseline_revalidation_v1.yaml)
  - [docs/f7_mlp_baseline_revalidation_v1.md](f7_mlp_baseline_revalidation_v1.md)
- `v2` activa:
  - [config/f7_mlp_baseline_revalidation_v2.yaml](../config/f7_mlp_baseline_revalidation_v2.yaml)
  - [docs/f7_mlp_baseline_revalidation_v2.md](f7_mlp_baseline_revalidation_v2.md)
- `v3` activa final:
  - [config/f7_mlp_baseline_revalidation_v3.yaml](../config/f7_mlp_baseline_revalidation_v3.yaml)
  - [docs/f7_mlp_baseline_revalidation_v3.md](f7_mlp_baseline_revalidation_v3.md)

### Objetivo

Dejar fijada la arquitectura y los hiperparametros baseline de `MLP` que gobernaran toda la campaña.

### Por que va aqui

Antes de runners, comparabilidad o timing, hay que saber que modelo estamos llamando “baseline de campaña”.

### Entregables esperados

- config baseline de `MLP` congelada;
- `config_id` o `base_config_id` canonico de campaña;
- trazabilidad clara de:
  - arquitectura
  - optimizer
  - scheduler
  - epochs
  - early stopping
  - batching base

### Riesgo si se deja ambiguo

- comparar datasets con modelos no equivalentes;
- reabrir tuning del modelo en mitad de la campaña;
- invalidar comparaciones agregadas por mezclar distintos baselines bajo el mismo nombre.

---

## 2. Congelar el baseline de `XGBoost`

### Estado

- cerrado el 2026-05-16

### Rationale canónico

- [docs/f7_block_02_xgb_baseline_rationale.md](f7_block_02_xgb_baseline_rationale.md)

### Objetivo

Definir una sola configuración de entrenamiento de `XGBoost` para toda la campaña.

### Por que va aqui

`XGBoost` entra como baseline sobria. Si no se congela pronto, se convierte en otro subestudio.

### Entregables esperados

- `model_config_id` de `XGBoost`;
- hiperparametros cerrados del booster;
- `objective` y `eval_metric` de entrenamiento fijados;
- política de features clara.

### Revalidacion de apoyo cerrada

Para cerrar este bloque se define una mini-revalidacion acotada de `XGBoost`:

- `20` configuraciones intencionales;
- `3` seeds;
- dataset oficial raw/no-scale;
- representacion fija:
  - numericas raw
  - `type` en one-hot
  - sin `post_cleaning_index`;
- sin abrir escalados, training losses ni batching/cycling;
- smoke test de SHAP sobre las `2-3` mejores cfgs.

Artefactos:

- [config/f7_xgb_baseline_revalidation_v1.yaml](../config/f7_xgb_baseline_revalidation_v1.yaml)
- [docs/f7_xgb_baseline_revalidation_v1.md](f7_xgb_baseline_revalidation_v1.md)

Y, tras ejecutar `v1`, se activa una micro-revalidacion final de cierre:

- [config/f7_xgb_baseline_revalidation_v2.yaml](../config/f7_xgb_baseline_revalidation_v2.yaml)
- [docs/f7_xgb_baseline_revalidation_v2.md](f7_xgb_baseline_revalidation_v2.md)

Tras ejecutar `v2`, el baseline final de `XGBoost` queda congelado en:

- [config/f7_xgb_base_v1.yaml](../config/f7_xgb_base_v1.yaml)
- [docs/f7_xgb_baseline_final_v1.md](f7_xgb_baseline_final_v1.md)

Decision final del bloque:

- representación fija:
  - dataset oficial raw/no-scale
  - `feature_policy = raw_numeric_plus_type_onehot`
- configuración final del booster:
  - `objective = reg:squarederror`
  - `eval_metric = rmse`
  - `n_estimators = 1200`
  - `learning_rate = 0.035`
  - `max_depth = 4`
  - `min_child_weight = 12`
  - `subsample = 0.85`
  - `colsample_bytree = 0.80`
  - `reg_alpha = 0.02`
  - `reg_lambda = 1.50`
  - `gamma = 0.0`
  - `tree_method = hist`
  - `max_bin = 256`
  - `early_stopping_rounds = 60`

Ademas, el bloque deja fijado que la capa SHAP de campaña para `XGBoost` no puede ser minima ni solo decorativa:

- cuando `XGBoost` entre en la parrilla de `F7`, la persistencia SHAP debe ser lo mas completa posible dentro de un coste razonable;
- debe permitir guardar al menos contribuciones por muestra y por feature, valores firmados, agregados absolutos, `expected_value`, nombres de features y material suficiente para analisis agregados posteriores.

### Riesgo si se deja ambiguo

- convertir la baseline en un segundo estudio de tuning;
- perder claridad sobre qué se está comparando frente a `MLP`.

---

## 3. Resolver device y runtime de `MLP`

### Estado

- cerrado el 2026-05-16

### Rationale canónico

- [docs/f7_block_03_mlp_runtime_rationale.md](f7_block_03_mlp_runtime_rationale.md)

### Objetivo

Dejar `train_mlp` listo para usar el device correcto de forma canónica, reproducible y explícita para `F7`.

### Por que va aqui

Antes de pensar en `17400` runs, hay que asegurar que el camino de ejecución real usa el device correcto.

### Entregables esperados

- soporte canónico de selección de device en `train_mlp`;
- criterio explícito de resolución de device;
- benchmark real de `cpu` frente a `mps` en esta máquina;
- decisión explícita de política de device para `F7`.

### Cierre del bloque

Queda fijado que:

- `train_mlp.py` ya resuelve el device mediante la utilidad canónica;
- el device efectivo se persiste en los artefactos de run;
- `mps` sigue disponible como opción explícita de exploración local;
- pero la campaña `F7` de `MLP` debe forzar `cpu` de forma explícita;
- no debe depender de `auto`.

La decisión se apoya en benchmark real ya ejecutado durante la revalidación de baseline:

- en esta máquina, para este tipo de runs, `cpu` resultó más rápida que `mps`;
- por tanto, `cpu` es a la vez la opción más simple y la más eficiente para la campaña.

### Riesgo si se deja ambiguo

- correr con un backend no deseado;
- estimar mal el tiempo de campaña;
- mezclar runs con distinto backend sin dejarlo trazado.

---

## 4. Definir y materializar el inventario de datasets

### Estado

- fase `4A` cerrada el 2026-05-17
- fase `4B` cerrada el 2026-05-18

### Rationale canónico

- [docs/f7_block_04_dataset_inventory_rationale.md](f7_block_04_dataset_inventory_rationale.md)

### Objetivo

Pasar del plan conceptual de datasets a un inventario materializable, verificable y congelable.

### Por que va aqui

Sin datasets concretos no existe campaña ejecutable ni `campaign spec` real.

### Entregables esperados

- definición explícita del inventario `MLP`:
  - `96` datasets
- definición explícita del inventario `XGBoost`:
  - `4` datasets
- política de materialización:
  - escalados
  - sintéticos
  - manifests
  - freeze posterior

### Cierre de la fase `4A`

Queda fijado el inventario lógico y machine-readable de campaña en:

- [config/f7_dataset_inventory_v1.yaml](../config/f7_dataset_inventory_v1.yaml)
- [config/f7_dataset_inventory_v1.csv](../config/f7_dataset_inventory_v1.csv)

Decisiones cerradas:

- `MLP` usa exactamente `96` datasets:
  - `6 x_base`
  - `4 y_transform`
  - `4 synthetic_policy`
- `XGBoost` usa exactamente `4` datasets:
  - base raw/no-scale fija
  - `4 synthetic_policy`
- la materialización final será:
  - por adelantado
  - después del bloque `5`
  - y con freeze posterior
- si cambia una receta validada:
  - no se toca el inventario existente
  - se crea una versión nueva
- las `synthetic_policy` de `F7` se tratan como datasets fijos, no como realizaciones distintas por seed.

### Cierre de la fase `4B`

Queda materializado y congelado el inventario final de `F7` en el árbol:

- `data/sets/official/init_temporal_processed_v1/` (`local-only`, no subible)

Distribución canónica:

- `raw/`:
  - bundle modeled-raw oficial base
- `scaled/`:
  - `24` datasets base no sintéticos de `MLP`
- `synthetic_pools/`:
  - `1` pool compartido `flowgen_official`
  - `1` pool compartido `flowgen_train_only`
- `augmented_scaled/`:
  - `24` datasets `MLP + kmeans_smote`
  - `24` datasets `MLP + flowgen_official`
  - `24` datasets `MLP + flowgen_train_only`
  - `3` datasets aumentados de `XGBoost`
- `xgboost/`:
  - `1` dataset base raw/no-scale de `XGBoost`
- `meta/`:
  - copia estable del inventario materializado y del batch final canónico
- `legacy_pre_f7/`:
  - artefactos históricos o de smoke que no deben usarse en campaña

El batch final válido es:

- `outputs/reports/f7_dataset_materialization/f7_dataset_materialization_20260518T132351821306Z/` (`local-only`)

Resumen del batch:

- `96` datasets `MLP`
- `4` datasets `XGBoost`
- `2` pools compartidos `FlowGen`
- `102` artefactos `ok`
- `0` fallos

Evidencia principal:

- `materialized_inventory.csv`
- `phase_summary.csv`
- `materialization_batch_manifest.json`

Nota operativa:

- el árbol local separa ya:
  - material canónico de campaña
  - material `legacy_pre_f7`
- la referencia local de consumo para campaña queda además copiada en:
  - `data/sets/official/init_temporal_processed_v1/meta/f7_canonical_materialized_inventory_v1.csv`
  - `data/sets/official/init_temporal_processed_v1/F7_CANONICAL_LAYOUT.md`

### Riesgo si se deja ambiguo

- materializar datasets con reglas distintas a mitad de campaña;
- perder comparabilidad por no congelar manifests;
- lanzar runners contra inventarios implícitos.

---

## 5. Alinear la política de masa sintética

### Estado

- cerrado el 2026-05-17

### Rationale canónico

- [docs/f7_block_05_synthetic_mass_cap_rationale.md](f7_block_05_synthetic_mass_cap_rationale.md)

### Objetivo

Garantizar que las tres `synthetic_policy` queden alineadas bajo el cap del `50%`.

### Por que va aqui

La cantidad de sintéticos afecta la interpretación tanto como el algoritmo. Hay que cerrarlo antes de la materialización final.

### Entregables esperados

- regla de cap implementada o al menos operacionalizada para:
  - `flowgen_official`
  - `flowgen_train_only`
  - `kmeans_smote`
- manifests o reports que dejen claro:
  - cuántos sintéticos por clase se añadieron;
  - cuánta masa final quedó por clase.

### Cierre del bloque

Queda fijado que:

- la clase mayoritaria se define usando solo los conteos reales de `train` del split oficial;
- en la clase mayoritaria:
  - `n_synth = 0`
- en cada clase estrictamente minoritaria `c`:
  - `n_synth(c) <= floor(0.5 * n_real(c))`
  - `n_real(c) + n_synth(c) <= max_real_train`
- si hubiera empate en la mayoría:
  - no se añaden sintéticos a ninguna de las clases empatadas arriba
  - solo a las estrictamente menores.

Queda además decidido que:

- la regla es idéntica para `flowgen_official`, `flowgen_train_only` y `kmeans_smote`;
- cualquier versión previa que no siga esta policy se trata como histórica o legacy;
- las versiones `F7` se reconstruyen desde cero;
- los manifests deben guardar `n_real`, `n_synth`, `n_final` y fracción sintética final por clase;
- y la aceptación de un dataset sintético como `campaign-ready` depende de una validación central obligatoria.

Artefactos canónicos:

- [config/f7_synthetic_cap_policy_v1.yaml](../config/f7_synthetic_cap_policy_v1.yaml)
- [data/f7_synthetic_cap_policy.py](../data/f7_synthetic_cap_policy.py)

### Riesgo si se deja ambiguo

- atribuir a la policy lo que en realidad proviene de meter distinta cantidad de sintéticos;
- mezclar datasets sintéticos conceptualmente distintos bajo la misma etiqueta.

---

## 6. Fijar el panel de `30` seeds

### Estado

- cerrado el 2026-05-18

### Rationale canónico

- [docs/f7_block_06_seed_panel_rationale.md](f7_block_06_seed_panel_rationale.md)

### Objetivo

Definir el panel de replicas y su identidad canónica.

### Por que va aqui

El panel de seeds no es un adorno: condiciona comparabilidad, agregación y coste.

### Entregables esperados

- `seed_set_id` de campaña;
- lista exacta de `30` seeds;
- semántica mínima de réplica:
  - índice
  - versión de panel
  - posible rol si hiciera falta
- artefactos canónicos:
  - [config/f7_seed_panel_v1.yaml](../config/f7_seed_panel_v1.yaml)
  - [config/f7_seed_panel_v1.csv](../config/f7_seed_panel_v1.csv)

### Decision tomada

Queda congelado un panel nuevo de campaña:

- `seed_set_id = f7_seed_panel_v1`
- `30` seeds explícitas y machine-readable
- panel único y común para:
  - `MLP`
  - `XGBoost`

La semántica operativa queda fijada así:

- en `MLP`, la seed de campaña gobierna la réplica de entrenamiento y debe bindearse al menos a `python random`, `numpy` y `torch`;
- en `XGBoost`, la seed de campaña gobierna la réplica del booster y debe persistirse como `random_state`;
- en datasets sintéticos `F7`, no rematerializa nada porque los datasets ya están congelados como artefactos fijos.

### Preguntas a responder contigo

- ¿Queremos que el panel de `30` seeds sea completamente nuevo o derivado de un panel previo?
- ¿Quieres que las mismas `30` seeds apliquen por igual a `MLP` y `XGBoost`?
- ¿Quieres que las seeds se fijen de forma manual o generadas por una regla reproducible?
- ¿Quieres versionar el panel desde ya (`seed_panel_version`) aunque solo exista una versión?

### Riesgo si se deja ambiguo

- comparaciones no homogéneas;
- dificultades al agregar por seed;
- reruns con paneles no equivalentes.

---

## 7. Cerrar la gramática meta de campaña

### Estado

- cerrado el 2026-05-18 a nivel de gramática base

### Rationale canónico

- [docs/f7_block_07_meta_grammar_rationale.md](f7_block_07_meta_grammar_rationale.md)

### Objetivo

Dejar lista la identidad canónica de:

- dataset
- run spec
- trial
- campaign
- familia estadística

### Por que va aqui

Sin esta capa, una campaña grande queda con manifests útiles pero todavía demasiado artesanales.

### Entregables esperados

- definición de:
  - `dataset_candidate_id`
  - `run_spec_id`
  - `trial_id`
  - `campaign_id`
  - `comparison_group_id`
  - `statistical_family_id`
- decisión de dónde vive cada campo:
  - config
  - manifest de dataset
  - run manifest
  - tabla de campaign spec
- artefacto canónico:
  - [config/f7_meta_grammar_v1.yaml](../config/f7_meta_grammar_v1.yaml)

### Decisión tomada

Queda fijada una gramática meta base para `F7` con estas decisiones:

- `campaign_id` único para toda la campaña:
  - `f7_campaign_v1`
- `dataset_candidate_id` heredado del inventario materializado canónico de `4B`
- `run_spec_id` estable y sin seed
- `trial_id` derivable y además persistido explícitamente
- `comparison_group_id` como grupo operativo de comparación, con posibilidad de cruces entre familias cuando el contraste sea explícito
- `statistical_family_id` como grupo narrow y analysis-specific

Importante:

- el bloque `7` cierra la gramática base;
- la congelación de familias estadísticas concretas para efectos, combinaciones y sinergias se refina más adelante en el bloque de análisis principal, sin reabrir esta gramática.

### Preguntas a responder contigo

- ¿Quieres ids legibles por humanos, ids semánticos compactos, hashes, o una combinación?
- ¿Qué parte de la identidad quieres que sea visible en el nombre humano del run y qué parte prefieres solo en manifest?
- ¿Quieres que `trial_id` sea derivable a partir de componentes o que se persista explícitamente?
- ¿Quieres que `statistical_family_id` agrupe por:
  - mismo dataset + mismo run spec
  - mismo model family + mismo dataset
  - u otra regla?
- ¿Quieres que `campaign_id` sea único para toda `F7` o distinguir subcampañas dentro de `F7`?

### Riesgo si se deja ambiguo

- agregaciones ad hoc;
- análisis estadístico más frágil;
- mucha dependencia de convenciones implícitas.

---

## 8. Cerrar la comparabilidad de métricas en `raw space`

### Estado

- cerrado el 2026-05-18

### Rationale canónico

- [docs/f7_block_08_raw_metric_comparability_rationale.md](f7_block_08_raw_metric_comparability_rationale.md)

### Objetivo

Asegurar que las métricas principales de campaña sean comparables entre runs con distinto `y_transform`.

### Por que va aqui

Este punto afecta directamente la validez de la comparación downstream.

### Entregables esperados

- verificación explícita del cálculo de métricas en `raw`;
- confirmación de la ruta de inversión desde el espacio transformado;
- claridad sobre:
  - `metric_name`
  - `metric_scope`
  - `value_space`
  - `split_role`
  principales.
- artefacto canónico:
  - [config/f7_raw_metric_contract_v1.yaml](../config/f7_raw_metric_contract_v1.yaml)

### Decisión tomada

Queda fijado que toda run `campaign-valid` de `MLP` o `XGBoost` debe persistir, en `raw space`:

- siempre para `train`
- siempre para `val`
- y para `test` solo cuando la run se declara como `holdout_run` mediante opt-in explícito

y para los scopes:

- `overall`
- `macro`
- `per_class`

al menos estas métricas:

- `r2`
- `mse`
- `rmse`
- `rrmse`
- `mape`

Además:

- las métricas en espacio transformado pueden guardarse;
- pero no gobiernan la comparación principal entre runs;
- y si una run no puede invertir correctamente a `raw` y producir su paquete mínimo obligatorio, no es `campaign-valid`.
- el contrato distingue entre:
  - `selection_run`: exige `train` y `val`
  - `holdout_run`: exige `train`, `val` y `test`

La `anchor metric` por defecto para ranking y lectura rápida queda en:

- `raw_real.macro.rrmse`

sin convertirla en métrica soberana única del análisis final.

### Preguntas a responder contigo

- ¿Qué métricas queremos que sean “principales” a efectos de campaña?
- ¿Queremos que todas las métricas principales se calculen siempre en `raw`, sin excepciones?
- ¿Queremos mantener algunas métricas en espacio transformado solo como diagnósticas?
- ¿Queremos que el runner falle si una run no puede producir la métrica principal en `raw space`?

### Riesgo si se deja ambiguo

- comparar runs en escalas no equivalentes;
- contaminar el efecto de dataset/modelo con efecto de representación del target.

---

## 9. Definir política de persistencia e interpretabilidad

### Estado

- cerrado el 2026-05-18

### Rationale canónico

- [docs/f7_block_09_artifact_persistence_rationale.md](f7_block_09_artifact_persistence_rationale.md)

### Objetivo

Separar claramente:

- lo que siempre se guarda;
- lo que solo se guarda para shortlist/finalistas/anchors;
- lo que puede recomputarse;
- y qué capa mínima de interpretabilidad exigimos en `MLP` y `XGBoost`.

### Por que va aqui

Antes de runners y footprint final, hay que decidir qué artefactos constituyen una run válida y defendible.

### Entregables esperados

- política escrita de persistencia;
- outputs mínimos de run válida;
- definición de interpretabilidad mínima para:
  - `MLP`
  - `XGBoost`
- artefacto canónico:
  - [config/f7_artifact_persistence_contract_v1.yaml](../config/f7_artifact_persistence_contract_v1.yaml)

### Decisión tomada

Queda congelada una política canónica con tres niveles:

- `must_persist_per_run`
- `must_persist_per_shortlist`
- `must_persist_per_finalist`

Y toda run `F7` debe persistir, como mínimo:

- `results.yaml`
- `run_manifest.json`
- `metrics_long.csv`
- snapshot de config
- ids meta
- validación del contrato raw
- `predictions_eval_raw.csv.gz`

Regla de sidecars:

- `val` siempre
- `test` solo en `holdout_run`
- `train` no por defecto

Regla por familia:

- `MLP`: no persiste pesos por defecto en toda run
- `XGBoost`: sí persiste booster por run

La interpretabilidad específica por familia no se implementa aquí; este bloque deja solo la política y los slots contractuales para `10` y `11`.

### Preguntas a responder contigo

- ¿Queremos guardar siempre los weights o solo para shortlist/finalistas?
- ¿Qué consideramos “interpretabilidad mínima” obligatoria para toda run `MLP`?
- ¿Qué consideramos “interpretabilidad mínima” obligatoria para toda run `XGBoost`?
- ¿Queremos una salida compacta uniforme (`interpretability_summary.json`) para ambas familias?
- ¿Qué artefactos interpretativos consideras demasiado pesados como para guardarlos en todas las runs?
- ¿Queremos que una run cuente como válida sin interpretabilidad mínima, o eso la deja incompleta?

### Riesgo si se deja ambiguo

- campañas imposibles de auditar sin reentrenar;
- `outputs/` demasiado grandes;
- resultados difíciles de defender sin insight model-dependent.

---

## 10. Diseñar la capa mínima de interpretabilidad para `MLP`

Estado:

- cerrado

Artefactos canónicos:

- [config/f7_mlp_interpretability_contract_v1.yaml](../config/f7_mlp_interpretability_contract_v1.yaml)
- [docs/f7_block_10_mlp_interpretability_rationale.md](f7_block_10_mlp_interpretability_rationale.md)

Decisión cerrada:

- toda run `MLP` `F7` produce interpretabilidad ligera en `raw` output space;
- la salida mínima es `global` + `per_class`;
- `val` se interpreta siempre y `test` también cuando la run es `holdout_run`;
- el método canónico es perturbación ligera, no SHAP;
- `flowpre_candidate_*` se interpreta primero en latentes y luego se proyecta a features semánticas canónicas con un cache condicionado por clase;
- la superficie principal para análisis posterior es la importancia proyectada a features semánticas;
- el cierre `10B` colapsa `ilr_chem_* -> chem_*` y `ilr_phase_* -> phase_*`, de modo que los artefactos finales primarios ya no contienen nombres `ilr_*`;
- la magnitud primaria sigue siendo una sensibilidad en unidades de cambio del target `raw`, no un porcentaje causal de aporte;
- además de `mean_abs_delta_pred_raw` y `mean_signed_delta_pred_raw`, toda run persiste ahora:
  - `sum_abs_delta_pred_raw`
  - `std_abs_delta_pred_raw`
  - `median_abs_delta_pred_raw`
  - `p90_abs_delta_pred_raw`
  - `p95_abs_delta_pred_raw`
  - `stderr_abs_delta_pred_raw`
  - `share_abs_importance`
- el coste real observado de interpretabilidad es bajo en `scaled` y bastante más alto en `flowpre_candidate_*`:
  - `scaled_standard`: ~`0.193s` medios por run
  - `flowpre_candidate_1`: ~`11.171s`
  - `flowpre_candidate_2`: ~`10.111s`

---

## 11. Diseñar la capa mínima de interpretabilidad para `XGBoost`

Estado:

- cerrado

Artefactos canónicos:

- [config/f7_xgb_interpretability_contract_v1.yaml](../config/f7_xgb_interpretability_contract_v1.yaml)
- [docs/f7_block_11_xgb_interpretability_rationale.md](f7_block_11_xgb_interpretability_rationale.md)

### Objetivo

Decidir cómo entra SHAP o su equivalente en la campaña sin inflar el coste innecesariamente.

### Por que va aqui

`XGBoost` tiene una vía natural de interpretabilidad, pero hay que convertirla en política de campaña.

Decisión cerrada:

- toda run `XGBoost` `F7` produce interpretabilidad en dos capas:
  - `SHAP` nativo con `TreeExplainer`
  - perturbación ligera puente con `MLP`
- ambas capas existen para:
  - `global`
  - `per_class`
  - `val`
  - `test` cuando la run es `holdout_run`
- la superficie canónica de features es exactamente la del modelo:
  - `type_*`
  - features numéricas raw
- `SHAP` usa como métrica principal:
  - `mean_abs_shap`
- la capa puente usa como métrica principal:
  - `mean_abs_delta_pred_raw`
- ambas capas persisten también:
  - `sum_abs_*`
  - `std_abs_*`
  - `median_abs_*`
  - `p90_abs_*`
  - `p95_abs_*`
  - `stderr_abs_*`
  - `share_abs_importance`
- no se guardan matrices completas de SHAP por muestra para todas las runs;
- la comparabilidad cruzada fuerte entre `MLP` y `XGBoost` se apoya en la capa puente de perturbación.

### Riesgo si se deja ambiguo

- o bien no guardar suficiente interpretabilidad;
- o bien cargar la campaña con demasiado artefacto pesado para una baseline.

---

## 12. Formalizar `campaign spec`

### Objetivo

Pasar de documentos narrativos a una especificación machine-readable de campaña.

### Por que va aqui

Es la pieza que conecta:

- inventario de datasets
- run specs
- panel de seeds
- ids canónicos
- política de artefactos

### Entregables esperados

- `campaign spec` formal de `F7`;
- inventario de trials derivable desde esa spec;
- coherencia entre plan, ids y runners.

### Cierre

Bloque cerrado con:

- spec raíz en [config/f7_campaign_spec_v1.yaml](../config/f7_campaign_spec_v1.yaml)
- normalización de `dataset_candidate_id`
- inventario derivado de `run_spec`
- inventario derivado de `trial_id`
- manifest de expansión con conteos y fingerprints

Artefactos derivados:

- `outputs/reports/f7_campaign_spec/f7_campaign_dataset_candidates_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_run_specs_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_trials_v1.csv`
- `outputs/reports/f7_campaign_spec/f7_campaign_expansion_manifest_v1.json`

Rationale:

- [docs/f7_block_12_campaign_spec_rationale.md](f7_block_12_campaign_spec_rationale.md)

### Riesgo si se deja ambiguo

- runners con lógica enterrada en código;
- difícil auditoría del plan real de campaña;
- más riesgo de lanzar combinaciones fuera de contrato.

---

## 13. Construir runners de campaña

### Objetivo

Tener runners reales para `MLP` y `XGBoost` alineados con la campaña, no solo scripts precedentes.

### Por que va aqui

Es el punto en que el diseño pasa a ser ejecución reproducible.

### Entregables esperados

- runner `MLP`
- runner `XGBoost`
- política de resume/restart
- política de outputs mínimos
- integración con manifests e ids

### Cierre

Bloque cerrado con:

- entrada única de campaña:
  - [scripts/run_f7_campaign.py](../scripts/run_f7_campaign.py)
- orquestación reusable:
  - [evaluation/f7_campaign_runner.py](../evaluation/f7_campaign_runner.py)
- capa de estado/ledger/registry:
  - [evaluation/f7_campaign_state.py](../evaluation/f7_campaign_state.py)

Semántica ya fijada:

- consumo directo del `trial inventory` congelado;
- estado mutable por trial en JSON;
- `trial_ledger.csv` como superficie humana canónica;
- `trial_attempts.jsonl` append-only;
- `campaign_manifest.json`, `summary.json` y `campaign_closeout.json`;
- `resume`, `rerun-failed`, `close` y `rebuild-state`;
- linaje explícito para campañas de extensión por seeds.

Validación realizada:

- tests unitarios y de integración mínima del runner;
- `preflight` real sobre la spec canónica;
- ejecución mixta mínima real `MLP + XGBoost` en test controlado.

Rationale:

- [docs/f7_block_13_campaign_runners_rationale.md](f7_block_13_campaign_runners_rationale.md)

### Riesgo si se deja ambiguo

- demasiada lógica implícita dentro del launcher;
- problemas de reanudación;
- inconsistencia entre familias de modelo.

---

## 14. Congelar la gramática del análisis principal

### Estado

- cerrado el 2026-05-21

### Cierre

El bloque queda congelado como gramática estadística canónica de `F7`, sin esperar ya a la campaña grande para decidir:

- cómo se selecciona antes del freeze;
- cómo se reporta después del freeze;
- cuál es la unidad observacional principal;
- cuál es la unidad agregada principal;
- qué capa métrica gobierna;
- qué papel tienen ANOVA/post-hoc;
- y cómo se trata explícitamente el riesgo tipo Simpson.

Queda fijado que:

- `val` gobierna shortlist y decisiones pre-freeze;
- `val` y `test` quedan como superficies co-principales del readout final una vez la campaña esté congelada;
- la unidad observacional principal es la `run` individual por seed;
- la unidad agregada principal es `lineage_trial_group_id`;
- la métrica principal de backbone es `raw_real macro rrmse`;
- esa métrica debe leerse obligatoriamente junto a:
  - `per_class_rrmse`
  - `worst_class_rrmse`
  - `class_dispersion_rrmse`
  - `simpson_gap_rrmse`
- la jerarquía metodológica principal es:
  - descriptivo multi-seed
  - comparaciones `paired / blocked`
  - `mixed-effects` o regresión con estructura repetida
- `ANOVA` y `post-hoc` quedan como capa de apoyo, no como backbone inferencial;
- `XGBoost` se mantiene como baseline acotada dentro del mismo marco analítico, no como subestudio paralelo;
- la interpretabilidad transversal primaria queda en `semantic_bridge_perturbation`;
- las superficies auxiliares quedan en:
  - `xgb_native_shap`
  - `mlp_flowpre_native_latent_perturbation`
- el `top-k` principal para estabilidad de interpretabilidad queda fijado en `10`;
- el umbral mínimo de relevancia práctica queda reconocido y versionado como `TBD_before_final_reporting`, no implícito.

### Entregables realizados

- plan narrativo ampliado:
  - [docs/f7_statistical_analysis_plan.md](f7_statistical_analysis_plan.md)
- plan estructurado:
  - [docs/f7_statistical_analysis_plan_structured.md](f7_statistical_analysis_plan_structured.md)
- spec machine-readable:
  - [docs/f7_statistical_analysis_spec.yaml](f7_statistical_analysis_spec.yaml)

Rationale:

- [docs/f7_block_14_analysis_grammar_rationale.md](f7_block_14_analysis_grammar_rationale.md)

### Riesgo si se deja ambiguo

- resultados abundantes pero no realmente defendibles;
- análisis posteriores demasiado improvisados;
- contrasts y métodos usados de forma inconsistente según la familia o el notebook;
- lectura macro que oculte otra vez el problema class-wise / Simpson.

---

## 15. Hacer benchmark real de coste

### Estado

- cerrado el 2026-05-21

### Cierre

El bloque queda cerrado usando como benchmark canónico la evidencia real ya observada en la cadena de validación final de Block `13`, sin lanzar una mini-campaña nueva solo para medir coste.

Fuente observada usada:

- `primary`: `104` runs
- `extension_1`: `52` runs
- `extension_2`: `52` runs
- total: `208` runs válidas con el runner, lineage, reporting e interpretabilidad canónicos ya activos

Resumen observado sobre esas `208` runs:

- `training_runtime_s`
  - mean: `4.751`
  - median: `4.715`
  - `p90`: `6.889`
  - `p95`: `7.708`
- `interpretability_runtime_s`
  - mean: `7.724`
  - median: `0.351`
  - `p90`: `16.639`
  - `p95`: `17.303`
- `total_runtime_s`
  - mean: `12.475`
  - median: `7.919`
  - `p90`: `22.601`
  - `p95`: `23.512`

Lectura por grupos relevantes:

- `MLP FlowPre`
  - mean total runtime: `21.573 s/run`
- `MLP no-FlowPre`
  - mean total runtime: `5.253 s/run`
- `XGBoost`
  - mean total runtime: `1.221 s/run`

Como la validación pequeña sobrepondera `FlowPre`, la estimación final para `17400` se corrige por la mezcla estructural real de la campaña grande.

Estimación media ajustada a la mezcla real de `17400`:

- mean total runtime equivalente: `10.628 s/run`
- tiempo medio por seed completa (`580` runs): `1.712 h`
- tiempo medio para la `primary` de `30` seeds: `51.367 h`
- tiempo medio por extensión adicional de `30` seeds: `51.367 h`
- tiempo medio para la cadena completa `30 + 30 + 30 + 30`: `205.468 h`

Lectura conservadora usando el `p90` bruto observado:

- `3.641 h` por seed equivalente
- `109.238 h` para la `primary` de `30` seeds
- `109.238 h` por extensión adicional de `30` seeds
- `436.953 h` para la cadena completa `30 + 30 + 30 + 30`

### Entregables realizados

- rationale de benchmark observado:
  - [docs/f7_block_15_cost_benchmark_rationale.md](f7_block_15_cost_benchmark_rationale.md)

### Riesgo si se deja ambiguo

- campaña demasiado larga sin preverlo;
- decisiones operativas pobres de batching/paralelismo.

---

## 16. Hacer preflight final de lanzamiento

### Estado

- cerrado el 2026-05-21

### Cierre

El bloque queda cerrado con una política de preflight final estricta y con ejecución real de una `readiness pass` no mutante sobre la cadena grande completa.

Queda fijado que:

- la validación real previa suficiente es la cadena pequeña `104 + 52 + 52` ya cerrada en el bloque `13`;
- el preflight final no exige una nueva mini-campaña de entrenamiento, pero sí una `readiness pass` estructural sobre:
  - `primary` de `30` seeds
  - `extension_1` de `30` seeds
  - `extension_2` de `30` seeds
  - `extension_3` de `30` seeds
- el preflight debe dejar un reporte canónico propio;
- la parte principal del preflight queda automatizada;
- el criterio de decisión es `all-green-or-no-go`;
- `test` permanece habilitado como output, pero no como superficie iterativa de selección;
- el `environment freeze` forma parte del readiness final.

Resultado observado del readiness pass:

- `go_no_go = go`
- `campaign_count = 4`
- `expected_total_trial_count = 69600`
- `observed_total_trial_count = 69600`
- `total_seed_count = 120`
- `blockers = []`

### Entregables realizados

- specs grandes de extensión:
  - [config/f7_campaign_extension1_v1.yaml](../config/f7_campaign_extension1_v1.yaml)
  - [config/f7_campaign_extension2_v1.yaml](../config/f7_campaign_extension2_v1.yaml)
  - [config/f7_campaign_extension3_v1.yaml](../config/f7_campaign_extension3_v1.yaml)
- seed panels de extensión:
  - [config/f7_seed_panel_extension1_v1.yaml](../config/f7_seed_panel_extension1_v1.yaml)
  - [config/f7_seed_panel_extension2_v1.yaml](../config/f7_seed_panel_extension2_v1.yaml)
  - [config/f7_seed_panel_extension3_v1.yaml](../config/f7_seed_panel_extension3_v1.yaml)
- readiness tooling:
  - [evaluation/f7_launch_readiness.py](../evaluation/f7_launch_readiness.py)
  - [scripts/report_f7_launch_readiness.py](../scripts/report_f7_launch_readiness.py)
- reporte canónico:
  - [f7_launch_readiness_v1.json](../outputs/reports/f7_launch_readiness/f7_launch_readiness_v1.json)
  - [f7_launch_readiness_v1.md](../outputs/reports/f7_launch_readiness/f7_launch_readiness_v1.md)

### Riesgo si se deja ambiguo

- lanzar toda la campaña con una cadena que solo estaba comprobada por piezas;
- tener que abortar después de muchas runs por un problema detectable antes.

---

## Uso recomendado de este documento

Este documento debe servir para:

- decidir qué trabajamos primero en cada bloque;
- usar las preguntas como guía de conversación antes de implementar;
- comprobar que no estamos saltando a código antes de haber decidido lo suficiente para hacerlo bien.

## Criterio práctico de cierre por bloque

Un bloque puede considerarse realmente cerrado cuando:

- sus preguntas críticas ya tienen respuesta o aplazamiento explícito;
- existe al menos un entregable verificable de ese bloque;
- el siguiente bloque puede arrancar sin depender de supuestos ocultos;
- el estado resultante ya puede reflejarse honestamente en [docs/f7_campaign_readiness_checklist.md](f7_campaign_readiness_checklist.md).

Si al cerrar un bloque seguimos necesitando improvisar varias decisiones básicas en el siguiente, probablemente el bloque anterior no estaba todavía bien rematado.
