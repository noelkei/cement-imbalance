# F7 Trial Traceability Rationale

## Proposito y estatus

Este documento define la gramatica de identidad, trazabilidad y comparabilidad que debe respetar cualquier prueba de `F7`.

No es:

- el contrato final de ejecucion de `F7`;
- la shortlist final de combinaciones;
- la decision final sobre `5x` vs `10x`;
- la implementacion de nuevas `synthetic_policy`.

Si es:

- un rationale operativo previo al contrato final;
- una guia para que cada trial, rerun, family aggregation y comparacion estadistica quede bien identificado;
- una referencia para no mezclar identidad de dataset, identidad de configuracion, identidad de replica e identidad de campaña.

Este documento complementa a [docs/f7_experimental_space_rationale.md](f7_experimental_space_rationale.md):

- el rationale del espacio experimental responde **que ejes existen**;
- este documento responde **como se identifica, congela, repite, compara y agrega cada prueba**.

Debe servir para soportar:

- repeticion por seeds;
- comparacion justa entre datasets y familias;
- ANOVA y post hoc;
- seleccion de finalistas en `val`;
- confirmacion final posterior.

## Principio central: cuatro niveles de identidad

Cada prueba de `F7` debe leerse como la composicion de cuatro capas de identidad:

1. `dataset identity`
2. `run config identity`
3. `replication identity`
4. `comparison/campaign identity`

Regla principal:

- nunca debe asumirse que dos runs son comparables solo porque tienen el mismo `run_id`-like naming;
- la comparabilidad defendible exige que las cuatro capas queden registradas por separado.

### Resumen de capas

| capa | pregunta que responde | cambia cuando... | no cambia cuando... |
| --- | --- | --- | --- |
| `dataset identity` | que bundle exacto se consumio | cambia transform, policy sintetica, procedencia upstream o manifest | solo cambia la seed de entrenamiento |
| `run config identity` | con que regimen se entreno / comparo | cambia modelo, base config, loss, batching o regla de seleccion | se mantiene la misma config y solo cambia la replica |
| `replication identity` | que repeticion concreta es | cambia la seed o el rol de esa seed | se agrega family-level sobre el mismo panel |
| `comparison/campaign identity` | en que experimento estadistico vive | cambia campaña, fase o alcance de analisis | se compara dentro del mismo grupo coherente |

## Dataset identity

La identidad de dataset describe el bundle que entra a downstream, independientemente de cuantas runs lo consuman.

### Campos obligatorios

| campo | rol |
| --- | --- |
| `split_id` | fija el split oficial usado por el bundle |
| `cleaning_policy_id` | fija la politica de cleaning upstream |
| `dataset_contract_id` | ancla el espacio contractual que define el bundle |
| `dataset_name` | nombre canonico del dataset materializado |
| `dataset_manifest_path` | manifest concreto del bundle consumido |
| `dataset_storage_family` | familia de almacenamiento del bundle |
| `dataset_level_axes` | identidad por ejes `dataset-level` |
| `synthetic_policy_id` | identidad concreta de la mutacion sintetica |
| `source_dataset_manifest` | raw bundle o bundle fuente inmediato |
| `source_split_manifest` | manifest del split |
| `source_cleaning_manifest` | manifest del cleaning |
| `upstream_model_manifests` | modelos upstream de los que depende el bundle |
| `counts_by_split` | conteos por split para trazabilidad |
| `counts_by_class` | conteos por clase para trazabilidad |
| `line` | `official`, `experimental_train_only`, `baseline_optional` |
| `is_local_only` | marca si el bundle es local-only o parte del camino canonico |

### `dataset_level_axes`

Los ejes obligatorios son:

- `x_transform`
- `y_transform`
- `synthetic_policy`

Reglas:

- `synthetic_policy` solo puede mutar `train`;
- `val/test` siguen anclados al mismo split oficial;
- dos datasets con igual `x_transform`, `y_transform` y `synthetic_policy` **no son el mismo dataset** si cambia:
  - el `dataset_manifest_path`;
  - el `synthetic_policy_id`;
  - el `source_*_manifest`;
  - la procedencia upstream real.

### Distincion de linea

El documento debe fijar que todo dataset candidato declare al menos:

- `line = official`
- `line = experimental_train_only`
- `line = baseline_optional`

Y, cuando aplique:

- `is_local_only = true`

para distinguir sin ambiguedad:

- bundles oficiales canonicos;
- bundles locales derivados de la rama `train_only`;
- bundles de baseline o comparativa externa.

### Relacion con lo que ya existe

Esto ya tiene soporte parcial real en el repo mediante:

- `dataset_level_axes` en [data/dataset_contract.py](../data/dataset_contract.py)
- manifests bajo `data/sets/official/.../meta/manifest.json`
- `synthetic_policy_id`, `source_*_manifest` y `upstream_model_manifests` en los bundles ya materializados

## Run config identity

La identidad de configuracion describe con que regimen de entrenamiento o comparacion se ejecuto la prueba.

No debe mezclarse con:

- la identidad del dataset;
- la replica por seed;
- la campaña de seleccion.

## `MLP run spec`

### Campos obligatorios

| campo | rol |
| --- | --- |
| `model_family = mlp` | identifica la familia de modelo |
| `closure_contract_id` | contrato de cierre que gobierna la corrida |
| `mlp_base_config_id` | id canonico del base config |
| `base_config_path` | ruta del base config usado |
| `allow_test_holdout` | deja claro si `test` estuvo habilitado |
| `objective_metric_id` | metrica de seleccion declarada |
| `run_level_axes` | ejes run-level activos |

### `run_level_axes` obligatorios en `MLP`

- `batch_policy`
- `cycling_policy`
- `loss_policy`
- `allow_synth`

### Hiperparametros congelados que afectan comparabilidad

Aunque no todos formen parte del id corto, deben quedar registrados como parte de la identidad congelada del regimen:

- `batch_size`
- `learning_rate`
- `num_epochs`
- `early_stopping_patience`
- `lr_scheduler`
- `optimizer`

### Reglas de interpretacion

- `objective_metric` hoy es criterio de seleccion post-run;
- `val_loss` sigue siendo la senal nativa de early stopping;
- cambiar cualquiera de estos elementos cambia la identidad de la `run spec`, aunque el dataset permanezca igual.

### Relacion con lo que ya existe

Esto ya tiene soporte parcial en:

- `comparison_contract` construido en [training/train_mlp.py](../training/train_mlp.py)
- [config/mlp_closure_base_v1.yaml](../config/mlp_closure_base_v1.yaml)
- [config/closure_contract_v1.yaml](../config/closure_contract_v1.yaml)

## `XGBoost run spec`

`XGBoost` se documenta como baseline opcional separado. No debe forzarse artificialmente a la misma gramatica de `MLP`.

### Campos obligatorios si entra

| campo | rol |
| --- | --- |
| `model_family = xgboost` | identifica la familia de modelo |
| `model_config_id` | id canonico de la config de boosting |
| `model_config_payload` o referencia a script/config | deja congelado el regimen real |
| `feature_policy` | politica de features de entrada |
| `objective_metric_id` | criterio comparativo declarado |
| `allow_test_holdout` | deja claro si `test` estuvo habilitado |
| `seed` | seed de esa run |

### Regla principal

- `XGBoost` no consume `batch_policy`, `cycling_policy` ni `loss_policy` en el mismo sentido que `MLP`;
- su metadata debe ser comparable, pero no simetrica por fuerza.

### Camino principal recomendado

Mientras siga siendo baseline opcional, el camino principal esperado debe ser:

- `raw`
- con `synthetic_policy` dataset-level si existe bundle comparable

y no una replica completa del espacio de transforms de `MLP`.

### Relacion con lo que ya existe

Precedente operativo actual:

- [scripts/run_xgboost_temporal_vs_legacy.py](../scripts/run_xgboost_temporal_vs_legacy.py)

## Replication identity

Esta capa fija que una prueba metodologica no es una run suelta, sino una unidad replicada por seed.

El documento **no** congela todavia si el panel final sera `5x` o `10x`, pero si congela la estructura que cualquier panel debe declarar.

### Campos obligatorios

| campo | rol |
| --- | --- |
| `seed` | seed concreta de la run |
| `seed_role` | rol metodologico de la seed |
| `seed_set_id` | panel canonico al que pertenece |
| `seed_panel_version` | version del panel de seeds |
| `replication_index` | indice de esa replica dentro del panel |
| `seed_source_policy` | politica de procedencia de la seed |
| `run_id` | identidad final de la run materializada |

### Valores sugeridos para `seed_role`

- `single`
- `panel_member`
- `anchor`
- `reseed`
- `confirmation`

### Reglas

- varias runs pertenecen al mismo experimento replicado solo si comparten `seed_set_id` o si la compatibilidad entre paneles queda declarada;
- cambiar el panel de seeds cambia la comparabilidad estadistica;
- no debe mezclarse en el mismo agregado una familia con paneles distintos sin declararlo de forma explicita.

### Regla critica para agregados

Todo agregado:

- family-level
- dataset-level
- campaign-level

debe declarar con que `seed_set_id` se calculo.

### Relacion con lo que ya existe

Esto ya aparece de forma parcial en:

- `seed_set_id` del `comparison_contract` en [training/train_mlp.py](../training/train_mlp.py)
- `seed_set_id` de `mlp_closure_base_v1`
- finalists manifests y reseeds de `FlowGen official` y `FlowGen train_only`

## Comparison / campaign identity

Esta capa agrupa runs para analisis, seleccion y estadistica.

### Campos obligatorios

| campo | rol |
| --- | --- |
| `comparison_group_id` | grupo de comparacion metodologica |
| `campaign_id` | campaña concreta de ejecucion o analisis |
| `phase_id` | fase del proyecto en la que vive |
| `selection_phase` | fase de seleccion declarada |
| `selection_role` | rol de la run o familia en la seleccion |
| `selection_status` | estado de la seleccion |
| `line` | `official`, `experimental_train_only`, `baseline_optional` |
| `statistical_family_id` | unidad estadistica de comparacion |
| `analysis_scope` | alcance de analisis |

### Valores sugeridos para `analysis_scope`

- `exploratory`
- `shortlist`
- `finalist`
- `confirmation`
- `final_comparison`

### Regla principal

Esta capa debe dejar clarisimo:

- que conjunto de runs puede entrar en el mismo ANOVA;
- que ids deben compartirse para que un post hoc sea defendible;
- que ids deben diferir para no mezclar campañas incompatibles.

## Semantica estadistica minima para ANOVA y post hoc

Esta seccion no fija todavia el test estadistico exacto, pero si la gramatica minima para que luego sea defendible.

### Unidad observacional

- una run individual con seed concreta

### Unidad agregada

- una familia o config agregada sobre un panel de seeds declarado

### Agrupaciones minimas que deben registrarse

- `dataset_candidate_id`
- `run_spec_id`
- `seed_set_id`
- `comparison_group_id`
- `model_family`
- `line`

### Comparaciones validas

Son validas cuando comparten:

- mismas reglas de split;
- mismo criterio de seleccion;
- mismo objetivo de comparacion;
- mismo panel de seeds o panel explicitamente declarado compatible.

### Comparaciones que no deben mezclarse

- `official` y `train_only` sin marcar la linea;
- `MLP` y `XGBoost` sin declarar `model_family`;
- datasets con distinta `synthetic_policy` bajo el mismo agregado sin declararlo;
- runs confirmatorias mezcladas con exploratorias como si fueran el mismo experimento;
- familias agregadas sobre paneles distintos sin declarar el `seed_set_id`.

## IDs canonicos recomendados

El documento debe proponer estos ids conceptuales:

| id | representa |
| --- | --- |
| `dataset_candidate_id` | identidad estable del dataset candidato |
| `run_spec_id` | identidad estable del regimen de entrenamiento/comparacion |
| `seed_set_id` | identidad del panel de replicas |
| `trial_id` | identidad final de la run concreta |
| `comparison_group_id` | grupo de comparacion metodologica |
| `campaign_id` | campaña operativa o analitica |
| `statistical_family_id` | unidad de comparacion estadistica |

### Regla para `trial_id`

`trial_id` debe ser la union de:

- `dataset identity`
- `run config identity`
- `replication identity`

No debe absorber directamente:

- el resultado numerico;
- la decision de seleccion posterior;
- el agregado estadistico de la campaña.

## Relacion con manifests, results y seleccion de finalistas

El documento debe aterrizar que piezas del repo ya soportan esta trazabilidad y cuales faltan.

### Precedentes reales ya existentes

- `evaluation_context` y `comparison_contract` en [training/train_mlp.py](../training/train_mlp.py)
- manifests de datasets derivados bajo `data/sets/official/.../meta/manifest.json`
- `run_context` y artefactos canonicos guardados por `train_mlp`
- finalists manifests de `FlowGen official` y `FlowGen train_only`
- `seed_set_id`, `base_config_id`, `comparison_group_id`

### Tabla de estado

| pieza | estado recomendado en este documento | nota |
| --- | --- | --- |
| `dataset_level_axes` | `already present` | ya existe en manifests y contract |
| `run_level_axes` | `already present` | ya existe en `comparison_contract` de `train_mlp` |
| `seed_set_id` | `already present` | ya existe, pero debe explicitarse como eje de comparabilidad |
| `base_config_id` | `already present` | ya existe en `train_mlp` y configs |
| `comparison_group_id` | `present but inconsistent` | existe campo, pero aun no hay gramática de uso cerrada |
| `dataset_candidate_id` | `missing and should be added later` | hace falta fijarlo conceptualmente |
| `run_spec_id` | `missing and should be added later` | hace falta fijarlo conceptualmente |
| `trial_id` | `missing and should be added later` | hoy se usa `run_id`, pero no cubre toda la semántica |
| `statistical_family_id` | `missing and should be added later` | necesario para ANOVA/post hoc robustos |
| `seed_panel_version` | `missing and should be added later` | necesario si cambia el panel sin romper trazabilidad |

## Que cierra este documento y que deja abierto

### Lo que si cierra

- la gramatica de identidad y trazabilidad;
- los cuatro niveles de identidad;
- los ids minimos que toda prueba debe poder declarar;
- la separacion entre dataset, config, seed y campaña;
- el tratamiento separado de `XGBoost`.

### Lo que no cierra

- `5x` vs `10x`;
- la shortlist final de `F7`;
- la implementacion concreta de nuevas `synthetic_policy`;
- la forma exacta del analisis estadistico final;
- si `XGBoost` entra finalmente o no.

## Recommended workflow despues de este documento

Secuencia recomendada:

1. cerrar este meta-rationale;
2. revisar si [docs/f7_experimental_space_rationale.md](f7_experimental_space_rationale.md) necesita una nota de cruce menor;
3. implementar nuevas `synthetic_policy` si se aprueban;
4. formalizar despues el contrato operativo de `F7`;
5. solo entonces cerrar la parrilla final y el panel de seeds.

### Regla practica

- no saltar directamente a la parrilla final mientras la trazabilidad de trial no este cerrada;
- no congelar el panel de seeds antes de fijar como se identifican y agregan sus replicas;
- no mezclar estadistica y seleccion si los ids de dataset, config, replica y campaña no estan separados.

## Fuentes del repo que anclan este rationale

- [docs/f7_experimental_space_rationale.md](f7_experimental_space_rationale.md)
- [docs/project_context.md](project_context.md)
- [docs/implementation_status.md](implementation_status.md)
- [config/closure_contract_v1.yaml](../config/closure_contract_v1.yaml)
- [config/mlp_closure_base_v1.yaml](../config/mlp_closure_base_v1.yaml)
- [data/dataset_contract.py](../data/dataset_contract.py)
- [data/sets.py](../data/sets.py)
- [training/train_mlp.py](../training/train_mlp.py)
- [scripts/run_xgboost_temporal_vs_legacy.py](../scripts/run_xgboost_temporal_vs_legacy.py)
- [outputs/models/official/flowgen_finalist/README.md](../outputs/models/official/flowgen_finalist/README.md)
- [outputs/models/experimental/train_only/flowgen_finalist/README.md](../outputs/models/experimental/train_only/flowgen_finalist/README.md)

## Criterio de lectura final

Este documento debe leerse como:

- mas concreto que un rationale de espacio experimental;
- menos rigido que un contrato final de ejecución;
- suficiente para diseñar despues el contrato operativo de `F7` sin perder trazabilidad ni comparabilidad estadistica.
