# F7 Block 07 Meta Grammar Rationale

## Decision

El bloque `7` congela la gramática meta mínima de campaña para `F7`.

El artefacto canónico es:

- [config/f7_meta_grammar_v1.yaml](../config/f7_meta_grammar_v1.yaml)

La identidad global de campaña queda fijada como:

- `campaign_id = f7_campaign_v1`

## Qué queda fijado

Se fijan como ids canónicos:

- `dataset_candidate_id`
- `run_spec_id`
- `trial_id`
- `campaign_id`
- `comparison_group_id`
- `statistical_family_id`

Y también:

- qué significa cada uno;
- de qué componentes depende;
- dónde debe persistirse;
- y qué ids son intrínsecos a la run frente a qué ids dependen del análisis.

## Decisiones principales

### 1. `dataset_candidate_id`

No se reconstruye desde nombres libres de runner.

Se copia desde el inventario materializado canónico de `4B`, que ya es la fuente de verdad de los datasets `F7`.

### 2. `run_spec_id`

Excluye la seed.

Esto fuerza una separación limpia entre:

- identidad del régimen de entrenamiento/comparación;
- e identidad de la réplica.

La seed pertenece a `trial_id`, no a `run_spec_id`.

### 3. `trial_id`

Se define como id derivable pero además persistido explícitamente.

Eso evita dos problemas:

- depender siempre de recomponerlo a posteriori;
- y permitir que una tabla o manifest pierda trazabilidad si faltan columnas auxiliares.

### 4. `campaign_id`

Queda único para toda `F7`.

No se abre aquí una semántica de subcampañas separadas porque complicaría la comparabilidad sin necesidad antes de tiempo.

### 5. `comparison_group_id`

Se fija como grupo operativo de comparación y fairness.

No tiene por qué quedarse encerrado en una sola familia de modelo.

Puede cruzar familias cuando la comparación sea explícita y defendible, por ejemplo:

- `MLP` frente a `XGBoost` sobre el mismo dataset candidato.

### 6. `statistical_family_id`

Aquí está la distinción más importante del bloque.

No se congela como un “grupo amplio” intrínseco a cada run, porque el análisis posterior no va a ser de una sola capa. Queremos poder estudiar:

- efectos individuales;
- combinaciones de efectos;
- y sinergias.

Por eso `statistical_family_id` se define como:

- narrow;
- analysis-specific;
- y potencialmente múltiple sobre el mismo conjunto subyacente de runs.

Eso significa que:

- la gramática del campo queda cerrada ya;
- pero las familias estadísticas concretas se refinan más adelante, cuando se congele la gramática del análisis principal.

## Semántica por familia de modelo

La campaña usa el mismo panel de seeds para `MLP` y `XGBoost`, pero no fuerza una simetría artificial de implementación interna.

Lo que sí exige es:

- misma identidad de réplica a nivel de campaña;
- distinto binding interno cuando corresponda;
- persistencia explícita del `seed_set_id` y de la seed concreta.

## Por qué esta división es la más limpia

Si mezcláramos ahora:

- gramática base de identidad;
- con la gramática final de todos los análisis estadísticos posibles,

cerraríamos demasiado pronto una decisión que todavía depende del bloque posterior de análisis.

La solución adoptada deja:

- identidad de dataset/config/réplica/campaña cerrada ya;
- comparabilidad operativa cerrada ya;
- y familias estadísticas finales aún refinables sin romper la gramática base.

## Propagación operativa

Esta gramática debe alimentar después, como mínimo:

- campaign spec;
- manifests de run;
- tablas canónicas de resultados;
- y la futura gramática del análisis principal.

## Rationale resumido

- la campaña necesitaba ids canónicos legibles y persistibles;
- `run_spec_id` excluye seed y `trial_id` la incorpora;
- `dataset_candidate_id` se hereda del inventario `4B`;
- `comparison_group_id` es el grupo operativo de comparación y puede cruzar familias cuando el contraste sea explícito;
- `statistical_family_id` queda definido como narrow y analysis-specific para soportar efectos individuales y sinergias sin contaminar agregados.
