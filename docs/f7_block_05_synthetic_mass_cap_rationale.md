# F7 Block 05 Synthetic Mass Cap Rationale

## Decision

El bloque `5` fija la política operativa de masa sintética para `F7`.

La política canónica queda en:

- [config/f7_synthetic_cap_policy_v1.yaml](../config/f7_synthetic_cap_policy_v1.yaml)

Y la validación central reusable queda en:

- [data/f7_synthetic_cap_policy.py](../data/f7_synthetic_cap_policy.py)

La capa de guardrails y auditoría por muestra queda separada y documentada en:

- [config/f7_synthetic_guardrails_v1.yaml](../config/f7_synthetic_guardrails_v1.yaml)
- [data/f7_synthetic_guardrails.py](../data/f7_synthetic_guardrails.py)
- [docs/f7_synthetic_guardrails_rationale.md](./f7_synthetic_guardrails_rationale.md)

## Regla final

La clase mayoritaria se define usando solo los conteos reales de `train` del split oficial.

Para cada clase:

- si es mayoritaria:
  - `n_synth = 0`
- si es estrictamente minoritaria:
  - `n_synth(c) <= floor(0.5 * n_real(c))`
  - `n_real(c) + n_synth(c) <= max_real_train`

Si hubiera empate en la clase mayoritaria:

- no se añaden sintéticos a ninguna de las clases empatadas arriba;
- solo a las estrictamente menores.

## Por que la regla es doble

No bastaba con limitar la masa sintética al `50%` de cada clase minoritaria.

También había que impedir que una minoritaria aumentada superase a la mayoritaria real.

La combinación de ambas condiciones hace la policy más defendible porque:

- mantiene el desbalance bajo control;
- evita que una policy “gane” por inyectar demasiado volumen;
- y conserva una lectura más limpia del efecto del algoritmo sintético.

## Aplicación homogénea

La misma regla aplica sin excepciones a:

- `flowgen_official`
- `flowgen_train_only`
- `kmeans_smote`

Esto es importante para que una comparación entre policies no mezcle:

- diferencias de generador;
- con diferencias de masa sintética.

## Política de reconstrucción

Para `F7` no se reutiliza ni se parchea silenciosamente material sintético previo.

La decisión es:

- preservar cualquier material anterior como histórico o legacy;
- y construir desde cero las versiones `F7` compatibles con esta policy.

## Política de manifest

Cada dataset sintético de campaña debe guardar, por clase:

- `n_real`
- `n_synth`
- `n_final`
- fracción sintética final

Y además un resumen central de validación que permita decidir si el dataset es o no `campaign-ready`.

## Política de implementación

La preferencia no es generar de más y luego recortar, sino:

- generar directamente con los objetivos compatibles con la policy;
- y adaptar `kmeans_smote` si hace falta para permitir ese control fino.

## Qué queda listo ya

El bloque deja ya preparada una validación central reusable:

- resume la masa real y sintética por clase;
- marca clases mayoritarias de referencia;
- calcula el máximo permitido por la regla F7;
- y declara si el dataset pasa o no la policy.

Además, los manifests de datasets sintéticos nuevos de:

- `kmeans_smote`
- `flowgen`

ya pueden incluir esta validación en su metadata.

## Qué no queda cerrado todavía

Este bloque fija la policy y la validación.

La rematerialización final de datasets sintéticos `F7-ready` forma parte de la fase `4B`, ya apoyada ahora en esta policy común.

## Rationale resumido

- la masa sintética debía alinearse igual para las tres policies;
- la regla final no es solo el `50%`, sino también no superar a la mayoritaria real;
- no se recicla material sintético previo;
- cada dataset sintético debe declarar sus conteos por clase;
- y la aceptación de un dataset `campaign-ready` debe pasar por una validación central obligatoria.
