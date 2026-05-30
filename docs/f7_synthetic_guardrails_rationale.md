# F7 Synthetic Guardrails Rationale

## Decision

La política de masa sintética del bloque `5` se mantiene separada de la política de aceptación de muestras.

La separación final queda así:

- política de cuotas por clase:
  - [config/f7_synthetic_cap_policy_v1.yaml](../config/f7_synthetic_cap_policy_v1.yaml)
  - [data/f7_synthetic_cap_policy.py](../data/f7_synthetic_cap_policy.py)
- política de guardrails y auditoría por muestra:
  - [config/f7_synthetic_guardrails_v1.yaml](../config/f7_synthetic_guardrails_v1.yaml)
  - [data/f7_synthetic_guardrails.py](../data/f7_synthetic_guardrails.py)

## Qué problema resuelve

La policy de masa por sí sola no basta para decidir si una muestra sintética individual debe aceptarse.

Hacía falta fijar también:

- qué es un sample imposible o inválido;
- qué debe provocar reintento;
- qué debe quedar solo como auditoría;
- y cómo dejar esa información persistida en manifests y reports.

## Regla por espacios

La referencia “raw” para esta capa no es el CSV industrial original.

La referencia canónica es el bundle modelable:

- post-cleaning;
- post-outlier-removal;
- pre-scaling;
- el mismo espacio usado para `FlowPre` y `FlowGen`.

Esto permite que:

- `FlowGen` genere y se valide primero en ese espacio modelable raw;
- `KMeans-SMOTE` genere en el bundle materializado, pero pueda validarse también contra ese espacio raw cuando la reconstrucción sea factible.

## Hard rejects

La capa F7 rechaza y reintenta, como mínimo:

- valores no finitos;
- clase inválida;
- violaciones de cuota F7 por clase;
- sintéticos en clase mayoritaria;
- duplicados exactos frente a reales o sintéticos ya aceptados;
- `init < 0`;
- y valores negativos en columnas modeladas que, por contrato práctico, deben permanecer no negativas.

Las excepciones explícitas de no negatividad son:

- `a`
- `b`
- columnas `ilr_*`

## Soft audits

Las reglas de cleaning learned no entran como rechazo duro en esta iteración.

Quedan como auditoría:

- reglas univariadas;
- `IsolationForest`;
- overlap learned;
- y resumen opcional de distancia al vecindario real.

La razón es metodológica:

- queremos visibilidad sobre muestras sospechosas;
- pero no endurecer todavía el pipeline con filtros learned que podrían sesgar demasiado la generación.

## Flujo final por policy

### `flowgen_official` y `flowgen_train_only`

- generar candidatos en modeled-raw;
- validar en modeled-raw;
- aceptar hasta cumplir cuota;
- y solo después reexpresar en el bundle downstream correspondiente.

### `kmeans_smote`

- generar dentro del bundle materializado no sintético;
- deshacer la normalización métrica interna;
- validar en el bundle materializado;
- y, cuando sea posible, reconstruir además al modeled-raw para aplicar checks de dominio y auditoría learned.

## Persistencia mínima exigida

Cada dataset sintético F7 debe dejar:

- policy id de cuotas;
- policy id de guardrails;
- conteos aceptados y rechazados;
- rechazos por motivo;
- intentos por clase;
- auditoría soft por familia de reglas;
- y fingerprints de las muestras aceptadas.

## Rationale resumido

- la cuota por clase y la aceptación por muestra son dos responsabilidades distintas;
- `FlowGen` y `KMeans-SMOTE` no generan en el mismo espacio y por eso la validación debía explicitarlo;
- los learned cleaning rules se monitorizan, pero no endurecen todavía la aceptación;
- y la persistencia debía ser suficientemente rica como para soportar análisis posteriores de calidad y comparabilidad.
