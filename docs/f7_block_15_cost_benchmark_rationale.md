# F7 Block 15 Cost Benchmark Rationale

## Decision

El bloque `15` queda cerrado sin lanzar una mini-campaña nueva de benchmark.

La decisión es usar como benchmark canónico la evidencia real ya observada bajo la infraestructura final de `F7` en la validación de Block `13`:

- `primary`: `104` runs
- `extension_1`: `52` runs
- `extension_2`: `52` runs
- total observado: `208` runs válidas

Esta evidencia ya cumple lo que el bloque necesitaba metodológicamente:

- runner canónico;
- artifacts y reporting canónicos;
- interpretabilidad activa;
- lineage y closeout reales;
- y tiempos persistidos por run dentro del `trial_ledger.csv`.

## Why No New Benchmark Run

Lanzar un benchmark artificial nuevo añadiría coste operativo, pero no mejoraría realmente la calidad de la evidencia si:

- ya existe una tanda real ejecutada con la cadena canónica;
- los tiempos están persistidos de forma auditable;
- y la estimación puede reconstruirse con trazabilidad desde artefactos reales.

Por tanto, el benchmark de coste se cierra mejor como:

- benchmark observado;
- documentado;
- y ajustado explícitamente por mezcla estructural cuando haga falta.

## Observed Runtime Surface

La fuente observada usada es la suma de los `trial_ledger.csv` de:

- `f7_campaign_block13_validation_primary_v1`
- `f7_campaign_block13_validation_extension_v1`
- `f7_campaign_block13_validation_extension2_v1`

Sobre las `208` runs observadas:

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

## Important Structural Correction

La tanda de validación pequeña no reproduce exactamente la mezcla estructural de la campaña grande.

En la validación observada:

- `MLP FlowPre`: `48`
- `MLP no-FlowPre`: `48`
- `XGBoost`: `8`

En la campaña grande por seed, la mezcla esperada es:

- `MLP FlowPre`: `192`
- `MLP no-FlowPre`: `384`
- `XGBoost`: `4`

Eso implica que la validación pequeña **sobrerrepresenta** `FlowPre`, que es precisamente la parte más lenta.

Por tanto:

- la media simple observada sobre las `208` runs es útil como benchmark bruto;
- pero para estimar la campaña `17400` conviene hacer una corrección por mezcla estructural real.

## Group Means Used For Planning

Medias observadas por grupo relevante:

- `MLP FlowPre`
  - mean total runtime: `21.573 s/run`
- `MLP no-FlowPre`
  - mean total runtime: `5.253 s/run`
- `XGBoost`
  - mean total runtime: `1.221 s/run`

Esto deja una lectura operativa limpia:

- `FlowPre` sigue siendo la parte dominante del coste;
- `MLP` no-`FlowPre` es bastante más barato;
- `XGBoost` tiene coste prácticamente despreciable frente a `MLP`.

## Canonical Planning Estimate

### Mean estimate adjusted to the real `17400` structural mix

Usando la mezcla real por seed:

- `192` runs `MLP FlowPre`
- `384` runs `MLP no-FlowPre`
- `4` runs `XGBoost`

la estimación media ajustada queda en:

- mean total runtime por run equivalente: `10.628 s/run`
- tiempo medio por seed completa (`580` runs): `1.712 h`
- tiempo medio para la `primary` de `30` seeds: `51.367 h`
- tiempo medio por extensión adicional de `30` seeds: `51.367 h`
- tiempo medio para la cadena completa `30 + 30 + 30 + 30`: `205.468 h`

### Conservative planning view from observed `p90`

Como lectura conservadora de planificación, usando el `p90` bruto observado de la validación:

- `22.601 s/run`

la campaña grande queda en:

- `3.641 h` por seed equivalente (`580` runs)
- `109.238 h` para la `primary` de `30` seeds
- `109.238 h` por extensión adicional de `30` seeds
- `436.953 h` para la cadena completa `30 + 30 + 30 + 30`

## Recommended Operational Read

La lectura más limpia y defendible es usar dos números, no uno solo:

- estimación media estructural ajustada:
  - para describir el coste esperado central;
- estimación conservadora basada en `p90` observado:
  - para planificación operativa y margen de seguridad.

Esto evita dos errores:

- subestimar por mirar solo runs baratas;
- sobreestimar por extrapolar sin corregir la sobrecarga relativa de `FlowPre` en la validación pequeña.

## Scope And Limits

Este benchmark sí incluye el coste canónico de:

- entrenamiento;
- métricas predictivas;
- interpretabilidad mínima canónica;
- persistencia de artifacts por run.

No pretende medir con precisión fina el overhead global de:

- materialización previa;
- reporting final;
- closeout total de campaña;

porque, a esta escala, ese overhead es secundario frente al coste agregado por run y puede tratarse aparte en el preflight final.

## Closeout

El bloque `15` se considera cerrado cuando:

- ya no dependemos de estimación teórica para hablar del coste de `F7`;
- existe una estimación observada y reproducible basada en la cadena canónica real;
- y el punto siguiente puede centrarse en `go / no-go`, no en adivinar la logística.
