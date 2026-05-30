# F7 Block 14 Main Analysis Grammar Rationale

## Decision

El bloque `14` queda cerrado como el freeze metodológico de la lectura estadística principal de `F7`.

Este bloque no implementa todavía la capa analítica final ni la shortlist del TFG. Su función es fijar, antes de la campaña grande, la gramática canónica con la que deberán leerse los resultados para que:

- la campaña no produzca solo abundancia de métricas;
- el análisis posterior no dependa de notebooks ad hoc;
- y la inferencia final quede alineada con las restricciones metodológicas ya fijadas en `F2`, `F3`, `F4`, `F12` y `F13`.

Documentos canónicos de este cierre:

- [docs/f7_statistical_analysis_plan.md](f7_statistical_analysis_plan.md)
- [docs/f7_statistical_analysis_plan_structured.md](f7_statistical_analysis_plan_structured.md)
- [docs/f7_statistical_analysis_spec.yaml](f7_statistical_analysis_spec.yaml)

## What Is Frozen Now

### Selection vs final reporting

Queda fijado que:

- `val` gobierna la shortlist y cualquier decisión pre-freeze;
- `test` no entra en la iteración de diseño;
- una vez la campaña y la shortlist estén congeladas, el análisis final debe reportar `val` y `test` como superficies co-principales del readout final.

Esto preserva disciplina anti-leakage sin degradar la lectura final del problema real.

### Observational and aggregate units

Queda fijado que:

- la unidad observacional principal es la `run` individual por seed;
- la unidad agregada principal es `lineage_trial_group_id`.

Esto deja alineadas la capa descriptiva, el pooling por extensiones, el bootstrap pareado y los modelos con estructura repetida.

### Main predictive backbone

La métrica principal de backbone queda fijada como:

- `raw_real macro rrmse`

pero no se permitirá leerla aislada como argumento principal. Debe ir obligatoriamente acompañada por la capa class-wise y Simpson:

- `per_class_rrmse`
- `worst_class_rrmse`
- `class_dispersion_rrmse`
- `simpson_gap_rrmse`

### Statistical family and comparability

La familia estadística comparable principal se leerá dentro del panel congelado por:

- mismo `lineage_trial_group_id` para pooling y agregación;
- contrasts pareados solo cuando las variantes compartan el resto del contexto estructural;
- `XGBoost` mantenido como baseline acotada dentro del mismo marco analítico, no como subestudio paralelo con grid estadístico propio.

## Methodological Backbone

La jerarquía metodológica congelada es:

1. descriptivo multi-seed con intervalos y percentiles;
2. comparaciones `paired / blocked` cuando el diseño lo permita;
3. `mixed-effects` o regresión con estructura repetida como capa principal de efectos e interacciones;
4. `ANOVA` solo como capa de apoyo sobre modelos ajustados;
5. `post-hoc` solo para contrasts dirigidos y estructuralmente motivados;
6. checks de sensibilidad no paramétricos y por permutación cuando hagan falta.

Queda explícitamente descartado como argumento principal:

- `ANOVA` omnibus aislada;
- tablas `all-vs-all`;
- ranking guiado solo por `p-values`.

## Confirmatory Contrast Policy

Se congela la necesidad de un `contrast registry` explícito con, como mínimo:

- contrasts confirmatorios;
- contrasts secundarios;
- contrasts exploratorios.

La política metodológica fijada es:

- contrasts confirmatorios con ajuste tipo `Holm` por defecto;
- contrasts exploratorios con control `FDR` por defecto;
- reporting siempre con tamaño de efecto e incertidumbre, no solo significancia.

## Simpson Layer

El problema de Simpson queda congelado como capa principal del análisis, no como apéndice.

La hipótesis estructural a evaluar es:

- una variante puede parecer buena en agregado;
- incluso inflar `R²`;
- y aun así comportarse mal por clase o interpolar entre clases.

Por eso el análisis final deberá incluir:

- macro;
- per-class;
- worst-class;
- dispersión entre clases;
- y, cuando sea factible, descomposición `within-class / between-class`.

## Interpretability Defaults

Quedan fijados como defaults:

- superficie transversal primaria:
  - `semantic_bridge_perturbation`
- superficies auxiliares:
  - `xgb_native_shap`
  - `mlp_flowpre_native_latent_perturbation`
- `top-k` principal para estabilidad:
  - `k = 10`

Esto no impide ampliar análisis auxiliares posteriores, pero sí fija una base comparable y reproducible.

## Practical Significance

El bloque `14` también congela que el análisis final no debe depender solo de significancia estadística.

Debe existir una capa de:

- superioridad;
- no inferioridad;
- diferencia operacionalmente irrelevante.

El umbral mínimo de relevancia práctica para `rrmse` queda reconocido como necesario, pero todavía no se fija numéricamente aquí. Se congela como:

- `minimum_practical_effect_rrmse = TBD_before_final_reporting`

Esto deja el hueco visible, versionado y controlado, en vez de implícito.

## Why This Is The Cleanest Freeze

Esta implementación del bloque `14` es la más limpia porque:

- no reabre la infraestructura ya cerrada en `13`;
- no intenta adelantar la shortlist ni la inferencia final sin campaña grande;
- deja la disciplina de selección y confirmación explícita;
- fija una lectura estadística fuerte antes del lanzamiento masivo;
- y convierte el plan estadístico en contrato revisable, no en intención difusa.

## Closeout

El bloque `14` se considera cerrado cuando:

- la gramática del análisis principal ya no depende de preguntas abiertas básicas;
- los documentos estructurados y machine-readable reflejan la misma postura metodológica;
- y el siguiente trabajo ya puede centrarse en benchmark, preflight y ejecución, no en redefinir cómo se leerán los resultados.
