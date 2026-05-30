# F7 Block 10 MLP Interpretability Rationale

## Decision

El bloque `10` queda cerrado con una capa canónica de interpretabilidad ligera para `MLP`.

La versión final de cierre incluye además el ajuste `10B`:

- la superficie primaria ya no se reporta en coordenadas `ilr_*`;
- las coordenadas `ilr_chem_*` se colapsan a `chem_*`;
- las coordenadas `ilr_phase_*` se colapsan a `phase_*`;
- los artefactos finales de campaña no deben contener nombres `ilr_*`.

Artefacto contractual:

- [config/f7_mlp_interpretability_contract_v1.yaml](../config/f7_mlp_interpretability_contract_v1.yaml)

## Qué queda fijado

Toda run `MLP` orientada a `F7` debe producir interpretabilidad mínima en espacio de predicción `raw`, con:

- `global`
- `per_class`
- `val` siempre
- `test` cuando `test_enabled=true`

La señal canónica de ranking es:

- `mean_abs_delta_pred_raw`

Y se persiste además:

- `mean_signed_delta_pred_raw`
- `sum_abs_delta_pred_raw`
- `std_abs_delta_pred_raw`
- `median_abs_delta_pred_raw`
- `p90_abs_delta_pred_raw`
- `p95_abs_delta_pred_raw`
- `stderr_abs_delta_pred_raw`
- `share_abs_importance`

Interpretación correcta de estas magnitudes:

- `mean_abs_delta_pred_raw` no es un porcentaje de aporte al target;
- representa el cambio medio absoluto en la predicción `raw` al perturbar esa feature;
- `share_abs_importance` sí da una lectura relativa útil, pero como porcentaje de masa total de sensibilidad dentro de una run/split, no como porcentaje causal del target.

## Método canónico

La interpretabilidad de `MLP` se fija como una perturbación ligera sobre el espacio real de entrada del modelo.

Para cada feature:

- se reemplaza por la media de `train` de esa feature dentro de la clase de la muestra;
- se vuelve a inferir;
- y se mide cuánto cambia la predicción en `raw`.

No se usa SHAP como capa canónica de campaña.

## Superficies de trabajo

La implementación final distingue tres superficies:

- `model_input_space`
  - features directas
  - `ilr_chem_*`
  - `ilr_phase_*`
- `latent_space`
  - solo para `flowpre_candidate_*`
- `final_semantic_space`
  - features directas passthrough
  - `chem_*`
  - `phase_*`

La superficie primaria para análisis downstream es solo `final_semantic_space`.

## Datasets `flowpre_candidate_*`

Cuando el `MLP` consume latentes de `FlowPre`, la interpretabilidad ya no se reporta en latentes como artefacto principal.

La política canónica pasa a ser:

- calcular primero la sensibilidad en latentes;
- proyectarla después a la superficie intermedia de input;
- y colapsar cualquier grupo `ILR` a componentes composicionales normalizados;
- y usar esa proyección como superficie principal para análisis estadístico y reporting.

Se guardan ambos artefactos:

- importancia latente
- importancia en `model_input_space`
- importancia final proyectada a features semánticas

## Por qué no sirve el `*_influence.json` histórico de `FlowPre`

El `*_influence.json` existente en `FlowPre` sirve como precedente útil, pero no como base canónica suficiente para `F7`, porque:

- usa una sola clase fija de referencia;
- no produce salida `per_class`;
- y por tanto no basta para la comparabilidad que requiere el análisis downstream de `MLP`.

Por eso la implementación canónica crea un cache nuevo de proyección `latent -> semantic_feature` condicionado por clase.

## Artefactos mínimos por run

Toda run `MLP` `F7` deja ahora, además de la persistencia del bloque `9`:

- `interpretability_summary.json`
- `input_feature_influence_global.csv`
- `input_feature_influence_per_class.csv`
- `feature_influence_global.csv`
- `feature_influence_per_class.csv`
- `top_features_global.csv`
- `top_features_per_class.csv`

Y cuando el input es `flowpre_candidate_*`:

- `latent_feature_influence_global.csv`
- `latent_feature_influence_per_class.csv`
- referencia al cache/projection manifest de `FlowPre`

Regla fuerte del cierre `10B`:

- `feature_influence_global.csv`
- `feature_influence_per_class.csv`
- `top_features_global.csv`
- `top_features_per_class.csv`

deben excluir nombres con prefijo `ilr_`.

Los artefactos `ILR` quedan solo como traza auxiliar en `input_feature_influence_*`.

## Valididad de campaña

Para `MLP` bajo `F7`:

- `campaign_valid` sigue siendo la validez predictiva/raw previa
- `campaign_valid_interpretability` valida esta capa nueva
- `campaign_valid_f7 = campaign_valid AND campaign_valid_interpretability`

Una run `MLP` F7 no cuenta como completamente válida si no puede producir esta interpretabilidad mínima o si el artefacto final todavía contiene coordenadas `ilr_*`.
