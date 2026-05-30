# F7 Block 11 XGBoost Interpretability Rationale

## Decision

El bloque `11` queda cerrado con una capa canónica de interpretabilidad para `XGBoost` con dos niveles obligatorios por run:

- capa nativa de familia basada en `SHAP` con `TreeExplainer`;
- capa puente de comparabilidad cruzada basada en perturbación ligera sobre el target `raw`.

Artefacto contractual:

- [config/f7_xgb_interpretability_contract_v1.yaml](../config/f7_xgb_interpretability_contract_v1.yaml)

## Qué queda fijado

Toda run `XGBoost` orientada a `F7` debe producir interpretabilidad mínima con:

- `global`
- `per_class`
- `val` siempre
- `test` cuando la run es `holdout_run`

La superficie canónica de features es exactamente la del modelo:

- `type_0`, `type_1`, `type_2`
- features numéricas raw en el orden fijo del dataset `XGBoost`

No se filtran ni se proyectan features en la capa contractual.

## Capa SHAP

La capa nativa de familia usa:

- `TreeExplainer`
- `mean_abs_shap` como métrica principal
- `mean_signed_shap` como métrica auxiliar

Y persiste además:

- `sum_abs_shap`
- `std_abs_shap`
- `median_abs_shap`
- `p90_abs_shap`
- `p95_abs_shap`
- `stderr_abs_shap`
- `share_abs_importance`

También se persiste `expected_value` por split.

No se guardan matrices completas de SHAP por muestra para todas las runs de campaña.

## Capa puente con MLP

Además de SHAP, toda run `XGBoost` produce una capa de perturbación ligera equivalente a la usada en `MLP`:

- baseline por media de `train` condicionada por clase;
- perturbación feature a feature en el espacio real de entrada del modelo;
- comparación en `raw target space`;
- `mean_abs_delta_pred_raw` como métrica principal;
- `mean_signed_delta_pred_raw` como métrica auxiliar.

Se persisten las mismas estadísticas de dispersión y `share_abs_importance` que en la capa SHAP.

## Gramática de comparabilidad

La comparabilidad entre familias queda explícita así:

- `MLP`:
  - capa canónica de familia = `mean_abs_delta_pred_raw`
- `XGBoost`:
  - capa canónica de familia = `mean_abs_shap`
- ambas familias:
  - comparten una capa puente de perturbación basada en `mean_abs_delta_pred_raw`

Por tanto:

- la comparación cruzada fuerte entre familias debe apoyarse en la capa puente;
- la lectura nativa de cada familia sigue siendo válida y preferente dentro de su propia familia;
- no debe afirmarse que `SHAP` y perturbación tienen la misma unidad numérica o el mismo significado causal.

## Valididad de campaña

Para `XGBoost` bajo `F7`:

- `campaign_valid` sigue representando la validez predictiva/raw previa;
- `campaign_valid_interpretability` exige que existan y validen ambas capas;
- `campaign_valid_f7 = campaign_valid AND campaign_valid_interpretability`

Una run `XGBoost` `F7` no cuenta como completamente válida si falla:

- la capa SHAP mínima;
- o la capa puente mínima;
- o la persistencia/validación contractual de cualquiera de las dos.

## Coste observado

La validación empírica corta de `10` runs `holdout_run` sobre los `4` datasets canónicos de `XGBoost` dejó este orden de magnitud:

- training medio por run: ~`0.80s`
- interpretabilidad total media por run: ~`0.32s`
- de ese coste interpretativo:
  - `SHAP`: ~`0.23s`
  - perturbación puente: ~`0.07s`
- runtime total medio por run: ~`1.26s`

Lectura práctica:

- `XGBoost` queda muy barato frente a `MLP`;
- la capa puente con `MLP` apenas penaliza;
- `SHAP` domina la parte interpretativa, pero su coste sigue siendo bajo y asumible para la campaña completa.
