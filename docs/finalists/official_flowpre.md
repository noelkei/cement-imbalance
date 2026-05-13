# Official FlowPre Finalists

## `official_flowpre_rrmse`

- linea: `official`
- rol semantico: specialized scaler/upstream for dataset derivation under the rrmse_primary lens
- `selection_role`: `rrmse`
- `selected_run_id`: `flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1`
- `selected_seed`: `5678`
- `cfg_signature`: `hf256_l3_rq3_lr1e-4_mson_skoff`
- dependencia upstream: `{"cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1", "official_split_id": "init_temporal_processed_v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- rationale resumido:
  - Esta carpeta representa la lente rrmse_primary.
  - Propósito operativo:
  - usar un scaler FlowPre orientado a reconstrucción global equilibrada, con peso explícito sobre train_sum_rrmse, val_sum_rrmse, gap_val_train_sum y los ratios de balance entre media y desviación estándar.
  - Rol vigente:
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/official_flowpre_rrmse.yaml`](../../config/finalists/config_snapshots/official_flowpre_rrmse.yaml)
  - manifiesto tracked: [`config/finalists/official_flowpre_rrmse.yaml`](../../config/finalists/official_flowpre_rrmse.yaml)

## `official_flowpre_mvn`

- linea: `official`
- rol semantico: specialized scaler/upstream for dataset derivation under the mvn lens
- `selection_role`: `mvn`
- `selected_run_id`: `flowpre_rrmse_tpv1_rq5_seed5678_v1`
- `selected_seed`: `5678`
- `cfg_signature`: `hf256_l4_rq5_lr1e-3_mson_skoff`
- dependencia upstream: `{"cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1", "official_split_id": "init_temporal_processed_v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- rationale resumido:
  - Esta carpeta representa la lente mvn.
  - Propósito operativo:
  - usar un scaler FlowPre orientado a acercar el espacio transformado a un comportamiento más cercano a una normal multivariante, priorizando isotropía, skewness, kurtosis y consistencia geométrica.
  - Rol vigente:
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/official_flowpre_mvn.yaml`](../../config/finalists/config_snapshots/official_flowpre_mvn.yaml)
  - manifiesto tracked: [`config/finalists/official_flowpre_mvn.yaml`](../../config/finalists/official_flowpre_mvn.yaml)

## `official_flowpre_fair`

- linea: `official`
- rol semantico: specialized scaler/upstream for dataset derivation under the fair lens
- `selection_role`: `fair`
- `selected_run_id`: `flowprex4_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed5678_v1`
- `selected_seed`: `5678`
- `cfg_signature`: `hf192_l3_rq6_lr1e-3_mson_skoff`
- dependencia upstream: `{"cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1", "official_split_id": "init_temporal_processed_v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- rationale resumido:
  - Esta carpeta representa la lente fair.
  - Propósito operativo:
  - usar un scaler FlowPre que prioriza equilibrio entre clases en el espacio transformado, penalizando peores clases y dispersión desigual entre clases.
  - Rol vigente:
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/official_flowpre_fair.yaml`](../../config/finalists/config_snapshots/official_flowpre_fair.yaml)
  - manifiesto tracked: [`config/finalists/official_flowpre_fair.yaml`](../../config/finalists/official_flowpre_fair.yaml)

## `official_flowpre_candidate_1`

- linea: `official`
- rol semantico: primary official FlowGen work base with priorfit bias
- `selection_role`: `candidate_1`
- `selected_run_id`: `flowprers1_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed9101_v1`
- `selected_seed`: `9101`
- `cfg_signature`: `hf192_l3_rq6_lr1e-3_mson_skoff`
- dependencia upstream: `{"cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1", "official_split_id": "init_temporal_processed_v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- rationale resumido:
  - Esta carpeta no representa un scaler descriptivo, sino un finalista operativo para FlowGen.
  - Rol operativo:
  - candidate_1 es la base con sesgo más priorfit, pensada para servir como espacio de partida cuando interesa una combinación fuerte de ajuste previo y calidad de superficie en el ranking flowgencandidate_priorfit.
  - Estado vigente:
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/official_flowpre_candidate_1.yaml`](../../config/finalists/config_snapshots/official_flowpre_candidate_1.yaml)
  - manifiesto tracked: [`config/finalists/official_flowpre_candidate_1.yaml`](../../config/finalists/official_flowpre_candidate_1.yaml)

## `official_flowpre_candidate_2`

- linea: `official`
- rol semantico: secondary official FlowGen work base with robust/hybrid bias
- `selection_role`: `candidate_2`
- `selected_run_id`: `flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1`
- `selected_seed`: `2468`
- `cfg_signature`: `hf256_l4_rq5_lr1e-3_mson_skoff`
- dependencia upstream: `{"cleaning_policy_id": "trainfit_overlap_cap1pct_holdoutflag_v1", "official_split_id": "init_temporal_processed_v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- rationale resumido:
  - Esta carpeta no representa un scaler descriptivo, sino un finalista operativo para FlowGen.
  - Rol operativo:
  - candidate_2 es la base con sesgo más robust/hybrid, pensada para servir como espacio finalista cuando interesa estabilidad práctica y buen comportamiento conjunto en flowgencandidate_robust y flowgencandidate_hybrid.
  - Estado vigente:
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/official_flowpre_candidate_2.yaml`](../../config/finalists/config_snapshots/official_flowpre_candidate_2.yaml)
  - manifiesto tracked: [`config/finalists/official_flowpre_candidate_2.yaml`](../../config/finalists/official_flowpre_candidate_2.yaml)
