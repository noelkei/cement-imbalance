# Official FlowGen Winner

## `official_flowgen_winner`

- linea: `official`
- rol semantico: unique official FlowGen winner promoted for downstream comparison
- `selection_role`: `unique_official_finalist`
- `selected_run_id`: `flowgen_tpv1_c2_train_e03_seed2468_v1`
- `selected_seed`: `2468`
- `family_policy_id`: `E03`
- dependencia upstream: `{"branch_id": "candidate_2", "paired_flowpre_run_id": "flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1", "paired_flowpre_source_id": "flowpre__candidate_2__init_temporal_processed_v1__v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- motivo de seleccion: Selected as the best final seed inside the winning E03 family because it is the strongest temporal-closeout compromise: best ValTemporalGuard, best classic, best ratio25, second-best ratio50 and Q3peak, while staying top-3 in TrainStrong60 and banded.
- rationale resumido:
  - Esta carpeta no representa una familia exploratoria ni un winner operativo provisional.
  - Representa el finalista oficial único y vigente de FlowGen.
  - 1. la exploración oficial de FlowGen se cerró primero con una selección pre-reseed;
  - 2. después se ejecutó el reseed final sobre 5 cfg candidatas;
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/official_flowgen_winner.yaml`](../../config/finalists/config_snapshots/official_flowgen_winner.yaml)
  - manifiesto tracked: [`config/finalists/official_flowgen_winner.yaml`](../../config/finalists/official_flowgen_winner.yaml)
