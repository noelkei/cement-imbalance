# Train-only FlowPre Finalists

## `trainonly_flowpre_candidate_1`

- linea: `experimental_train_only`
- rol semantico: primary train-only FlowGen prior for local downstream comparison
- `selection_role`: `candidate_trainonly_1`
- `selected_run_id`: `flowpre_trainonly_explore12_e09_hf224_l3_rq5_frq5_lr1e-3_mson_skoff_seed6769_v1`
- `selected_seed`: `6769`
- `cfg_signature`: `hf224|l3|rq1x5|frq5|lr1e-3|mson|skoff`
- dependencia upstream: `{"expected_flowgen_role": "primary train-only prior", "monitoring_policy": "train_only", "official_split_id": "init_temporal_processed_v1"}`
- rationale resumido:
  - Esta carpeta materializa el **prior principal** de FlowPre train-only para arrancar FlowGen train-only.
  - Rol operativo:
  - nombre lógico: candidate_trainonly_1
  - alias: primary
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/trainonly_flowpre_candidate_1.yaml`](../../config/finalists/config_snapshots/trainonly_flowpre_candidate_1.yaml)
  - manifiesto tracked: [`config/finalists/trainonly_flowpre_candidate_1.yaml`](../../config/finalists/trainonly_flowpre_candidate_1.yaml)

## `trainonly_flowpre_candidate_2`

- linea: `experimental_train_only`
- rol semantico: backup train-only FlowGen prior for local downstream comparison
- `selection_role`: `candidate_trainonly_2`
- `selected_run_id`: `flowpre_trainonly_top20_cfg11_hf256_l4_rq6_frq6_lr1e-3_mson_skoff_seed6769_v1`
- `selected_seed`: `6769`
- `cfg_signature`: `hf256|l4|rq1x6|frq6|lr1e-3|mson|skoff`
- dependencia upstream: `{"expected_flowgen_role": "backup train-only prior", "monitoring_policy": "train_only", "official_split_id": "init_temporal_processed_v1"}`
- rationale resumido:
  - Esta carpeta materializa el **prior backup** de FlowPre train-only para la fase FlowGen train-only.
  - Rol operativo:
  - nombre lógico: candidate_trainonly_2
  - alias: backup
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/trainonly_flowpre_candidate_2.yaml`](../../config/finalists/config_snapshots/trainonly_flowpre_candidate_2.yaml)
  - manifiesto tracked: [`config/finalists/trainonly_flowpre_candidate_2.yaml`](../../config/finalists/trainonly_flowpre_candidate_2.yaml)
