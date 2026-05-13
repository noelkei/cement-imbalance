# Train-only FlowGen Finalist

## `trainonly_flowgen_winner`

- linea: `experimental_train_only`
- rol semantico: unique local train-only FlowGen finalist for downstream comparison
- `selection_role`: `unique_trainonly_finalist`
- `selected_run_id`: `flowgen_trainonly_tpv1_ct1_reseedfinal_r3a2_t06_clip125_seed15427_v1`
- `selected_seed`: `15427`
- `family_policy_id`: `R3A2_t06_clip125`
- dependencia upstream: `{"branch_id": "candidate_trainonly_1", "monitoring_policy": "train_only", "paired_flowpre_run_id": "flowpre_trainonly_explore12_e09_hf224_l3_rq5_frq5_lr1e-3_mson_skoff_seed6769_v1", "paired_flowpre_source_id": "flowpre__candidate_trainonly_1__init_temporal_processed_v1__v1", "raw_bundle_dataset_name": "df_input_cp_trainfit_overlap_cap1pct_holdoutflag_v1"}`
- motivo de seleccion: Selected as the best final seed inside the winning train-only family because it is the strongest balanced_trainonly closeout while staying high on x_priority and preserving the clip125 T06 regime that dominated the family-level reseed comparison.
- rationale resumido:
  - Esta carpeta no representa el winner canónico global del proyecto, sino el **finalista experimental único y vigente** de FlowGen train-only.
  - 1. se cerró la exploración train_only de FlowGen en varias rondas;
  - 2. se eligieron 4 cfg/familias finales para reseed por la misma lógica de ranking train_only usada en la shortlist previa;
  - 3. se ejecutó un panel de 5 seeds por familia (6769 + 4 seeds nuevas);
- capa local-only original:
  - config snapshot tracked: [`config/finalists/config_snapshots/trainonly_flowgen_winner.yaml`](../../config/finalists/config_snapshots/trainonly_flowgen_winner.yaml)
  - manifiesto tracked: [`config/finalists/trainonly_flowgen_winner.yaml`](../../config/finalists/trainonly_flowgen_winner.yaml)
