# Finalists Registry

Capa tracked y public-safe de finalistas/winners. Resume la verdad ligera que hoy vive materialmente en `outputs/`, sin copiar pesos ni artefactos pesados.

- indice machine-readable: [`config/finalists_registry.yaml`](../config/finalists_registry.yaml)
- dossiers humanos:
  - [`official_flowpre`](finalists/official_flowpre.md)
  - [`official_flowgen`](finalists/official_flowgen.md)
  - [`trainonly_flowpre`](finalists/trainonly_flowpre.md)
  - [`trainonly_flowgen`](finalists/trainonly_flowgen.md)

| artifact_id | line | model_family | selection_role | selected_run_id | seed | snapshot |
| --- | --- | --- | --- | --- | ---: | --- |
| `official_flowpre_rrmse` | `official` | `flowpre` | `rrmse` | `flowprex2_rrmse_tpv1_hf256_l3_rq3_lr1e-4_mson_skoff_seed5678_v1` | `5678` | [`config/finalists/config_snapshots/official_flowpre_rrmse.yaml`](../config/finalists/config_snapshots/official_flowpre_rrmse.yaml) |
| `official_flowpre_mvn` | `official` | `flowpre` | `mvn` | `flowpre_rrmse_tpv1_rq5_seed5678_v1` | `5678` | [`config/finalists/config_snapshots/official_flowpre_mvn.yaml`](../config/finalists/config_snapshots/official_flowpre_mvn.yaml) |
| `official_flowpre_fair` | `official` | `flowpre` | `fair` | `flowprex4_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed5678_v1` | `5678` | [`config/finalists/config_snapshots/official_flowpre_fair.yaml`](../config/finalists/config_snapshots/official_flowpre_fair.yaml) |
| `official_flowpre_candidate_1` | `official` | `flowpre` | `candidate_1` | `flowprers1_rrmse_tpv1_hf192_l3_rq6_lr1e-3_mson_skoff_seed9101_v1` | `9101` | [`config/finalists/config_snapshots/official_flowpre_candidate_1.yaml`](../config/finalists/config_snapshots/official_flowpre_candidate_1.yaml) |
| `official_flowpre_candidate_2` | `official` | `flowpre` | `candidate_2` | `flowprers1_rrmse_tpv1_hf256_l4_rq5_lr1e-3_mson_skoff_seed2468_v1` | `2468` | [`config/finalists/config_snapshots/official_flowpre_candidate_2.yaml`](../config/finalists/config_snapshots/official_flowpre_candidate_2.yaml) |
| `official_flowgen_winner` | `official` | `flowgen` | `unique_official_finalist` | `flowgen_tpv1_c2_train_e03_seed2468_v1` | `2468` | [`config/finalists/config_snapshots/official_flowgen_winner.yaml`](../config/finalists/config_snapshots/official_flowgen_winner.yaml) |
| `trainonly_flowpre_candidate_1` | `experimental_train_only` | `flowpre` | `candidate_trainonly_1` | `flowpre_trainonly_explore12_e09_hf224_l3_rq5_frq5_lr1e-3_mson_skoff_seed6769_v1` | `6769` | [`config/finalists/config_snapshots/trainonly_flowpre_candidate_1.yaml`](../config/finalists/config_snapshots/trainonly_flowpre_candidate_1.yaml) |
| `trainonly_flowpre_candidate_2` | `experimental_train_only` | `flowpre` | `candidate_trainonly_2` | `flowpre_trainonly_top20_cfg11_hf256_l4_rq6_frq6_lr1e-3_mson_skoff_seed6769_v1` | `6769` | [`config/finalists/config_snapshots/trainonly_flowpre_candidate_2.yaml`](../config/finalists/config_snapshots/trainonly_flowpre_candidate_2.yaml) |
| `trainonly_flowgen_winner` | `experimental_train_only` | `flowgen` | `unique_trainonly_finalist` | `flowgen_trainonly_tpv1_ct1_reseedfinal_r3a2_t06_clip125_seed15427_v1` | `15427` | [`config/finalists/config_snapshots/trainonly_flowgen_winner.yaml`](../config/finalists/config_snapshots/trainonly_flowgen_winner.yaml) |

## Regla operativa

- esta capa tracked es la referencia ligera canónica de finalists/winners;
- `outputs/` sigue siendo la fuente local pesada/original;
- para restore completo siguen haciendo falta `data/raw/`, `config/local/` y los outputs promovidos.
