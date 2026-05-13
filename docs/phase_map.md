# Phase Map

Mapa operativo del proyecto hasta el punto actual real.

Reglas de lectura:

- `cerrada` = fase ya resuelta metodologicamente y con referencia vigente;
- `abierta/diferida` = fase aun no congelada;
- este documento describe estado actual, no una historia final cerrada del TFG.

| fase | estado | objetivo principal | artefactos principales | nota |
| --- | --- | --- | --- | --- |
| `F0` datos fuente, anonimización y preprocess | `cerrada/local` | partir del dataset industrial permitido y llevarlo a `processed` con superficie anonimizante y usable | `data/raw/`, `data/processed/df_processed.csv`, `data/preprocess.py` | material valioso pero local-only para git/publicacion |
| `F1` canon documental y frontera | `cerrada` | fijar que es canonico, historico, regenerable y sensible | `docs/project_context.md`, `docs/target_architecture.md`, `AGENTS.md` | la frontera practica se apoya ahora tambien en `docs/repo_visibility_matrix.md` |
| `F2` split temporal oficial | `cerrada` | congelar el split temporal canónico y su contrato minimo | `data/splits/official/init_temporal_processed_v1/manifest.json` y artefactos asociados | referencia metodologica vigente para todo lo downstream |
| `F3` cleaning split-aware y anti-leakage | `cerrada` | mover cleaning learned a `train` y dejar holdout en `flag-only` | `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/manifest.json` | referencia vigente de politica de cleaning |
| `F4` evaluacion canonica | `cerrada` | centralizar metricas, normalizacion y agregacion ligera | `evaluation/`, `evaluation/results.py`, `evaluation/metrics.py` | `test` sigue bloqueado por defecto |
| `F5` canonizacion de derivados y contrato experimental | `cerrada` | fijar espacio `dataset-level` vs `run-level` y materializar los `16` clasicos oficiales | `config/closure_contract_v1.yaml`, `data/sets/official/init_temporal_processed_v1/scaled/` | la parte FlowPre/FlowGen queda ligada pero no reabre esas fases |
| `F6a` cierre de `FlowPre` | `cerrada` | cerrar revalidate, exploraciones, reseed y finalistas | `outputs/models/official/flowpre_finalists/README.md`, `outputs/reports/f6_reseed_topcfgs_v4_continue/resume_summary.md`, `config/finalists/official_flowpre_*.yaml` | `rrmse`, `mvn`, `fair` quedan como upstreams; `candidate_1/2` como procedencia de `FlowGen` |
| `F6b` cierre de `FlowGen` | `cerrada` | cerrar exploracion oficial, reseed y winner final unico | `outputs/models/official/flowgen_finalist/README.md`, `outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json`, `config/finalists/official_flowgen_winner.yaml` | winner vigente: `flowgen_tpv1_c2_train_e03_seed2468_v1` |
| `F7` cierre experimental principal con `MLP` | `abierta/diferida` | comparar combinaciones finales de datasets con `MLP` bajo reglas justas y seleccion en `val` | pendiente | no se congela en esta iteracion |

Nota adicional de lectura:

- existe una rama experimental local `train_only` ya cerrada bajo `outputs/models/experimental/train_only/`;
- esa rama no reabre `F6a` ni `F6b`;
- su papel actual es alimentar decisiones downstream de `F7`, no convertirse en una fase canonica paralela del mapa principal.
- para backup ligero y lectura tracked, los cierres de `F6a/F6b` y de `train_only` tambien quedan resumidos bajo `config/finalists*` y `docs/finalists*`.

## Fuera de este mapa corto

- `F8` notebooks finales del TFG;
- `F9` limpieza final de la superficie publica del repo.

Esas fases siguen existiendo como trabajo posterior, pero este mapa se corta en `F7` para mantener foco en el estado actual real.
