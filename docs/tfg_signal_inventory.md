# TFG Signal Inventory

Inventario de artefactos y superficies que ya aportan señal para redactar mejor la memoria, sin asumir que todo este cerrado ni que todo sea publicable.

Reglas de lectura:

- `estado` describe si el artefacto esta vigente, historico o pendiente de cierre posterior;
- `visibilidad` indica si es mostrable, local-only o si requiere revision;
- `uso sugerido` describe para que puede servir al redactar el TFG.

| fase | artefacto o superficie | tipo | estado | visibilidad | uso sugerido |
| --- | --- | --- | --- | --- | --- |
| `F1` | `docs/project_context.md` | contexto canonico | vigente | `mostrable/publicable` | fijar framing del problema y restricciones metodologicas |
| `F1` | `docs/target_architecture.md` | arquitectura objetivo | vigente | `mostrable/publicable` | justificar que el cierre se apoya en una arquitectura minima y no en un refactor total |
| `F1` | `docs/repo_visibility_matrix.md` | frontera publico/local | vigente | `mostrable/publicable` | documentar criterio NDA y que parte del repo se ensena realmente |
| `F1` | `docs/finalists_registry.md` | indice humano tracked | vigente | `mostrable/publicable` | resumir winners/finalists cerrados sin depender de `outputs/` |
| `F1` | `config/finalists_registry.yaml` | indice machine-readable tracked | vigente | `mostrable/publicable` | reconstruir ids, seeds, configs y refs locales de cierre |
| `F1` | `docs/backup_and_restore.md` | politica de restore | vigente | `mostrable/publicable` | dejar claro que recupera GitHub y que sigue requiriendo backup externo |
| `F2` | `data/splits/official/init_temporal_processed_v1/manifest.json` | contrato de split | vigente | `local-only` | fuente tecnica para explicar la politica temporal y los conteos por split |
| `F2` | `data/splits/official/init_temporal_processed_v1/plots/` | figuras ya materializadas | vigente | `revision antes de mostrar` | candidatas para explicar cobertura temporal, mix por clase y drift |
| `F3` | `data/cleaned/official/init_temporal_processed_v1/trainfit_overlap_cap1pct_holdoutflag_v1/manifest.json` | contrato de cleaning | vigente | `local-only` | soporte para explicar `fit on train`, `flag-only` y guardrails |
| `F5` | `config/closure_contract_v1.yaml` | contrato experimental | vigente | `mostrable/publicable` | explicar separacion entre `dataset-level` y `run-level` |
| `F5` | `config/mlp_closure_base_v1.yaml` | base config congelada | vigente como prior | `mostrable/publicable` | soporte para futuras decisiones de `MLP`, sin congelar `F7` aun |
| `F6a` | `outputs/models/official/flowpre_finalists/README.md` | rationale de cierre | vigente | `local-only` | narrar como se paso de superficie oficial a finalistas por lente |
| `F6a` | `outputs/reports/f6_reseed_topcfgs_v4_continue/resume_summary.md` | resumen de reseed | vigente | `local-only` | aportar conteos, cierre operacional y evidencia de completitud |
| `F6a` | `outputs/models/official/flowpre_finalists/*/RATIONALE.md` | rationales por finalista | vigente | `local-only` | recuperar tradeoffs por lente y por candidato |
| `F6a` | `config/finalists/official_flowpre_*.yaml` + `config/finalists/config_snapshots/official_flowpre_*.yaml` | manifests ligeros tracked | vigente | `mostrable/publicable` | llevar fuera de `outputs/` la metadata minima de seleccion y recreacion |
| `F6b` | `outputs/models/official/flowgen_finalist/README.md` | resumen de winner final | vigente | `local-only` | explicar por que el winner final vigente ya no es el pre-reseed |
| `F6b` | `outputs/models/official/flowgen_finalist/RATIONALE.md` | rationale de seleccion | vigente | `local-only` | justificar la seleccion final dentro de la familia `E03` |
| `F6b` | `outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json` | resumen machine-readable | vigente | `local-only` | soporte tecnico para tablas y referencias cruzadas |
| `F6b` | `outputs/models/official/flowgen/campaign_summaries/post_reseed/FLOWGEN_POST_RESEED_FINAL_DECISION.md` | cierre agregativo | vigente | `local-only` | soporte para una tabla o figura de decision final |
| `F6b` | `config/finalists/official_flowgen_winner.yaml` + `config/finalists/config_snapshots/official_flowgen_winner.yaml` | export tracked del winner | vigente | `mostrable/publicable` | defender el winner con config sanitizada, metrics summary y refs locales |
| rama experimental `train_only` | `outputs/models/experimental/train_only/flowpre_finalists/README.md` | cierre local de bases `FlowPre train_only` | vigente | `local-only` | explicar que priors locales quedaron listos para la rama generativa experimental |
| rama experimental `train_only` | `outputs/models/experimental/train_only/flowgen_finalist/README.md` | cierre local de `FlowGen train_only` | vigente | `local-only` | documentar el winner local y su rol downstream, sin confundirlo con el winner official |
| rama experimental `train_only` | `outputs/models/experimental/train_only/flowgen_finalist/RATIONALE.md` | rationale de seleccion local | vigente | `local-only` | recuperar por que gano la familia final `train_only` y como se eligio la seed representante |
| rama experimental `train_only` | `outputs/models/experimental/train_only/flowgen_finalist/flowgen_trainonly_final_selection_manifest.json` | resumen machine-readable | vigente | `local-only` | soporte tecnico para tablas, apendice metodologico o discusion downstream |
| rama experimental `train_only` | `config/finalists/trainonly_*.yaml` + `config/finalists/config_snapshots/trainonly_*.yaml` | export tracked ligero local | vigente | `mostrable/publicable` | distinguir la linea local, conservar su metadata y no depender solo de `outputs/` |
| transversal | `docs/phase_map.md` | mapa de fases | vigente | `mostrable/publicable` | ayudar a estructurar la memoria sin cerrar fases abiertas |
| transversal | `docs/artifact_lineage.md` | trazabilidad | vigente | `mostrable/publicable` | conectar inputs, procesos, outputs y referencias vigentes |

## Señal ya util, pero no publica por defecto

- manifests y tablas de `data/` oficial;
- rationales y summaries bajo `outputs/models/official/` y `outputs/reports/`;
- rationales, manifests y rankings bajo `outputs/models/experimental/train_only/` cuando se quieran usar para justificar la rama experimental o un apendice metodologico;
- cualquier notebook con outputs incrustados o referencias a datos reales.

## Señal pendiente o aun no cerrada

- `F7` comparacion final con `MLP`;
- shortlist final de combinaciones ya con inputs oficiales cerrados y posible inclusion de la rama local `train_only`;
- tablas y figuras finales derivadas de esa comparacion.

Este inventario debe crecer con artefactos de alta señal, no con ruido operativo.
