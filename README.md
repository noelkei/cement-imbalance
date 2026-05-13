# Cement Imbalance

TFG de machine learning aplicado a prediccion de `init` en cemento bajo `distribution shift`, con foco en split temporal oficial, control de leakage, reproducibilidad y una superficie de repo clara en la parte que hoy es mostrable.

## Estado actual

- `FlowPre` esta cerrado como fase experimental y de seleccion bajo el split oficial `init_temporal_processed_v1`.
- `FlowGen` tambien esta cerrado y ya tiene winner final promovido en `outputs/models/official/flowgen_finalist/`.
- existe ademas una rama experimental local `train_only`, tambien ya cerrada localmente, con artefactos promovidos bajo `outputs/models/experimental/train_only/`.
- El siguiente frente metodologico pendiente del proyecto es el cierre downstream con `MLP`, pero `F7` no se congela en esta iteracion.

## Superficie visible del repo

- `docs/` es la fuente de verdad narrativa y metodologica.
- `docs/finalists_registry.md` y `config/finalists_registry.yaml` forman la capa tracked ligera de finalistas/winners para backup y trazabilidad.
- `config/finalists/` contiene manifests ligeros y snapshots de config sanitizados exportados desde los artifacts locales promovidos.
- `docs/backup_and_restore.md` fija que recupera GitHub por si solo y que sigue requiriendo backup externo.
- `config/type_mapping.yaml`, `config/column_groups.yaml` y `config/cleaning_contract.yaml` son copias tracked public-safe o contractuales; las versiones privadas operativas viven bajo `config/local/`.
- `data/raw/`, `data/processed/`, `data/cleaned/`, `data/splits/`, `data/sets/` y los artefactos generados bajo `outputs/` se tratan como superficies locales para git/publicacion.
- La referencia practica de esta frontera vive en `docs/repo_visibility_matrix.md`.

## Fuente de verdad

Leer primero:

- `docs/repo_visibility_matrix.md`
- `docs/phase_map.md`
- `docs/artifact_lineage.md`
- `docs/project_context.md`
- `docs/implementation_status.md`
- `docs/target_architecture.md`
- `docs/target_architecture_north_star.md`

Si codigo, notebooks o artefactos historicos contradicen esos documentos, manda `docs/`.

## Referencias vigentes ahora

- mostrables en el repo:
  - `docs/phase_map.md`
  - `docs/artifact_lineage.md`
  - `docs/tfg_signal_inventory.md`
  - `docs/repo_visibility_matrix.md`
  - `docs/finalists_registry.md`
  - `config/finalists_registry.yaml`
  - `docs/backup_and_restore.md`
- cierres metodologicos disponibles en local, pero no asumidos como superficie publica:
  - `outputs/models/official/flowpre_finalists/README.md`
  - `outputs/reports/f6_reseed_topcfgs_v4_continue/resume_summary.md`
  - `outputs/models/official/flowgen_finalist/README.md`
  - `outputs/models/official/flowgen_finalist/RATIONALE.md`
  - `outputs/models/official/flowgen_finalist/flowgen_final_selection_manifest.json`

## Que no asumir

- no asumir que los artefactos locales de `outputs/` o `data/` son publicables por defecto;
- no asumir que notebooks y `training_scripts/` son API canonica;
- no asumir que `FlowGen` sigue abierto: su estado vigente es `cerrado`, con winner final official ya promovido y una rama `train_only` cerrada solo como linea local experimental.
