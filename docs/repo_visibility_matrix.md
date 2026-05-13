# Repo Visibility Matrix

Este documento fija la frontera operativa entre material:

- `mostrable/publicable`
- `local/privado/NDA`
- `mixto/separar`

Su objetivo no es convertir todo el repo en publicable, sino dejar claro:

- qué sí se puede enseñar o subir;
- qué debe quedarse local;
- qué necesita doble fichero, plantilla o versión separada.

## Regla general

- Si una pieza contiene nombres reales, mappings privados, fechas exactas sensibles, valores absolutos industriales, rutas locales o artefactos generados con datos reales, se trata como `local/privado/NDA`.
- Si una pieza mezcla responsabilidad pública y dependencia privada, se trata como `mixto/separar`.
- Lo `mostrable/publicable` no implica que sea canon final del TFG; puede seguir siendo histórico o auxiliar.

## Matriz de visibilidad

| superficie | estado | decision practica | nota |
| --- | --- | --- | --- |
| `README.md` | `mostrable/publicable` | mantener versionable y alineado con el estado real | no debe vender fases abiertas como cerradas ni viceversa |
| `docs/` canonicos y de apoyo | `mostrable/publicable` | mantener versionables, evitando valores o nombres sensibles | deben dejar claro que partes siguen abiertas |
| `docs/finalists_registry.md`, `docs/finalists/` y `docs/backup_and_restore.md` | `mostrable/publicable` | mantener versionables como capa tracked ligera de cierre y restore | resumen finalists/winners y politica de recuperacion sin subir artifacts pesados |
| `AGENTS.md` | `mostrable/publicable` | mantener versionable como politica del repo | puede incluir reglas operativas, no datos sensibles |
| `config/type_mapping.yaml` | `mixto/separar` | dejar en repo una copia publica segura y usar overlay local privado para operar con datos reales | la version local vive bajo `config/local/` |
| `config/column_groups.yaml` | `mixto/separar` | dejar en repo una copia publica segura y usar overlay local privado para operar con datos reales | la version local vive bajo `config/local/` |
| `config/cleaning_contract.yaml` | `mixto/separar` | dejar en repo un contract publico seguro y usar overlay local privado para la operativa real del raw | la version local vive bajo `config/local/` y evita hardcodes sensibles en `data/cleaning.py` |
| `config/local/` | `local/privado/NDA` | ignorar en git | contiene overlays privados requeridos para la operativa local |
| `config/mlp.yaml`, `config/flow_pre.yaml`, `config/flowgen.yaml`, `config/closure_contract_v1.yaml`, `config/mlp_closure_base_v1.yaml` | `mostrable/publicable` | mantener versionables tras revisar que no incluyan paths o referencias privadas | son configs base, no mappings sensibles |
| `config/finalists_registry.yaml`, `config/finalists/`, `config/finalists/config_snapshots/` | `mostrable/publicable` | mantener versionables como export tracked y sanitizado de la metadata de cierre | contiene manifests ligeros, snapshots de config y refs `local_only`, no pesos ni datasets |
| `data/raw/` | `local/privado/NDA` | ignorar en git | raw industrial y nombres de ficheros no publicables |
| `data/processed/`, `data/cleaned/`, `data/splits/`, `data/sets/` | `local/privado/NDA` | ignorar en git | contienen datos reales, manifests detallados y artefactos derivados no subibles |
| artefactos generados bajo `outputs/` | `local/privado/NDA` | ignorar en git | artefactos generados, logs, checkpoints, estudios y tablas locales |
| `notebooks/` | `mixto/separar` | no subir por defecto como producto final; revisar individualmente si alguna notebook se quiere mostrar | hoy son historicas u operativas |
| `training_scripts/` | `mixto/separar` | mantener como historico/semioperativo; revisar individualmente antes de mostrar | no son la API canonica |
| `training/`, `evaluation/`, `models/`, `losses/`, `data/*.py` | `mostrable/publicable` | mantener versionables si no exponen detalles sensibles adicionales | el codigo puede mostrarse aunque dependa de datos locales |
| `scripts/export_finalists_registry.py` | `mostrable/publicable` | mantener versionable como entrypoint canónico del export tracked | su rol es regenerar la capa ligera desde `outputs/` locales |

## Casos mixtos que requieren separacion

### 1. Mappings sensibles

- tracked/publico: `config/type_mapping.yaml`, `config/column_groups.yaml`
- local/privado: `config/local/type_mapping.yaml`, `config/local/column_groups.yaml`
- decision: el loader debe preferir `config/local/` cuando exista.

### 2. Contract operativo del raw

- tracked/publico: `config/cleaning_contract.yaml`
- local/privado: `config/local/cleaning_contract.yaml`
- decision: el repo tracked conserva un contract public-safe; la operativa real del preprocess privado no debe depender de hardcodes sensibles en `data/cleaning.py`

### 3. Notebooks y scripts historicos

- no se consideran publicables por defecto solo por estar en el repo;
- si alguna pieza se quiere mostrar, debe revisarse por contenido, outputs incrustados y rutas.

### 4. Metadata ligera de cierre

- `outputs/` sigue siendo la fuente local pesada/original de finalists, manifests completos, rationales y resultados materiales;
- `config/finalists*` y `docs/finalists*` son la exportacion tracked, pequena y public-safe de esa verdad de cierre;
- esta capa no sustituye el backup externo de `data/raw/`, `config/local/` ni de los outputs promovidos.

## Checklist minima antes de subir algo

1. comprobar si la pieza contiene nombres reales, fechas exactas o valores absolutos sensibles;
2. comprobar si depende de `config/local/`, `data/` real u `outputs/` locales;
3. comprobar si es canon actual, historico o semioperativo;
4. si es mixta, decidir entre version publica segura, plantilla o exclusion completa;
5. si hay duda, tratarla como `local/privado/NDA`.
