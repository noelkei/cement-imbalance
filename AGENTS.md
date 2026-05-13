# AGENTS.md

## Fuente de verdad

Este proyecto se gobierna primero por estos documentos:

- `docs/project_context.md`: estado actual, contexto metodológico y restricciones reales.
- `docs/target_architecture.md`: arquitectura objetivo implementable para cierre del TFG y preparación del repo público.
- `docs/repo_visibility_matrix.md`: frontera práctica entre material mostrable, local/privado y mixto.

Si el código, notebooks o artefactos históricos contradicen esos documentos, manda `docs/`.

## Objetivo del proyecto

Cerrar un TFG de ML aplicado a un problema industrial real: predecir `init` de cemento de forma metodológicamente defendible, con foco en shift temporal, control de leakage, reproducibilidad y preparación de una superficie de repo seria en la parte que realmente sea mostrable.

## Prioridades metodológicas

Priorizar siempre, en este orden:

- reproducibilidad;
- anti-leakage;
- split temporal oficial por encima de splits aleatorios históricos;
- selección en `val` y confirmación final única en `test`;
- comparaciones justas entre variantes;
- cierre útil del TFG por encima de refactors cosméticos.

## Reglas de anti-leakage

- El split oficial se define antes de tuning, selección y comparación final.
- Todo lo que aprende parámetros se ajusta solo con `train`.
- `val` se usa para selección de variantes y decisiones intermedias.
- `test` no se usa para iterar.
- La limpieza estadística, escalados, transforms, `FlowPre`, selección de outliers y cualquier normalización aprendida deben fittear en `train` y aplicarse después a `val/test`.
- El preprocess global puede incluir filtros de calidad de dominio aceptados aguas arriba del split oficial, como el control de `sum_chem` y `sum_phase`, siempre que se documenten como tales y no se confundan con cleaning estadístico aprendido.
- En la operativa canónica, `train_*` no debe calcular ni guardar métricas de `test` por defecto; `test` solo se habilita con opt-in explícito.
- No promover resultados de notebooks o artefactos antiguos como válidos sin verificar su política de split y leakage.

## Frontera público vs privado/local

Distinguir siempre entre material `publicable` y material `local-only`.

Tratar `config/type_mapping.yaml`, `config/column_groups.yaml` y `config/cleaning_contract.yaml` como piezas mixtas:

- la copia tracked en `config/` debe ser pública segura y no revelar nombres o semántica sensible;
- la copia operativa real debe vivir en `config/local/` y tratarse como `local-only` / `NDA`;
- no asumir que la versión tracked reproduce los valores o nombres reales usados en local.

En el caso de `config/cleaning_contract.yaml`, esto incluye especialmente:

- nombre real del raw source;
- nombres reales de columnas raw operativas;
- nombres reales de tipos, procesos o filtros de arranque;
- columnas sensibles que se excluyen en la operativa privada.

Reglas:

- no asumir que esos archivos son publicables;
- si hacen falta para operar en local, deben vivir fuera del repo público o bajo `.gitignore`;
- si se necesita una versión pública, crear equivalente anonimizado o contractual, no exponer la privada;
- para piezas mixtas, preferir tracked copy pública segura en `config/` y overlay operativo privado en `config/local/`;
- cuando exista material sensible con equivalente público, preferir la versión pública anonimizada o documentada antes que exponer la versión local;
- no inferir que todo `data/` es privado por defecto, pero sí revisar riesgo de leakage de nombres, mappings, clases, rutas o detalles industriales antes de publicar cualquier transformación.

## Canónico vs regenerable vs histórico

Tratar como `canónico`:

- `docs/`
- código fuente mantenido en `data/`, `models/`, `losses/`, `training/`, `evaluation/`
- configs base no sensibles y mantenidas a mano
- la política oficial de split y evaluación que se vaya cerrando
- el bundle raw oficial versionado bajo `data/sets/official/<split_id>/raw/<dataset_name>/`

Tratar como `regenerable` salvo promoción explícita:

- `data/processed/`
- `data/cleaned/`
- `data/splits/`
- `data/sets/`
- checkpoints, logs, samples, outputs intermedios
- configs generadas por sweeps o runners masivos

Tratar como `histórico` hasta revisión puntual:

- notebooks operativas antiguas
- scripts versionados antiguos de `training_scripts/`
- `outputs/sweeps/`, `outputs/retrained/`, `outputs/retrained_v2/`, `outputs/logs/`
- comparativas o variantes fuera de la shortlist final
- datasets derivados heredados del split shuffled o sin manifest canónico completo

## Qué leer primero

Priorizar lectura de:

- `docs/project_context.md`
- `docs/target_architecture.md`
- `data/*.py`
- `evaluation/*.py`
- `models/*.py`
- `training/*.py`
- `losses/*.py`
- `config/*.yaml` canónicos y no generados
- `training_scripts/` a nivel estructural
- `notebooks/` a nivel estructural y de rol

## Qué no leer a fondo salvo necesidad real

No gastar esfuerzo principal en:

- `outputs/`
- `outputs/sweeps/`
- `outputs/retrained/`
- `outputs/retrained_v2/`
- `outputs/logs/`
- contenido pesado de `data/raw/`
- contenido pesado de `data/processed/`, `data/cleaned/`, `data/splits/`, `data/sets/`
- checkpoints, samples y artefactos grandes
- YAMLs históricos o generados en tandas

Se pueden inventariar, pero no analizar en profundidad sin necesidad clara.

## Carpetas y piezas a priorizar

Priorizar trabajo útil sobre:

- `data/`: limpieza, split, transforms, dataset building
- `models/`: `MLP`, `FlowPre`, `FlowGen`
- `losses/`: criterios puros
- `training/`: loops y operativa principal
- `evaluation/`: convertirla en capa canónica real
- `config/`: separar canon público de sensibilidad local

Tomar `training_scripts/` y notebooks como apoyo histórico, no como API pública final.

La carpeta interna `cement_imbalance/` del repo, actualmente mínima, no es el centro arquitectónico del proyecto y no debe usarse como destino automático de futuras reorganizaciones salvo que aporte valor real.

## Principios de refactor

- Refactorizar para clarificar responsabilidades, no por estética.
- Conservar lógica valiosa aunque hoy esté mal ubicada.
- Preferir extracción y reubicación gradual frente a reescritura total.
- Separar datos, transforms, entrenamiento, evaluación y reporting.
- Revisar también si nombres de archivos, carpetas o módulos son intuitivos, correctos y consistentes con su responsabilidad.
- No renombrar por gusto: solo cuando un nombre sea confuso, misleading o demasiado histórico.

## Notebooks, scripts versionados y outputs

- Las notebooks actuales son exploratorias, operativas o históricas hasta prueba en contrario.
- No convertir notebooks en mecanismo principal de entrenamiento final.
- Extraer de notebooks solo la lógica reusable o el conocimiento metodológico relevante.
- En `training_scripts/`, las últimas `1-2` versiones por familia suelen ser las mejores candidatas operativas; el resto es histórico salvo evidencia contraria.
- `outputs/` se trata como generado y regenerable, nunca como fuente primaria de verdad.

## Regla de trabajo sobre ramas

- No trabajar directamente sobre `main`.
- Crear siempre una rama o worktree específico antes de cambiar nada.
- Si el agente crea rama, usar prefijo `codex/`.
- No mezclar limpieza estructural amplia con cambios metodológicos críticos en la misma fase.

## Cambios de bajo riesgo y por fases

Preferir cambios pequeños, reversibles y verificables:

1. aclarar canon documental y frontera público/privado;
2. separar responsabilidades sin romper la operativa actual;
3. formalizar split, evaluación y manifests mínimos;
4. mover o renombrar solo cuando la responsabilidad quede más clara y el riesgo sea bajo;
5. dejar para el final la limpieza pública del repo y la poda histórica.

## Validaciones mínimas tras cambios

Tras cualquier cambio relevante, validar como mínimo:

- imports y rutas;
- que no se rompe la carga de configs canónicas;
- que no se introduce leakage obvio en split, cleaning o transforms;
- que scripts o módulos tocados ejecutan su smoke test razonable;
- que no se promueven artefactos regenerables a canon por accidente;
- que no se exponen nombres, mappings o detalles sensibles en código, docs o rutas.
- que la frontera definida en `docs/repo_visibility_matrix.md` sigue siendo consistente con los cambios hechos.

## Regla práctica adicional

- `main.py` no debe asumirse entrypoint canónico.
- Una mala ubicación actual no implica que una pieza sea descartable.
- Antes de publicar o mover algo desde `config/`, `data/` o notebooks, revisar si revela información sensible o dependencias locales no publicables.
