# Backup and Restore

Este documento fija que recupera el repo tracked por si solo, que sigue viviendo solo en local, y que hace falta guardar fuera de GitHub si queremos poder volver al estado practico actual del proyecto.

## Regla base

- GitHub puede guardar bien el **repo**.
- GitHub no sustituye el backup externo del **proyecto operativo completo**.
- La nueva capa tracked de finalists/winners bajo `config/finalists*` y `docs/finalists*` mejora mucho la recuperacion semantica del cierre, pero no reemplaza datos, overlays privados ni artifacts pesados.

## Que aporta ya la capa tracked ligera

La superficie nueva:

- `config/finalists_registry.yaml`
- `config/finalists/`
- `config/finalists/config_snapshots/`
- `docs/finalists_registry.md`
- `docs/finalists/`
- `scripts/export_finalists_registry.py`

permite recuperar desde GitHub:

- que artifact gano o fue promovido en cada linea;
- con que `run_id`, `seed` y rol semantico;
- una copia sanitizada de la config relevante;
- un resumen de resultados y rationale de seleccion;
- referencias `local_only` a los manifests, results y rationales originales bajo `outputs/`.

## Escenario 1: GitHub solo

Con solo clonar GitHub se recupera:

- codigo fuente mantenido;
- docs canonicos y de apoyo;
- configs base public-safe;
- la capa tracked ligera de finalists/winners;
- la politica de split, evaluacion y cierres narrativos.

No se recupera:

- `data/raw/`;
- `config/local/`;
- `data/processed/`, `data/cleaned/`, `data/splits/`, `data/sets/`;
- checkpoints, reports y outputs pesados;
- el estado operativo completo listo para rerun inmediato.

Lectura correcta:

- sirve como backup serio del **repo**;
- no basta para restaurar por si solo el **estado local completo**.

## Escenario 2: GitHub + `data/raw/` + `config/local/`

Con esta combinacion ya se puede:

- reinstalar el entorno;
- volver a cargar overlays privados reales;
- rerunear preprocess, cleaning y split;
- regenerar buena parte del pipeline derivado;
- reconstruir bastante del estado metodologico del proyecto.

Sigue faltando para una recuperacion practica rapida:

- finalists y winners promovidos ya materializados en `outputs/`;
- manifests locales de cierre;
- resultados de cierre ya agregados;
- checkpoints o run dirs cuya regeneracion seria costosa o lenta.

Lectura correcta:

- este escenario ya permite **reconstruccion fuerte**;
- pero no garantiza **recuperacion inmediata** del mismo estado local actual.

## Escenario 3: GitHub + `data/raw/` + `config/local/` + outputs promovidos/finalistas

Este es el escenario que mas se parece a una copia de seguridad completa del estado practico actual.

Ademas de lo anterior, guardar:

- `outputs/models/official/flowpre_finalists/`
- `outputs/models/official/flowgen_finalist/`
- `outputs/models/experimental/train_only/flowpre_finalists/`
- `outputs/models/experimental/train_only/flowgen_finalist/`
- cualquier report local de cierre que se quiera conservar como soporte

permite:

- recuperar finalists/winners materiales;
- verificar la capa tracked contra sus sources locales originales;
- reanudar trabajo downstream sin tener que rehacer los cierres ya congelados;
- defender mejor el estado exacto que tenia el proyecto al momento del backup.

## Backup externo minimo obligatorio

Si queremos una politica de backup sensata, hay que guardar fuera de GitHub como minimo:

1. `data/raw/`
2. `config/local/`
3. `outputs/models/.../finalists` y winners promovidos

Opcional pero recomendable para restore rapido:

4. `data/splits/official/`
5. `data/sets/official/`

## Que es regenerable y que no conviene regenerar

### Regenerable, si existen `data/raw/` y `config/local/`

- preprocess;
- cleaning split-aware;
- split oficial;
- bundles oficiales raw/scaled;
- parte de los reports y manifests derivados.

### Mejor conservar por backup, no solo por regeneracion

- finalists/winners ya promovidos;
- manifests locales de cierre;
- results y rationales usados en la seleccion final;
- run dirs cuya recreacion sea costosa o dependa de mucho compute.

## Flujo recomendado de restore

1. clonar GitHub;
2. restaurar `data/raw/`;
3. restaurar `config/local/`;
4. restaurar outputs promovidos/finalistas si se quiere volver al mismo estado practico;
5. reinstalar entorno;
6. usar `config/finalists_registry.yaml` y `docs/finalists_registry.md` como indice para comprobar que artifacts deberian existir;
7. si hace falta, rerunear `scripts/export_finalists_registry.py --check --strict` para validar que la capa tracked sigue alineada con los artifacts locales restaurados.

## Regla final

- `outputs/` sigue siendo la fuente local pesada/original;
- `config/finalists*` y `docs/finalists*` son la referencia tracked ligera y public-safe;
- el backup serio del proyecto requiere ambas capas mas los datos y overlays privados.
