# F7 Block 03 MLP Runtime Rationale

## Decision

Para `F7`, las runs de `MLP` deben ejecutarse en `cpu` de forma explícita.

No deben depender de:

- `auto`;
- `mps`;
- ni de una selección implícita de device en tiempo de campaña.

La política canónica queda reflejada en:

- [config/f7_mlp_base_v1.yaml](../config/f7_mlp_base_v1.yaml)

## Por que no se usa `MPS`

Este bloque no buscaba demostrar que `MPS` funciona, sino decidir qué backend conviene para la campaña real en esta máquina.

La evidencia ya reunida durante la revalidación del baseline mostró que:

- `train_mlp.py` ya usa la utilidad canónica de selección de device;
- `mps` puede pedirse explícitamente;
- pero, para las runs reales observadas en esta máquina, `cpu` fue claramente más rápida que `mps`.

La diferencia no fue marginal: en las tandas comparativas ya ejecutadas, `cpu` quedó muy por delante por coste medio por run.

## Por que se fuerza `cpu` en vez de usar `auto`

La decisión no es solo de rendimiento.

Forzar `cpu` mejora también la robustez operativa porque:

- evita fallback implícito no deseado;
- evita ambigüedad sobre el backend real de campaña;
- hace más predecible el benchmark de coste;
- y simplifica la trazabilidad de resultados en `F7`.

## Qué queda validado

Este bloque da por validado que:

- `train_mlp.py` resuelve el device mediante la utilidad canónica;
- el device efectivo queda persistido en artefactos de run;
- `cpu` es la política recomendada para `F7` en esta máquina;
- y no hace falta repetir un benchmark adicional para justificar la elección.

## Qué no implica esta decisión

Esto no significa que:

- `mps` quede eliminado del código;
- o que futuras exploraciones locales no puedan pedir `mps` explícitamente.

Significa solo que:

- para la parrilla `F7`,
- en esta máquina,
- la política canónica y recomendada es `cpu`.

## Rationale resumido

- `MPS` ya quedó soportado canónicamente en `train_mlp.py`;
- el benchmark real mostró que `cpu` rinde mejor para este tipo de runs en esta máquina;
- por claridad y reproducibilidad, `F7` no debe depender de `auto`;
- y las runs de campaña de `MLP` deben forzar `device=cpu`.
