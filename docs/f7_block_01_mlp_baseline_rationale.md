# F7 Block 01 MLP Baseline Rationale

Este bloque ya quedó razonado en detalle en:

- [docs/f7_mlp_baseline_final_v1.md](f7_mlp_baseline_final_v1.md)

Este archivo existe para dar una ruta de búsqueda estable y homogénea para la memoria del TFG.

## Decision final

El baseline estructural final de `MLP` para `F7` queda congelado en:

- `hidden_dim = 320`
- `num_layers = 4`
- `embedding_dim = 12`
- `batch_size = 198`
- `learning_rate = 1e-4`
- `num_epochs = 300`
- `optimizer = adam`
- `lr_scheduler = plateau`
- `early_stopping_patience = 20`

Config canónica:

- [config/f7_mlp_base_v1.yaml](../config/f7_mlp_base_v1.yaml)

## Rationale completo

Para evitar duplicar y desalinear narrativa, el rationale metodológico completo del bloque `1` queda concentrado en:

- [docs/f7_mlp_baseline_final_v1.md](f7_mlp_baseline_final_v1.md)

## Revalidaciones de apoyo

- [docs/f7_mlp_baseline_revalidation_v1.md](f7_mlp_baseline_revalidation_v1.md)
- [docs/f7_mlp_baseline_revalidation_v2.md](f7_mlp_baseline_revalidation_v2.md)
- [docs/f7_mlp_baseline_revalidation_v3.md](f7_mlp_baseline_revalidation_v3.md)
