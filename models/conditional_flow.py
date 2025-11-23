import torch
import torch.nn as nn

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm, BatchNorm
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)

import math
# at top (optional: newer, non-deprecated weight_norm)
try:
    from torch.nn.utils.parametrizations import weight_norm as _wn   # PyTorch ≥2.0
except Exception:
    from torch.nn.utils import weight_norm as _wn                    # fallback

# ───────────────────────────────
# Create MLP with optional weight norm (hidden only) and zero-init last
# ───────────────────────────────
def create_mlp(
    in_features: int,
    out_features: int,
    hidden_features: int,
    num_layers: int,
    *,
    apply_weight_norm_hidden: bool = True,
    zero_init_last: bool = True,
    last_bias_init: float = 0.0,
):
    layers = []
    for i in range(num_layers):
        inp = in_features if i == 0 else hidden_features
        lin = nn.Linear(inp, hidden_features)
        if apply_weight_norm_hidden:
            lin = _wn(lin)
        layers += [lin, nn.ReLU()]

    # final layer: NO weight_norm if we want to zero-init safely
    last = nn.Linear(hidden_features, out_features)
    if zero_init_last:
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, last_bias_init)

    layers.append(last)
    return nn.Sequential(*layers)


class ContextMLP(nn.Module):
    def __init__(
        self,
        in_features,
        context_dim,
        out_features,
        hidden_features,
        num_layers,
        device="cpu",
        *,
        apply_weight_norm_hidden: bool = True,
        zero_init_last: bool = True,
        last_bias_init: float = 0.0,
    ):
        super().__init__()
        self.net = create_mlp(
            in_features + context_dim,
            out_features,
            hidden_features,
            num_layers,
            apply_weight_norm_hidden=apply_weight_norm_hidden,
            zero_init_last=zero_init_last,
            last_bias_init=last_bias_init,
        )
        self.to(device)

    def forward(self, x, context):
        # Ensure context is 2-D: (B, C)
        if context.dim() == 1:
            context = context.unsqueeze(1)  # (B,) → (B,1)
        if context.dim() == 3 and context.size(1) == 1:
            context = context.squeeze(1)  # (B,1,E) → (B,E)
        assert x.dim() == 2 and context.dim() == 2, f"x {x.shape}, context {context.shape}"
        x_context = torch.cat([x, context], dim=1)
        return self.net(x_context)


# ───────────────────────────────
# NoFlowPrior: standard Normal prior p(z)=N(0,I) (ignores context)
# ───────────────────────────────

class NoFlowPrior(nn.Module):
    """
    A drop-in, parameter-free 'prior' that matches the Flow interface the trainer expects.
    - log_prob(z, c): returns log N(z | 0, I) per-sample (B,)
    - sample(n, c): returns z ~ N(0, I) with shape (n, Dz)
    - sample_z/log_prob_z aliases provided for compatibility
    """
    def __init__(self, latent_dim: int, num_classes: int, device: str = "cpu"):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)  # kept for symmetry; unused
        self.device = torch.device(device)

    def to(self, device):
        self.device = torch.device(device)
        return super().to(device)

    def log_prob(self, inputs: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # inputs: (B, Dz)
        x = inputs
        Dz = x.shape[1]
        # log N(x | 0, I) per sample
        # = -0.5 * (Dz*log(2π) + ||x||^2)
        const = -0.5 * Dz * math.log(2.0 * math.pi)
        quad = -0.5 * torch.sum(x * x, dim=1)
        return const + quad  # (B,)

    # aliases for CVAECNF convenience
    def log_prob_z(self, z: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.log_prob(z, context)

    def sample(self, num_samples: int, context: torch.Tensor | None = None) -> torch.Tensor:
        return torch.randn(num_samples, self.latent_dim, device=self.device)

    def sample_z(self, num_samples: int, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.sample(num_samples, context)

# ───────────────────────────────
# Conditional Normalizing Flow
# ───────────────────────────────
class ConditionalFlow(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        embedding_dim: int = 8,
        hidden_features: int = 64,
        num_layers: int = 2,
        use_actnorm: bool = True,
        use_learnable_permutations: bool = True,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        initial_affine_layers: int = 2,
        affine_rq_ratio: tuple = (1, 3),
        n_repeat_blocks: int = 4,
        final_rq_layers: int = 3,
        lulinear_finisher: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        apply_weight_norm_hidden: bool = True,
        zero_init_last: bool = True,
        last_bias_init: float = 0.0,

    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.embedding_dim = embedding_dim

        self.context_embedding = nn.Embedding(num_classes, embedding_dim).to(device)

        def add_permutation():
            return LULinear(latent_dim) if use_learnable_permutations else RandomPermutation(latent_dim)

        def build_affine(mask):
            return AffineCouplingTransform(
                mask=mask.cpu(),
                transform_net_create_fn=lambda in_f, out_f: ContextMLP(
                    in_f, embedding_dim, out_f, hidden_features, num_layers, device,
                    apply_weight_norm_hidden=apply_weight_norm_hidden,
                    zero_init_last=zero_init_last,
                    last_bias_init=last_bias_init,
                )
            )

        def build_rq(mask):
            return PiecewiseRationalQuadraticCouplingTransform(
                mask=mask.cpu(),
                transform_net_create_fn=lambda in_f, out_f: ContextMLP(
                    in_f, embedding_dim, out_f, hidden_features, num_layers, device,
                    apply_weight_norm_hidden=apply_weight_norm_hidden,
                    zero_init_last=zero_init_last,
                    last_bias_init=last_bias_init,
                ),
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound
            )

        masks = [
            (torch.arange(latent_dim) % 2).to(device),
            ((torch.arange(latent_dim) + 1) % 2).to(device)
        ]

        transforms = []

        # Initial affine layers
        for i in range(initial_affine_layers):
            if use_actnorm:
                transforms.append(ActNorm(latent_dim))
            else:
                transforms.append(BatchNorm(latent_dim))

            transforms.append(add_permutation())
            transforms.append(build_affine(mask=masks[i % 2]))

        # Alternating blocks
        for i in range(n_repeat_blocks):
            for _ in range(affine_rq_ratio[0]):
                transforms.append(add_permutation())
                transforms.append(build_affine(mask=masks[i % 2]))

            for _ in range(affine_rq_ratio[1]):
                transforms.append(add_permutation())
                transforms.append(build_rq(mask=masks[i % 2]))

        # Final RQ layers
        for i in range(final_rq_layers):
            transforms.append(add_permutation())
            transforms.append(build_rq(mask=masks[i % 2]))

        if lulinear_finisher:
            transforms.append(LULinear(latent_dim))

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal(shape=[latent_dim])
        ).to(device)

    def get_context(self, class_labels):
        # Accept (B,), (B,1), (...,) and ensure 1-D long on the right device.
        cl = class_labels.to(self.device)
        if cl.dim() > 1:
            cl = cl.view(-1)  # collapse e.g. (B,1) → (B,)
        cl = cl.long()
        ctx = self.context_embedding(cl)  # (B, embed_dim)
        return ctx

    def forward(self, z0, class_labels):
        context = self.get_context(class_labels)
        return self.flow._transform.forward(z0.to(self.device), context)

    def inverse(self, zK, class_labels):
        context = self.get_context(class_labels)
        return self.flow._transform.inverse(zK.to(self.device), context)

    def log_prob(self, zK, class_labels):
        context = self.get_context(class_labels)
        return self.flow.log_prob(inputs=zK.to(self.device), context=context)

    # in models/conditional_flow.py
    def sample(self, num_samples, class_labels):
        """
        Draw z ~ p(z|c) with shapes aligned per-row.
        - If num_samples == len(class_labels): return (B, Dz) (one sample per context row).
        - If num_samples is a multiple of B: return (B, k, Dz) (k samples per context row).
        - Otherwise: fall back to a simple (num_samples, Dz) using the *mean* context.
        """
        # Normalize labels -> (B,)
        cl = class_labels.to(self.device)
        if cl.dim() > 1:
            cl = cl.view(-1)
        cl = cl.long()
        ctx = self.get_context(cl)  # (B, E)
        B = ctx.shape[0]
        Dz = self.latent_dim

        # Normalized n
        n = int(num_samples)

        # Case A: one sample per context row → (B, Dz)
        if n == B:
            u = torch.randn(B, Dz, device=self.device)
            z, _ = self.flow._transform.inverse(u, ctx)  # (B, Dz)
            return z

        # Case B: k per context → (B, k, Dz)
        if n % B == 0:
            k = n // B
            u = torch.randn(B * k, Dz, device=self.device)
            ctx_k = ctx.repeat_interleave(k, dim=0)  # (B*k, E)
            z, _ = self.flow._transform.inverse(u, ctx_k)  # (B*k, Dz)
            return z.view(B, k, Dz)

        # Case C: fallback — sample w.r.t. a single representative context
        # (kept for completeness; not hit by your current code path)
        u = torch.randn(n, Dz, device=self.device)
        # use the first row's context
        z, _ = self.flow._transform.inverse(u, ctx[:1].expand(n, -1))
        return z

    def to_device(self, device):
        self.device = device
        self.flow.to(device)
        self.context_embedding.to(device)

    def log_abs_det_jacobian(self, z0, class_labels):
        """
        Alias for a forward transform that returns (zK, sum_logabsdet) explicitly.
        Matches how your wrapper tries to call it.
        """
        return self.forward(z0, class_labels)

    # (Optional niceties; zero math changes but clearer names)
    def transform(self, z0, class_labels):
        """Same as forward; returns (zK, sum_logabsdet)."""
        return self.forward(z0, class_labels)

    def inverse_transform(self, zK, class_labels):
        """Inverse map; returns (z0, sum_logabsdet_inv)."""
        return self.inverse(zK, class_labels)

    # --- helpers for prior usage ---
    def log_prob_z(self, z, class_labels):
        """log p(z|c) under the conditional flow prior."""
        return self.log_prob(z, class_labels)

    def sample_z(self, num_samples, class_labels):
        """Draw z ~ p(z|c) from the flow prior."""
        return self.sample(num_samples, class_labels)


