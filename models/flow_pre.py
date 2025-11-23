import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    RandomPermutation,
    BatchNorm
)
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm

# ───────────────────────────────
# Create MLP with weight norm
# ───────────────────────────────
def create_mlp(in_features, out_features, hidden_features, num_layers):
    layers = []
    for i in range(num_layers):
        input_size = in_features if i == 0 else hidden_features
        layers.append(weight_norm(nn.Linear(input_size, hidden_features)))
        layers.append(nn.ReLU())
    layers.append(weight_norm(nn.Linear(hidden_features, out_features)))
    return nn.Sequential(*layers)

class ContextMLP(nn.Module):
    def __init__(self, in_features, context_dim, out_features, hidden_features, num_layers, device="cpu"):
        super().__init__()
        self.net = create_mlp(in_features + context_dim, out_features, hidden_features, num_layers)
        self.to(device)

    def forward(self, x, context):
        x_context = torch.cat([x, context], dim=1)
        return self.net(x_context)

# ───────────────────────────────
# FlowPre class with flexible architecture
# ───────────────────────────────
class FlowPre(nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.embedding_dim = embedding_dim
        self.context_embedding = nn.Embedding(num_classes, embedding_dim).to(device)

        def add_permutation():
            return LULinear(input_dim) if use_learnable_permutations else RandomPermutation(input_dim)

        def build_affine(mask):
            return AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_f, out_f: ContextMLP(
                    in_f, embedding_dim, out_f, hidden_features, num_layers, device)
            )

        def build_rq(mask):
            return PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_f, out_f: ContextMLP(
                    in_f, embedding_dim, out_f, hidden_features, num_layers, device),
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound
            )

        masks = [
            (torch.arange(input_dim) % 2).to(device),
            ((torch.arange(input_dim) + 1) % 2).to(device)
        ]

        transforms = []

        # Initial affine layers
        for i in range(initial_affine_layers):
            if use_actnorm:
                transforms.append(ActNorm(input_dim))
            transforms.append(add_permutation())
            transforms.append(build_affine(mask=masks[i % 2].cpu()))

        # Blocked alternation
        for i in range(n_repeat_blocks):
            for _ in range(affine_rq_ratio[0]):
                transforms.append(add_permutation())
                transforms.append(build_affine(mask=masks[i % 2].cpu()))
            for _ in range(affine_rq_ratio[1]):
                transforms.append(add_permutation())
                transforms.append(build_rq(mask=masks[i % 2].cpu()))

        # Final RQ layers
        for i in range(final_rq_layers):
            transforms.append(add_permutation())
            transforms.append(build_rq(mask=masks[i % 2].cpu()))

        # Optional final LULinear
        if lulinear_finisher:
            transforms.append(LULinear(input_dim))

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal(shape=[input_dim])
        ).to(device)

    def get_context(self, class_labels):
        return self.context_embedding(class_labels.to(self.device))

    def forward(self, x, class_labels):
        context = self.get_context(class_labels)
        return self.flow._transform.forward(x.to(self.device), context)

    def inverse(self, z, class_labels):
        context = self.get_context(class_labels)
        return self.flow._transform.inverse(z.to(self.device), context)

    def log_prob(self, x, class_labels):
        context = self.get_context(class_labels)
        return self.flow.log_prob(inputs=x.to(self.device), context=context)

    def sample(self, num_samples, class_labels):
        context = self.get_context(class_labels)
        return self.flow.sample(num_samples, context=context)

    def to_device(self, device):
        self.device = device
        self.flow.to(device)
        self.context_embedding.to(device)

