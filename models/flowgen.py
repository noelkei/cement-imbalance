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

torch.set_default_dtype(torch.float32)

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
        self.to(dtype=torch.float32, device=device)

    def forward(self, x, context):
        x_context = torch.cat([x, context], dim=1)
        return self.net(x_context)


# ───────────────────────────────
# FlowGen: joint flow over [X, y] | c  (y appended after X)
# ───────────────────────────────
class FlowGen(nn.Module):
    """
    Joint conditional flow over the concatenated vector [X, y] given class labels c.
    - X has dimension x_dim
    - y has dimension y_dim (>= 1)
    - The model is invertible over R^{x_dim + y_dim} and uses the same architecture as FlowPre.
    - Helpers are provided to concatenate/split [X, y], so you can pass either the full XY vector
      or (X, y) separately for convenience.

    Notes:
    - For training: pass the full concatenated XY into forward/log_prob, or use forward_xy/log_prob_xy helpers.
    - For sampling: sample(...) returns concatenated XY; use sample_xy(...) to get (X, y) split.
    """
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
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
        assert x_dim >= 1 and y_dim >= 1, "x_dim and y_dim must be >= 1"
        input_dim = int(x_dim + y_dim)

        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
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

        # alternating binary masks across the FULL concatenated dimension [X, y]
        _base = (torch.arange(input_dim) % 2).float()  # 0/1 as float32 on CPU
        masks = [_base, 1.0 - _base]

        transforms = []

        # Initial affine layers
        for i in range(initial_affine_layers):
            if use_actnorm:
                transforms.append(ActNorm(input_dim))
            transforms.append(add_permutation())
            transforms.append(build_affine(mask=masks[i % 2].cpu()))

        # Blocked alternation (Affine then RQ per ratios)
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
        ).to(dtype=torch.float32, device=device)

    # ---------------------------
    # Helpers for context and XY assembly/splitting
    # ---------------------------
    def get_context(self, class_labels: torch.Tensor):
        return self.context_embedding(class_labels.to(self.device))

    def _concat_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and y.dim() == 2, "x and y must be 2D tensors [N, d]"
        assert x.size(0) == y.size(0), "Batch size mismatch between x and y"
        assert x.size(1) == self.x_dim and y.size(1) == self.y_dim, "x/y dims mismatch"
        return torch.cat([x, y], dim=1)

    def _split_xy(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert xy.dim() == 2 and xy.size(1) == self.input_dim, "xy must be [N, x_dim + y_dim]"
        x = xy[:, : self.x_dim]
        y = xy[:, self.x_dim :]
        return x, y

    # ---------------------------
    # Forward / Inverse / Log-Prob
    # ---------------------------
    def forward(self, xy: torch.Tensor, class_labels: torch.Tensor):
        """
        Forward transform of concatenated XY: returns (z, logabsdet).
        """
        context = self.get_context(class_labels)
        return self.flow._transform.forward(xy.to(self.device, dtype=torch.float32), context)

    def forward_xy(self, x: torch.Tensor, y: torch.Tensor, class_labels: torch.Tensor):
        """
        Convenience: forward on separate (x, y) -> (z, logabsdet).
        """
        xy = self._concat_xy(x.to(self.device), y.to(self.device))
        return self.forward(xy, class_labels)

    def inverse(self, z: torch.Tensor, class_labels: torch.Tensor):
        """
        Inverse transform from latent z to concatenated XY: returns (xy, logabsdet).
        """
        context = self.get_context(class_labels)
        return self.flow._transform.inverse(z.to(self.device, dtype=torch.float32), context)

    def inverse_xy(self, z: torch.Tensor, class_labels: torch.Tensor):
        """
        Convenience: inverse that returns split (x, y).
        """
        xy, logabsdet = self.inverse(z, class_labels)
        x, y = self._split_xy(xy)
        return (x, y), logabsdet

    def log_prob(self, xy: torch.Tensor, class_labels: torch.Tensor):
        """
        Log-probability of concatenated XY under the conditional flow.
        """
        context = self.get_context(class_labels)
        return self.flow.log_prob(inputs=xy.to(self.device, dtype=torch.float32), context=context)

    def log_prob_xy(self, x: torch.Tensor, y: torch.Tensor, class_labels: torch.Tensor):
        """
        Convenience: log_prob on separate (x, y).
        """
        xy = self._concat_xy(x.to(self.device), y.to(self.device))
        return self.log_prob(xy, class_labels)

    # ---------------------------
    # Sampling
    # ---------------------------
    def sample(self, num_samples: int, class_labels: torch.Tensor):
        """
        Sample concatenated XY given class labels.
        If a single class id is provided, it will be expanded to num_samples.
        If a per-sample vector of class ids is provided, its length must equal num_samples.
        """
        device = self.device
        if not torch.is_tensor(class_labels):
            class_labels = torch.as_tensor(class_labels, dtype=torch.long, device=device)

        class_labels = class_labels.to(torch.long).to(device)
        if class_labels.dim() == 0:
            class_labels = class_labels.view(1)

        if class_labels.numel() == 1:
            class_labels = class_labels.expand(num_samples)
        else:
            assert class_labels.numel() == num_samples, \
                f"class_labels has {class_labels.numel()} items, expected {num_samples}"

        context = self.get_context(class_labels)  # (num_samples, emb)
        z = torch.randn(num_samples, self.input_dim, device=device, dtype=torch.float32)
        xy, _ = self.flow._transform.inverse(z, context)
        return xy

    def sample_xy(self, num_samples: int, class_labels: torch.Tensor):
        """
        Convenience: sample and split to (X, y).
        """
        xy = self.sample(num_samples, class_labels)
        x, y = self._split_xy(xy)
        return x, y

    # ---------------------------
    # Device
    # ---------------------------
    def to_device(self, device: str | torch.device):
        self.device = str(device) if isinstance(device, str) else device
        self.flow.to(dtype=torch.float32, device=device)
        self.context_embedding.to(dtype=torch.float32, device=device)
        return self



