import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# ───────────────────────────────
# Create MLP with weight norm
# ───────────────────────────────
def create_mlp(in_features, out_features, hidden_features, num_layers, dropout=0.0):
    layers = []
    for i in range(num_layers):
        input_dim = in_features if i == 0 else hidden_features
        layers.append(weight_norm(nn.Linear(input_dim, hidden_features)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(weight_norm(nn.Linear(hidden_features, out_features)))
    return nn.Sequential(*layers)

# ───────────────────────────────
# Context-Conditioned MLP
# ───────────────────────────────
class ContextMLP(nn.Module):
    def __init__(
        self,
        in_features,
        context_dim,
        out_features,
        hidden_features,
        num_layers,
        dropout=0.0,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.net = create_mlp(in_features + context_dim, out_features, hidden_features, num_layers, dropout)
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
# Decoder Module
# ───────────────────────────────
class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,             # dim(X) + dim(y)
        num_classes: int,
        latent_dim: int,
        embedding_dim: int = 8,
        hidden_features: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_variance_head: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        self.context_embedding = nn.Embedding(num_classes, embedding_dim).to(device)

        self.decoder_mlp = ContextMLP(
            in_features=latent_dim,
            context_dim=embedding_dim,
            out_features=output_dim,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

        # Optional variance head (predicts log-variance, same output dim as mean)
        self.var_head = None
        if use_variance_head:
            # Lightweight head; you can mirror decoder_mlp if you want something beefier.
            self.var_head = ContextMLP(
                in_features=latent_dim,
                context_dim=embedding_dim,
                out_features=output_dim,  # outputs log-variance per feature
                hidden_features=hidden_features,
                num_layers=max(1, num_layers - 1),  # can be smaller than the mean path
                dropout=dropout,
                device=device
            )

    def get_context(self, class_labels):
        # Accept (B,), (B,1), (...,) and ensure 1-D long on the right device.
        cl = class_labels.to(self.device)
        if cl.dim() > 1:
            cl = cl.view(-1)  # collapse e.g. (B,1) → (B,)
        cl = cl.long()
        ctx = self.context_embedding(cl)  # (B, embed_dim)
        return ctx

    def forward(self, z, class_labels):
        context = self.get_context(class_labels)
        return self.decoder_mlp(z.to(self.device), context)

    def to_device(self, device):
        self.device = device
        self.decoder_mlp.to(device)
        self.context_embedding.to(device)

    def decode(self, z, class_labels):
        """
        Deterministic reconstruction.
        Returns:
          - mean (Tensor) if no variance head is present, OR
          - (mean, logvar) if a variance head exists (self.var_head).
        """
        context = self.get_context(class_labels)
        mean = self.decoder_mlp(z.to(self.device), context)

        if self.var_head is not None:
            logvar = self.var_head(z.to(self.device), context)
            return mean, logvar

        return mean


