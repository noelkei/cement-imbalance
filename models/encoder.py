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
# Encoder Module
# ───────────────────────────────
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,             # dim(X) + dim(y)
        num_classes: int,
        latent_dim: int,
        embedding_dim: int = 8,
        hidden_features: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        self.context_embedding = nn.Embedding(num_classes, embedding_dim).to(device)

        self.mlp_mu = ContextMLP(
            in_features=input_dim,
            context_dim=embedding_dim,
            out_features=latent_dim,
            hidden_features=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )

        self.mlp_logvar = ContextMLP(
            in_features=input_dim,
            context_dim=embedding_dim,
            out_features=latent_dim,
            hidden_features=hidden_features,
            num_layers=num_layers,
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

    def forward(self, x, class_labels):
        context = self.get_context(class_labels)
        mu = self.mlp_mu(x.to(self.device), context)
        logvar = self.mlp_logvar(x.to(self.device), context)
        return mu, logvar

    def encode(self, x, class_labels):
        """Returns sampled z, along with mu and logvar."""
        mu, logvar = self.forward(x, class_labels)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def to_device(self, device):
        """Move model components to a new device."""
        self.device = device
        self.context_embedding.to(device)
        self.mlp_mu.to(device)
        self.mlp_logvar.to(device)


