import torch
import torch.nn as nn

class CVAECNF(nn.Module):
    def __init__(self, encoder, flow, decoder, device=None):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder

        self.device = device or (
            encoder.device if hasattr(encoder, "device") else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    # ---------- Core helper APIs (used by loss/trainer) ----------

    def get_context(self, class_labels):
        """
        Single source of context embedding.
        Prefer the flow’s embedding (used by the conditional prior); fall back to encoder’s.
        """
        if hasattr(self.flow, "get_context"):
            return self.flow.get_context(class_labels)
        return self.encoder.get_context(class_labels)

    def encode_params(self, x, class_labels):
        """
        Posterior parameters of q(z|x,c): returns (mu_q, logvar_q).
        """
        mu, logvar = self.encoder.forward(x, class_labels)
        return mu, logvar

    def encode(self, x, class_labels, sample=True):
        """
        Sample from the posterior q(z|x,c) and return (z_q, mu_q, logvar_q).
        If sample=False, returns z_q = mu_q (deterministic).
        """
        mu, logvar = self.encode_params(x, class_labels)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z, class_labels):
        """
        Deterministic reconstruction mean (extend later to return (mean, logvar) if needed).
        """
        if hasattr(self.decoder, "decode"):
            return self.decoder.decode(z, class_labels)
        return self.decoder(z, class_labels)

    def reconstruct(self, x, class_labels, use_mean_latent=True):
        """
        End-to-end reconstruction using the encoder's posterior:
        - if use_mean_latent=True, use z_q = mu_q (no sampling noise)
        - else, sample z_q ~ q(z|x,c)
        """
        z_q, _, _ = self.encode(x, class_labels, sample=not use_mean_latent)
        out = self.decode(z_q, class_labels)
        if isinstance(out, tuple):  # (mean, logvar) → use mean for metrics
            out = out[0]
        return out

    def prior(self, class_labels):
        """
        Parameters of a simple Normal prior in z-space (unused when using a flow prior).
        Kept for compatibility; override if you adopt a learned non-flow prior.
        """
        if hasattr(self.encoder, "latent_dim"):
            D = self.encoder.latent_dim
        else:
            raise RuntimeError("encoder.latent_dim not found; please expose it on the Encoder.")
        device = self.device
        shape = (class_labels.shape[0], D)
        mu = torch.zeros(shape, device=device, dtype=torch.float32)
        logvar = torch.zeros(shape, device=device, dtype=torch.float32)
        return mu, logvar

    # ---------- Forward / device movement ----------

    def forward(self, x, class_labels):
        """
        Prior-flow forward:
        - Encoder produces z_q ~ q(z|x,c) and (mu_q, logvar_q)
        - Decoder reconstructs from z_q
        NOTE: the conditional flow is NOT applied here; it defines p(z|c) and is used in the loss and sampling.
        """
        z_q, mu, logvar = self.encode(x, class_labels, sample=True)
        recon = self.decode(z_q, class_labels)
        return {
            "recon": recon,
            "z": z_q,
            "post_mu": mu,
            "post_logvar": logvar,
        }

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.flow.to(device)
        self.decoder.to(device)
        return self

    # ---- Prior (flow) utilities ----
    def prior_log_prob(self, z, class_labels):
        """
        log p(z|c) under the conditional flow prior.
        Falls back to flow.log_prob if aliases aren't defined.
        """
        return self.flow.log_prob_z(z, class_labels) if hasattr(self.flow, "log_prob_z") \
               else self.flow.log_prob(z, class_labels)

    def sample(self, num_samples, class_labels):
        """
        Generate samples from the conditional prior and decode:
        u ~ N(0,I) ; z = g(u,c) ; x_hat = Dec(z,c)
        """
        z = self.flow.sample_z(num_samples, class_labels) if hasattr(self.flow, "sample_z") \
            else self.flow.sample(num_samples, class_labels)
        return self.decode(z, class_labels)



