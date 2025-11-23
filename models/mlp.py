# mlp.py
import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _cast_module_dtype_(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    for p in module.parameters(recurse=True):
        if p.data.is_floating_point():
            p.data = p.data.to(dtype)
        if p.grad is not None and p.grad.is_floating_point():
            p.grad = p.grad.to(dtype)
    for b in module.buffers(recurse=True):
        if b.data.is_floating_point():
            b.data = b.data.to(dtype)
    return module


def _get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":  return nn.ReLU()
    if name == "gelu":  return nn.GELU()
    if name in ("silu", "swish"): return nn.SiLU()
    if name == "elu":   return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


class MLPBlock(nn.Module):
    """Linear -> (BN?) -> Act -> (Dropout?) with optional weight_norm."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: str = "relu",
        batchnorm: bool = False,
        dropout: float = 0.0,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        lin = nn.Linear(in_dim, out_dim)
        self.lin = weight_norm(lin) if use_weight_norm else lin
        self.bn  = nn.BatchNorm1d(out_dim) if batchnorm else None
        self.act = _get_activation(activation)
        self.do  = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        if self.do is not None:
            x = self.do(x)
        return x


class ContextMLPRegressor(nn.Module):
    """
    Predict y from [X ⊕ context(c)].

    forward(x, c, return_context=False) -> y_pred  (or (y_pred, ctx_vec) if return_context)

    Context modes:
      - "embed" (default): learned nn.Embedding(num_classes, embedding_dim)
      - "onehot": raw one-hot vector of length num_classes (no learned params)

    Notes:
    - Caller passes a resolved device ('cuda'/'mps'/'cpu' or torch.device).
    - On MPS, dtype coerced to float32 unless float16 explicitly requested.
    """
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        y_dim: int = 1,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        use_weight_norm: bool = True,
        residual: bool = True,
        final_activation: Optional[Literal["sigmoid", "tanh"]] = None,
        task: Literal["regression", "classification"] = "regression",
        context_mode: Literal["embed", "onehot"] = "embed",
        device: torch.device | str = "cpu",
        dtype: Optional[torch.dtype] = None,
        # optional metadata hooks (not used by model math; handy for logging/SHAP)
        feature_names_in: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
    ):
        super().__init__()
        assert input_dim >= 1 and num_classes >= 1 and y_dim >= 1

        self.device = torch.device(device)
        # dtype policy
        if dtype is None:
            self.compute_dtype = torch.float32 if self.device.type == "mps" else torch.float32
        else:
            if self.device.type == "mps" and dtype not in (torch.float32, torch.float16):
                self.compute_dtype = torch.float32
            else:
                self.compute_dtype = dtype

        self.task = task
        self.residual = bool(residual)
        self.num_classes = int(num_classes)
        self.context_mode = context_mode

        # context representation
        if self.context_mode == "embed":
            self.context_embedding = nn.Embedding(num_classes, embedding_dim)
            ctx_dim = embedding_dim
        elif self.context_mode == "onehot":
            self.context_embedding = None
            ctx_dim = num_classes
        else:
            raise ValueError("context_mode must be 'embed' or 'onehot'")

        # backbone
        concat_in = input_dim + ctx_dim
        blocks = []
        prev = concat_in
        for _ in range(max(0, num_layers)):
            blocks.append(
                MLPBlock(
                    in_dim=prev,
                    out_dim=hidden_dim,
                    activation=activation,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
            prev = hidden_dim
        self.backbone = nn.Sequential(*blocks)

        # head
        head_in = prev if num_layers > 0 else concat_in
        head = nn.Linear(head_in, y_dim)
        self.head = weight_norm(head) if use_weight_norm else head

        # optional output activation
        if final_activation is None:
            self.out_act = None
        elif final_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif final_activation == "tanh":
            self.out_act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported final_activation '{final_activation}'")

        # init + move/cast
        self.apply(self._init_weights)
        self.to(self.device)
        _cast_module_dtype_(self, self.compute_dtype)

        # metadata (purely informational)
        self.feature_names_in = feature_names_in
        self.target_names = target_names
        self.class_names = class_names

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    # ---- API: move / cast safely ----
    def to_device_dtype(self, device: torch.device | str, dtype: Optional[torch.dtype] = None):
        dev = torch.device(device)
        dt = dtype if dtype is not None else self.compute_dtype
        if dev.type == "mps" and dt not in (torch.float32, torch.float16):
            dt = torch.float32
        self.device = dev
        self.compute_dtype = dt
        self.to(self.device)
        _cast_module_dtype_(self, self.compute_dtype)
        return self

    # ---- API: load weights robustly ----
    def load_weights(self, ckpt_path: str, *, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

        chk_dtype = None
        for v in sd.values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                chk_dtype = v.dtype
                break
        chk_dtype = chk_dtype or torch.float32

        cpu_self = self.to("cpu")
        _cast_module_dtype_(cpu_self, chk_dtype)
        cpu_self.load_state_dict(sd, strict=strict)

        self.to(self.device)
        _cast_module_dtype_(self, self.compute_dtype)
        return self

    # ---- context helpers ----
    def _validate_classes(self, c: torch.Tensor) -> torch.Tensor:
        c = c.to(self.device)
        if c.dtype != torch.long:
            c = c.long()
        if torch.any((c < 0) | (c >= self.num_classes)):
            raise ValueError("class_labels contain ids outside [0, num_classes).")
        return c

    def get_context(self, class_labels: torch.Tensor) -> torch.Tensor:
        c = self._validate_classes(class_labels)
        if self.context_mode == "embed":
            ctx = self.context_embedding(c)
        else:  # onehot
            ctx = F.one_hot(c, num_classes=self.num_classes).to(self.device, dtype=self.compute_dtype)
        return ctx.to(self.device, dtype=self.compute_dtype)

    # ---- forward ----
    @torch.no_grad()
    def encode_context(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Public, no-grad context encoder (useful for caching/inspection)."""
        return self.get_context(class_labels)

    def forward(
        self,
        x: torch.Tensor,
        class_labels: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device, dtype=self.compute_dtype)
        ctx = self.get_context(class_labels)
        h = torch.cat([x, ctx], dim=1)

        if len(self.backbone) > 0:
            for layer in self.backbone:
                h_in = h
                h = layer(h)
                if self.residual and h.shape == h_in.shape:
                    h = h + h_in

        y = self.head(h)
        if self.out_act is not None:
            y = self.out_act(y)

        return (y, ctx) if return_context else y

    # convenience for SHAP wrappers (expects a single tensor = X, and a fixed c)
    def predict_with_fixed_context(self, x: torch.Tensor, fixed_class_id: int) -> torch.Tensor:
        c = torch.full((x.shape[0],), int(fixed_class_id), device=self.device, dtype=torch.long)
        return self.forward(x, c)
