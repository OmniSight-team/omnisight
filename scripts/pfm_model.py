"""OMNI-904: the Predictive Forecasting Module, Algorithm 4 of Phase A (§3.5.4, Appendix B.4).

    1: h <- Backbone_Transformer(W)        shared across domains
    2: (p, Δt) <- Head_d(h)                Δt measured forward from t
    3: κ <- 1 - H(p) / log(|C_d|)          normalised confidence (inverse entropy)
    L = Σ_d λ_d [ CE(p_d, y_d) + α · MSE(Δt_d, τ_d) ]

Steps 4-5 (the threshold and the candidate alert) are OMNI-905's and OMNI-910's.

**The window** `W` is a sequence of clip vectors on the grid (D-54), from its window's start to `t`
(D-55), padded at the end with a mask. **The backbone:** the vectors projected to `width`, a position
counted back from `t` (0 = the newest vector, so "how long before now" means the same in every sample),
a learned summary token in front, and a pre-norm Transformer encoder that ignores the padding. The
summary token's output is `h`. **A head per domain:** one linear layer giving the classes' logits and
`Δt` (softplus: a time ahead is never negative). The MSE is taken on positive samples only (a negative
has no event, so no `τ`).

Every setting is in `Config`, saved with each checkpoint.
"""

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

# Index 0 is Normal in every domain. UCF-Crime: its 13 event classes (UCF-Crime's own names).
CLASSES = {
    "ccd": ["Normal", "Crash"],
    "ucf_crime": ["Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
                  "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"],
}


@dataclass
class Config:
    input_dim: int  # 512 InternVideo2, 768 Cosmos-Embed1
    domains: list[str]
    max_length: int = 64  # UCF-Crime's 60 s window at 1 s is 60 vectors
    width: int = 128
    layers: int = 2
    heads: int = 4
    ff: int = 256
    dropout: float = 0.1
    alpha: float = 1.0  # weight of the Δt term
    lambdas: dict[str, float] = field(default_factory=dict)  # per domain; missing = 1.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch: int = 256
    epochs: int = 30

    def lam(self, domain: str) -> float:
        return self.lambdas.get(domain, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)


class PFM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.input_dim, config.width)
        self.position = nn.Embedding(config.max_length, config.width)
        self.summary = nn.Parameter(torch.zeros(1, 1, config.width))
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.ff, config.dropout,
                                           batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(config.width),
                                              enable_nested_tensor=False)
        self.heads = nn.ModuleDict({d: nn.Linear(config.width, len(CLASSES[d]) + 1) for d in config.domains})
        nn.init.normal_(self.summary, std=0.02)

    def backbone_out(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`h` (n, width) for windows `x` (n, L, input_dim), `mask` (n, L) True on the sequence."""
        n, length, _ = x.shape
        if length > self.config.max_length:
            raise ValueError(f"{length} vectors, the model takes at most {self.config.max_length}")
        sizes = mask.sum(dim=1, keepdim=True)
        back = (sizes - 1 - torch.arange(length, device=x.device)).clamp(min=0)  # 0 = the newest vector
        tokens = self.project(x) + self.position(back)
        tokens = torch.cat([self.summary.expand(n, -1, -1), tokens], dim=1)
        padding = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=x.device), ~mask], dim=1)
        return self.backbone(tokens, src_key_padding_mask=padding)[:, 0]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, domain: str) -> tuple[torch.Tensor, torch.Tensor]:
        """`(logits, Δt)` from domain `domain`'s head: logits (n, |C_d|), Δt (n,) in seconds."""
        out = self.heads[domain](self.backbone_out(x, mask))
        return out[:, :-1], F.softplus(out[:, -1])


def kappa(p: torch.Tensor) -> torch.Tensor:
    """Algorithm 4, step 3: 1 - H(p) / log(|C|), per row of probabilities `p` (n, |C|)."""
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    return 1.0 - entropy / math.log(p.shape[-1])


def domain_loss(logits: torch.Tensor, dt: torch.Tensor, y: torch.Tensor, tau: torch.Tensor, alpha: float) -> tuple[torch.Tensor, dict]:
    """`CE + α · MSE(Δt, τ)` for one domain's samples; the MSE on positives only (`τ` NaN on a negative)."""
    ce = F.cross_entropy(logits, y)
    positive = ~torch.isnan(tau)
    mse = F.mse_loss(dt[positive], tau[positive]) if positive.any() else dt.sum() * 0.0
    return ce + alpha * mse, {"ce": float(ce.detach()), "mse": float(mse.detach())}


def loss(outputs: dict[str, tuple], targets: dict[str, tuple], config: Config) -> tuple[torch.Tensor, dict]:
    """Appendix B.4: Σ_d λ_d [CE + α · MSE]. `outputs[d]` = (logits, Δt), `targets[d]` = (y, τ)."""
    total, parts = 0.0, {}
    for d, (logits, dt) in outputs.items():
        value, part = domain_loss(logits, dt, *targets[d], config.alpha)
        total = total + config.lam(d) * value
        parts[d] = part
    return total, parts
