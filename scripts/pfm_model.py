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

**The input** (`Config.input`, D-61): `raw`, the clip vectors as they are (Algorithm 4), or `minus_first`, each
vector minus its window's first vector, so what never changes in the window (the place) cancels out and only what
changes is left (OMNI-914, K-36). Done inside the model, so serving it needs nothing else.

**The labels** (`Config.labels`, D-62): `classes`, each domain's own classes (as above), or `event`, every event class
merged into one, so a head answers only "an event is coming" or Normal (OMNI-916). CCD is the same either way.

**Distance weights** (`Config.distance_weights`, Fix 1 of D-65, OMNI-X19): with a domain's `(τ0, s)` set, an event
sample's cross-entropy is multiplied by `w = exp(-max(0, τ - τ0) / s)` and a normal sample's by 1, and the class
error is `Σ w · CE / Σ w`, so a second long before the event, where nothing is visible yet, counts less
(DSA / DSTA's exponential weighting). D-65's settings are `DISTANCE_WEIGHTS`. Empty (the default, and every
checkpoint saved before it) is the plain mean. The time error is unchanged.

**Ranking inside the same window** (`Config.rank_gaps`, Fix 2 of D-65, OMNI-X20): with a domain's gap set, each step
also scores two seconds of each of `rank_windows` train windows, the later at least the gap after the earlier
(`train_pfm.draw_pairs`), and adds `λ_rank` × the mean of `ranking_loss`: on an event window the later must score
higher by `rank_margin`, on a normal window it must not score higher at all (Ma et al., CVPR 2016). The score is
`1 - p(Normal)`. D-65's gaps are `RANK_GAPS`. Empty (the default, and every older checkpoint) is off.

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


INPUTS = ("raw", "minus_first")
DISTANCE_WEIGHTS = {"ucf_crime": [8.0, 10.0], "ccd": [0.0, 1.0]}  # D-65: (τ0, s) in seconds; CCD is DSTA's
RANK_GAPS = {"ucf_crime": 5.0, "ccd": 1.0}  # D-65: the later second at least this many seconds after the earlier
LABELS = ("classes", "event")


def class_names(domain: str, labels: str = "classes") -> list[str]:
    """The classes a head of `domain` answers with: its own, or Normal and one merged Event (D-62)."""
    names = CLASSES[domain]
    return names if labels == "classes" or len(names) == 2 else ["Normal", "Event"]


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
    alphas: dict[str, float] = field(default_factory=dict)  # per domain (D-59: 1 / var τ); missing = alpha
    lambdas: dict[str, float] = field(default_factory=dict)  # per domain; missing = 1.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch: int = 256
    epochs: int = 30
    input: str = "raw"  # or "minus_first" (D-61); a checkpoint saved before D-61 is raw
    labels: str = "classes"  # or "event" (D-62); a checkpoint saved before D-62 is classes
    distance_weights: dict[str, list[float]] = field(default_factory=dict)  # per domain (τ0, s), D-65; missing = off
    rank_gaps: dict[str, float] = field(default_factory=dict)  # per domain, Fix 2 (D-65); missing = no ranking
    rank_windows: int = 64  # windows per domain in the ranking batch
    rank_margin: float = 0.1
    lambda_rank: float = 1.0

    def __post_init__(self) -> None:
        if self.input not in INPUTS:
            raise ValueError(f"input {self.input!r}, expected one of {INPUTS}")
        if self.labels not in LABELS:
            raise ValueError(f"labels {self.labels!r}, expected one of {LABELS}")

    def lam(self, domain: str) -> float:
        return self.lambdas.get(domain, 1.0)

    def alpha_of(self, domain: str) -> float:
        return self.alphas.get(domain, self.alpha)

    def weights_of(self, domain: str) -> tuple[float, float] | None:
        w = self.distance_weights.get(domain)
        return None if w is None else (float(w[0]), float(w[1]))

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
        self.heads = nn.ModuleDict({d: nn.Linear(config.width, len(class_names(d, config.labels)) + 1) for d in config.domains})
        nn.init.normal_(self.summary, std=0.02)

    def backbone_out(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`h` (n, width) for windows `x` (n, L, input_dim), `mask` (n, L) True on the sequence."""
        n, length, _ = x.shape
        if length > self.config.max_length:
            raise ValueError(f"{length} vectors, the model takes at most {self.config.max_length}")
        if self.config.input == "minus_first":  # the window's first vector is always at position 0
            x = (x - x[:, :1]) * mask.unsqueeze(-1)
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


def distance_weight(tau: torch.Tensor, tau0: float, s: float) -> torch.Tensor:
    """Fix 1 (D-65): `exp(-max(0, τ - τ0) / s)` on an event sample, 1 on a normal one (`τ` NaN)."""
    w = torch.exp(-(torch.nan_to_num(tau, nan=0.0) - tau0).clamp(min=0) / s)
    return torch.where(torch.isnan(tau), torch.ones_like(w), w)


def domain_loss(logits: torch.Tensor, dt: torch.Tensor, y: torch.Tensor, tau: torch.Tensor, alpha: float,
                weights: tuple[float, float] | None = None) -> tuple[torch.Tensor, dict]:
    """`CE + α · MSE(Δt, τ)` for one domain's samples; the MSE on positives only (`τ` NaN on a negative).
    With `weights = (τ0, s)` the CE is the distance-weighted mean `Σ w · CE / Σ w` (Fix 1)."""
    if weights is None:
        ce = F.cross_entropy(logits, y)
    else:
        w = distance_weight(tau, *weights)
        ce = (w * F.cross_entropy(logits, y, reduction="none")).sum() / w.sum()
    positive = ~torch.isnan(tau)
    mse = F.mse_loss(dt[positive], tau[positive]) if positive.any() else dt.sum() * 0.0
    return ce + alpha * mse, {"ce": float(ce.detach()), "mse": float(mse.detach())}


def ranking_loss(early: torch.Tensor, late: torch.Tensor, event: torch.Tensor, margin: float) -> torch.Tensor:
    """Fix 2 (D-65): the mean over pairs of `max(0, margin - (late - early))` on an event window and
    `max(0, late - early)` on a normal one; `early`, `late` are `1 - p(Normal)` of the two seconds."""
    rise = late - early
    return torch.where(event, F.relu(margin - rise), F.relu(rise)).mean()


def loss(outputs: dict[str, tuple], targets: dict[str, tuple], config: Config) -> tuple[torch.Tensor, dict]:
    """Appendix B.4: Σ_d λ_d [CE + α · MSE], the CE distance-weighted where `config` says (Fix 1).
    `outputs[d]` = (logits, Δt), `targets[d]` = (y, τ)."""
    total, parts = 0.0, {}
    for d, (logits, dt) in outputs.items():
        value, part = domain_loss(logits, dt, *targets[d], config.alpha_of(d), config.weights_of(d))
        total = total + config.lam(d) * value
        parts[d] = part
    return total, parts
