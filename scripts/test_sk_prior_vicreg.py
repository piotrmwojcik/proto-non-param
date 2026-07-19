"""Self-check for the PMSN-style Sinkhorn prior and VICReg loss (runs T/U).

Run once before submitting the T/U training jobs:
    python scripts/test_sk_prior_vicreg.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from modeling.pnp import sinkhorn_knopp, vicreg_loss

torch.manual_seed(0)
B, V = 64, 100
logits = torch.randn(B, V) * 0.15  # cosine-scale

# 1) prior=None reproduces old behavior: rows sum 1, columns near-uniform
Q = sinkhorn_knopp(logits, eps=0.10, n_iter=3)
assert torch.allclose(Q.sum(dim=1), torch.ones(B), atol=1e-5), "rows must sum to 1"
col = Q.sum(dim=0) / B
assert col.max() / col.min() < 3.0, f"uniform prior: cols should be near-uniform, ratio={col.max()/col.min():.2f}"
print(f"uniform prior OK: col ratio {col.max()/col.min():.2f}")

# 2) non-uniform prior: column marginals should track the prior
prior = torch.linspace(1.0, 10.0, V)
prior = prior / prior.sum()
Qp = sinkhorn_knopp(logits, eps=0.10, n_iter=10, prior=prior)
assert torch.allclose(Qp.sum(dim=1), torch.ones(B), atol=1e-5), "rows must sum to 1"
colp = Qp.sum(dim=0) / B  # empirical column marginal
corr = torch.corrcoef(torch.stack([colp, prior]))[0, 1]
assert corr > 0.99, f"column marginal should track prior, corr={corr:.4f}"
print(f"power prior OK: marginal-prior corr {corr:.4f}")

# 3) vicreg: collapsed embeddings score much worse than spread ones
z_spread = F.normalize(torch.randn(64, 512), dim=-1)
z_collapsed = F.normalize(torch.randn(1, 512).repeat(64, 1) + 0.01 * torch.randn(64, 512), dim=-1)
l_spread = vicreg_loss(z_spread)
l_collapsed = vicreg_loss(z_collapsed)
assert l_collapsed > l_spread * 2, f"collapsed {l_collapsed:.3f} should be >> spread {l_spread:.3f}"
assert torch.isfinite(l_spread) and torch.isfinite(l_collapsed)
print(f"vicreg OK: spread {l_spread:.4f} << collapsed {l_collapsed:.4f}")

# 4) vicreg gradient flows
z = F.normalize(torch.randn(32, 512, requires_grad=True), dim=-1)
vicreg_loss(z).backward()
print("vicreg grad OK")
