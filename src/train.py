"""src/train.py
Core training logic that implements the DyHPO-style GP surrogate with the
monotonicity regulariser.  Only dataset/model-specific parts are expressed as
place-holders in preprocess.py; every bit of the optimisation logic here is
complete and production-ready.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import gpytorch
import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import nn, Tensor
from tqdm import trange

# --------------------------------------------------------------------------------------
#                               GP SURROGATE + REGULARISER
# --------------------------------------------------------------------------------------
class DeepFeatureExtractor(nn.Module):
    """A small MLP that plays the role of DyHPO's deep kernel feature extractor."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: Tensor) -> Tensor:  # (N, d) -> (N, out_dim)
        return self.net(x)


class SurrogateGPModel(gpytorch.models.ExactGP):
    """Exact GP with a deep kernel; works for small tabular MF-HPO data sets."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood,
                 feature_extractor: nn.Module):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --------------------------------------------------------------------------------------
#                                TRAINING UTILITIES
# --------------------------------------------------------------------------------------

def compute_monotone_penalty(pred_mean: Tensor, budgets: Tensor) -> Tensor:
    """Hinge-squared penalty that enforces m(x,b_low) ≤ m(x,b_high).

    Args
    ----
    pred_mean : (N,) predictive means for the batch
    budgets   : (N,) corresponding scalar budgets (unnormalised or normalised)
    """
    idx_i, idx_j = torch.triu_indices(len(budgets), len(budgets), offset=1, device=pred_mean.device)
    mask = budgets[idx_i] < budgets[idx_j]  # ONLY strictly increasing pairs
    if not mask.any():
        # Single budget in batch – no penalty.
        return torch.tensor(0.0, device=pred_mean.device)
    diff = pred_mean[idx_i[mask]] - pred_mean[idx_j[mask]]
    penalty = torch.relu(diff).pow(2).mean()
    return penalty


class SurrogateTrainer:
    """End-to-end optimiser for the GP surrogate.

    This class is deliberately *model-agnostic*: it will work unchanged for any
    ExactGP that exposes a differentiable mean vector, hence future
    model-specific variants can plug-in without touching the training logic.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg["misc"].get("seed", 0))
        self.logs: Dict[str, list] = {"train_loss": [], "val_rmse": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train_tuple: Tuple[Tensor, Tensor, Tensor],
            val_tuple: Tuple[Tensor, Tensor, Tensor]) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.GaussianLikelihood, Dict]:
        """Train the GP and return it together with the likelihood and logs."""
        train_x, train_b, train_y = train_tuple
        val_x, val_b, val_y = val_tuple

        # concat features and budget so that the GP sees the budget as an input
        train_inputs = torch.cat([train_x, train_b.unsqueeze(-1)], dim=-1).to(self.device)
        val_inputs = torch.cat([val_x, val_b.unsqueeze(-1)], dim=-1).to(self.device)
        train_targets = train_y.to(self.device)
        val_targets = val_y.to(self.device)

        feature_extractor = DeepFeatureExtractor(in_dim=train_inputs.shape[1],
                                                 hidden_dim=self.cfg["model"].get("hidden_dim", 64),
                                                 out_dim=32).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = SurrogateGPModel(train_inputs, train_targets, likelihood, feature_extractor).to(self.device)

        model.train()
        likelihood.train()

        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["train"].get("lr", 1e-2))

        lam = float(self.cfg.get("monotonicity_lambda", 0.1))
        epochs = int(self.cfg["train"].get("epochs", 100))

        for epoch in trange(epochs, desc="Training GP", leave=False):
            optimizer.zero_grad()
            output = model(train_inputs)
            nll = -mll(output, train_targets)
            mono_pen = compute_monotone_penalty(output.mean, train_b.to(self.device))
            loss = nll + lam * mono_pen  # <-- crucial: **no** gradient blockage here!
            loss.backward()
            optimizer.step()

            # ----------------------- validation -----------------------
            model.eval(); likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(val_inputs))
                rmse = torch.sqrt(torch.mean((pred_dist.mean - val_targets) ** 2)).item()

            self.logs["train_loss"].append(loss.item())
            self.logs["val_rmse"].append(rmse)
            model.train(); likelihood.train()

        return model, likelihood, self.logs

    # ------------------------------------------------------------------
    # I/O helpers – standalone because main.py needs them
    # ------------------------------------------------------------------
    @staticmethod
    def save_logs(logs: Dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fp:
            json.dump(logs, fp, indent=2)

    @staticmethod
    def load_logs(path: Path) -> Dict:
        with path.open() as fp:
            return json.load(fp)
