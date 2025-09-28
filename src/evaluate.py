"""src/evaluate.py
Universal evaluation and visualisation utilities shared by *all* experimental
variants.  No dataset/model specifics appear in this file – these are already
handled upstream.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import Tensor

# --------------------------------------------------------------------------------------
#                              METRIC COMPUTATION
# --------------------------------------------------------------------------------------

def rmse(pred: Tensor, target: Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


@torch.no_grad()
def monotonicity_violation_rate(pred_mean: Tensor, budgets: Tensor) -> float:
    """Fraction of pairwise violations – lower is better."""
    idx_i, idx_j = torch.triu_indices(len(budgets), len(budgets), offset=1, device=pred_mean.device)
    mask = budgets[idx_i] < budgets[idx_j]
    if not mask.any():
        return 0.0
    violations = (pred_mean[idx_i[mask]] > pred_mean[idx_j[mask]]).float()
    return violations.mean().item()


@torch.no_grad()
def evaluate(model: gpytorch.models.ExactGP, likelihood: gpytorch.likelihoods.GaussianLikelihood,
             data_tuple: Tuple[Tensor, Tensor, Tensor]) -> Dict:
    """Compute RMSE, NLL and monotonicity violation rate on *any* dataset."""
    model.eval(); likelihood.eval()
    x, b, y = data_tuple
    device = next(model.parameters()).device
    inputs = torch.cat([x, b.unsqueeze(-1)], dim=-1).to(device)
    y = y.to(device)

    with gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(inputs))
        nll = -pred_dist.log_prob(y).mean().item()
        _rmse = rmse(pred_dist.mean, y)
        mono_rate = monotonicity_violation_rate(pred_dist.mean, b.to(device))

    return {"rmse": _rmse, "nll": nll, "violation_rate": mono_rate}

# --------------------------------------------------------------------------------------
#                              VISUALISATION
# --------------------------------------------------------------------------------------

def _annotate(ax):
    for line in ax.lines:
        x, y = line.get_xdata(), line.get_ydata()
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)


def plot_training_curves(logs: Dict, images_dir: Path, exp_name: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)

    # --------------- training loss ---------------
    plt.figure(figsize=(4, 3))
    sns.lineplot(x=list(range(len(logs["train_loss"]))), y=logs["train_loss"], label="Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
    _annotate(plt.gca())
    plt.legend()
    fname = images_dir / f"training_loss_{exp_name}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # --------------- validation RMSE -------------
    plt.figure(figsize=(4, 3))
    sns.lineplot(x=list(range(len(logs["val_rmse"]))), y=logs["val_rmse"], label="Validation RMSE", color="darkorange")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Validation RMSE")
    _annotate(plt.gca())
    plt.legend()
    fname2 = images_dir / f"validation_rmse_{exp_name}.pdf"
    plt.savefig(fname2, bbox_inches="tight")
    plt.close()

    print("Generated figures:")
    for f in [fname, fname2]:
        print(f" - {f.name}")
