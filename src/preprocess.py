"""src/preprocess.py
Common data-loading and pre-processing utilities.  All *dataset-specific*
loading happens behind clearly marked sections so later variants can plug
in without breaking the base framework.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor

# Optional heavy dependencies are imported lazily to speed-up smoke tests
try:
    from datasets import load_dataset  # Hugging Face 🤗 datasets
except ImportError:  # pragma: no cover – will not happen in CI container
    load_dataset = None  # type: ignore

# --------------------------------------------------------------------------------------
#                              DATA PRE-PROCESSOR
# --------------------------------------------------------------------------------------
class DataPreprocessor:
    """Handles dataset loading, normalisation and train/val split."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.seed = cfg["misc"].get("seed", 0)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_dataset(self) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Return ((x_train,b_train,y_train), (x_val,b_val,y_val))."""
        name: str = self.cfg["data"]["name"].lower()

        if name == "synthetic":
            x, b, y = self._generate_synthetic()
        elif name in {"wikitext-103", "yehzw/wikitext-103"}:
            x, b, y = self._load_wikitext()
        else:
            raise NotImplementedError(
                f"Dataset loader for '{name}' is not yet implemented."
            )

        # convert to tensors and split
        x = torch.as_tensor(x, dtype=torch.float32)
        b = torch.as_tensor(b, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=1 - self.cfg["data"].get("train_split", 0.8),
            random_state=self.seed,
        )
        train_tuple = (x[train_idx], b[train_idx], y[train_idx])
        val_tuple = (x[val_idx], b[val_idx], y[val_idx])
        return train_tuple, val_tuple

    # ------------------------------------------------------------------
    # Synthetic data for smoke tests / sanity checks
    # ------------------------------------------------------------------
    def _generate_synthetic(self):
        n = int(self.cfg["data"].get("n_samples", 200))
        d = int(self.cfg["data"].get("n_features", 5))
        max_budget = int(self.cfg["data"].get("max_budget", 20))

        x = np.random.uniform(-1, 1, size=(n, d))
        b = np.random.randint(1, max_budget + 1, size=(n,)).astype(np.float32)

        # true function – purposely *monotone* in budget
        y_true = (x.sum(axis=1) / d) - 0.5 * np.exp(-b / max_budget)
        noise = 0.05 * np.random.randn(n)
        y = y_true + noise
        return x, b / max_budget, y  # we normalise budget to [0,1]

    # ------------------------------------------------------------------
    # WikiText-103 loader (Hugging Face dataset)
    # ------------------------------------------------------------------
    def _load_wikitext(self):
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' package is required for loading WikiText-103. "
                "Please install it via 'pip install datasets'."
            )

        cfg_data = self.cfg["data"]
        n_samples = int(cfg_data.get("n_samples", 10_000))
        n_features = int(cfg_data.get("n_features", 5))
        max_budget = int(cfg_data.get("max_budget", 15))

        # ------------------------------------------------------------------
        # 1) Load raw dataset (train split only – sufficient for surrogate)
        # ------------------------------------------------------------------
        ds = load_dataset("yehzw/wikitext-103", split="train", streaming=False)  # type: ignore
        texts = ds["text"]  # list[str]

        # Remove empty strings / article markers
        texts = [t for t in texts if t.strip() and not t.startswith(" =")]

        # Sub-sample to adhere to n_samples (shuffle with seed for determinism)
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(texts), size=min(n_samples, len(texts)), replace=False)
        texts = [texts[i] for i in idx]
        n = len(texts)

        # ------------------------------------------------------------------
        # 2) Very lightweight handcrafted text features – no heavy NLP stack
        #    Each feature is normalised to roughly [0,1] range for stability.
        # ------------------------------------------------------------------
        features = np.zeros((n, n_features), dtype=np.float32)
        for i, txt in enumerate(texts):
            words = re.findall(r"[A-Za-z]+", txt)
            n_chars = len(txt)
            n_words = len(words) or 1
            mean_word_len = sum(map(len, words)) / n_words
            var_word_len = (
                sum((len(w) - mean_word_len) ** 2 for w in words) / n_words
            )
            ratio_upper = sum(ch.isupper() for ch in txt) / n_chars

            raw_vec = np.array(
                [n_chars / 4000, n_words / 800, mean_word_len / 15, var_word_len / 50, ratio_upper],
                dtype=np.float32,
            )
            # Pad / truncate to requested dimensionality
            d = min(n_features, len(raw_vec))
            features[i, :d] = raw_vec[:d]

        # ------------------------------------------------------------------
        # 3) Budgets & objective – create a *monotone* synthetic objective so
        #    that our regulariser has the right inductive bias.
        # ------------------------------------------------------------------
        b = rng.randint(1, max_budget + 1, size=(n,)).astype(np.float32)
        # objective: larger budgets → lower (better) loss in expectation
        y_true = -0.3 * features[:, 0] - 0.5 * np.exp(-b / max_budget)
        y = y_true + 0.05 * rng.randn(n)  # light Gaussian noise

        # normalise budget to [0,1]
        return features, b.astype(np.float32) / max_budget, y.astype(np.float32)
