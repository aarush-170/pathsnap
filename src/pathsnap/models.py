"""Candidate classifiers compared in Phase 4.

Every model exposes the same interface:  fit(X, y, groups) -> self ;  predict_proba(X) -> P(is_true)
so the training script treats them identically. Configurations here are the *untuned* starting
points; Phase 5 tunes the selected model with Bayesian optimisation.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ----------------------------------------------------------------------------------- wrappers
class SklearnModel:
    """Wraps a sklearn estimator; optional row subsampling for models that do not scale to 1M rows."""

    def __init__(self, make, subsample: int | None = None, seed: int = 42):
        self.make, self.subsample, self.seed = make, subsample, seed
        self.history = None

    def fit(self, X, y, groups=None):
        if self.subsample and len(X) > self.subsample:
            idx = np.random.default_rng(self.seed).choice(len(X), self.subsample, replace=False)
            X, y = X[idx], y[idx]
        self.est = self.make()
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)[:, 1]


class LGBMModel:
    def __init__(self, params: dict, seed: int = 42):
        self.params, self.seed, self.history = params, seed, None

    def fit(self, X, y, groups=None):
        import lightgbm as lgb
        self.est = lgb.LGBMClassifier(**self.params, random_state=self.seed, verbose=-1)
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)[:, 1]


class XGBModel:
    def __init__(self, params: dict, seed: int = 42):
        self.params, self.seed, self.history = params, seed, None

    def fit(self, X, y, groups=None):
        import xgboost as xgb
        self.est = xgb.XGBClassifier(**self.params, random_state=self.seed)
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)[:, 1]


class TorchMLP:
    """Feed-forward network trained with BCE loss and early stopping on held-out *trips*.

    Records an epoch-by-epoch train/validation loss history (the only model with real
    training curves).
    """

    def __init__(self, hidden=(128, 64), dropout=0.1, lr=1e-3, batch=2048, max_epochs=30, patience=4,
                 val_frac=0.1, weight_decay=1e-5, seed: int = 42):
        self.cfg = dict(hidden=hidden, dropout=dropout, lr=lr, batch=batch, max_epochs=max_epochs,
                        patience=patience, val_frac=val_frac, weight_decay=weight_decay)
        self.seed = seed
        self.history = None

    def _net(self, d_in):
        import torch.nn as nn
        layers, prev = [], d_in
        for h in self.cfg["hidden"]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.cfg["dropout"])]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, groups):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ug = np.unique(groups)
        val_g = set(rng.choice(ug, max(1, int(round(len(ug) * self.cfg["val_frac"]))), replace=False))
        vm = np.isin(groups, list(val_g))
        self.scaler = StandardScaler().fit(X[~vm])
        Xt = torch.tensor(self.scaler.transform(X[~vm]), dtype=torch.float32)
        yt = torch.tensor(y[~vm], dtype=torch.float32)
        Xv = torch.tensor(self.scaler.transform(X[vm]), dtype=torch.float32)
        yv = torch.tensor(y[vm], dtype=torch.float32)
        self.net = self._net(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        lossf = torch.nn.BCEWithLogitsLoss()
        best, best_state, bad, hist = np.inf, None, 0, []
        n, b = len(Xt), self.cfg["batch"]
        for ep in range(1, self.cfg["max_epochs"] + 1):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, b):
                idx = perm[i:i + b]
                opt.zero_grad()
                loss = lossf(self.net(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                vl = lossf(self.net(Xv).squeeze(1), yv).item()
            hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl))
            if vl < best - 1e-5:
                best, best_state, bad = vl, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.history = hist
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(self.scaler.transform(X), dtype=torch.float32)).squeeze(1)
        return torch.sigmoid(z).numpy()


# ----------------------------------------------------------------------------------- registry
@dataclass
class ModelSpec:
    name: str
    family: str
    config: dict
    build: callable = field(repr=False)
    note: str = ""


def registry(seed: int = 42) -> dict[str, ModelSpec]:
    lr_cfg = dict(C=1.0, penalty="l2", max_iter=1000, scaling="standard")
    knn_cfg = dict(n_neighbors=25, weights="distance", scaling="standard", train_subsample=200_000)
    rf_cfg = dict(n_estimators=200, max_depth=None, min_samples_leaf=5, max_features="sqrt")
    lgb_cfg = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
    xgb_cfg = dict(n_estimators=400, learning_rate=0.1, max_depth=8, min_child_weight=5, subsample=0.8,
                   colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="logloss")
    mlp_cfg = dict(hidden=(128, 64), dropout=0.1, lr=1e-3, batch=2048, max_epochs=30, patience=4,
                   val_frac=0.1, weight_decay=1e-5)
    svm_cfg = dict(C=1.0, scaling="standard", calibration="sigmoid, 3-fold", train_subsample=200_000)

    specs = [
        ModelSpec("logreg", "Linear", lr_cfg, lambda: SklearnModel(lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)), seed=seed)),
        ModelSpec("gnb", "Probabilistic", {}, lambda: SklearnModel(GaussianNB, seed=seed),
                  "assumes features independent and Gaussian within each class"),
        ModelSpec("knn", "Instance-based", knn_cfg, lambda: SklearnModel(lambda: make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
            subsample=200_000, seed=seed), "trained on a 200k-row random sample per fit (memory/time)"),
        ModelSpec("random_forest", "Bagging (tree ensemble)", rf_cfg, lambda: SklearnModel(
            lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                           n_jobs=-1, random_state=seed), seed=seed)),
        ModelSpec("lightgbm", "Boosting", lgb_cfg, lambda: LGBMModel(lgb_cfg, seed=seed)),
        ModelSpec("xgboost", "Boosting", xgb_cfg, lambda: XGBModel(xgb_cfg, seed=seed)),
        ModelSpec("mlp", "Neural network", mlp_cfg, lambda: TorchMLP(**mlp_cfg, seed=seed),
                  "early stopping on 10% of training trips"),
        ModelSpec("linear_svm", "Kernel-based (linear)", svm_cfg, lambda: SklearnModel(lambda: make_pipeline(
            StandardScaler(), CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=3)),
            subsample=200_000, seed=seed), "optional; Platt-calibrated for probabilities; 200k-row sample"),
    ]
    return {s.name: s for s in specs}


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0
