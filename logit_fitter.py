"""Greedy, stagewise fitting of logistic features."""

from __future__ import annotations

from typing import Any

import numpy as np


class LogitFitter:
    """Fit one logistic feature coefficient at a time.

    The model's current logits are stored in ``strengths``. For every feature
    passed to :meth:`evaluate`, only its scalar coefficient is optimized:

    ``probability = sigmoid(strengths + coefficient * feature)``

    Accepted coefficients are appended to ``coefficients`` and their feature
    is added to ``features``. Rejected features do not change the model state.
    """

    def __init__(self, y: Any, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 1 or self.y.size == 0:
            raise ValueError("y must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(self.y)) or not np.all(np.isin(self.y, (0.0, 1.0))):
            raise ValueError("y must contain only finite 0 and 1 values")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.strengths = np.zeros(self.y.shape, dtype=float)
        self.coefficients: list[float] = []
        self.features: list[np.ndarray] = []
        self.losses: list[float] = []
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        probabilities = np.empty_like(logits, dtype=float)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return probabilities

    def _log_loss(self, logits: np.ndarray) -> float:
        return float(np.mean(np.logaddexp(0.0, logits) - self.y * logits))

    def _fit_coefficient(self, feature: np.ndarray) -> float:
        coefficient = 0.0
        for _ in range(self.max_iter):
            logits = self.strengths + coefficient * feature
            probabilities = self._sigmoid(logits)
            gradient = float(np.dot(feature, probabilities - self.y))
            hessian = float(
                np.dot(feature * feature, probabilities * (1.0 - probabilities))
            )
            if hessian <= np.finfo(float).tiny:
                break

            step = gradient / hessian
            if abs(step) <= self.tol:
                break

            current_loss = self._log_loss(logits)
            step_scale = 1.0
            while step_scale > 1e-12:
                candidate = coefficient - step_scale * step
                candidate_loss = self._log_loss(self.strengths + candidate * feature)
                if candidate_loss <= current_loss:
                    coefficient = candidate
                    break
                step_scale *= 0.5
            else:
                break

            if abs(step_scale * step) <= self.tol:
                break

        return coefficient

    def evaluate(self, feature: Any) -> float | None:
        """Fit and, if useful, accept one feature coefficient.

        Returns the fitted coefficient for an accepted feature and ``None`` if
        the feature does not strictly improve the current log-loss.
        """
        values = np.asarray(feature, dtype=float)
        if values.shape != self.y.shape:
            raise ValueError(
                f"feature must have shape {self.y.shape}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("feature must contain only finite values")

        baseline_loss = self._log_loss(self.strengths)

        coefficient = self._fit_coefficient(values)
        candidate_strengths = self.strengths + coefficient * values
        candidate_loss = self._log_loss(candidate_strengths)

        if not np.isfinite(candidate_loss) or candidate_loss >= baseline_loss:
            return None

        self.strengths = candidate_strengths
        self.coefficients.append(coefficient)
        self.features.append(values.copy())
        self.losses.append(candidate_loss)
        return coefficient

    @property
    def log_loss(self) -> float:
        """Return the current mean Bernoulli negative log-likelihood."""
        return self._log_loss(self.strengths)

    @property
    def probabilities(self) -> np.ndarray:
        """Return positive-class probabilities for the current strengths."""
        return self._sigmoid(self.strengths)


__all__ = ["LogitFitter"]
