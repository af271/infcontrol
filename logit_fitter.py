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

    def __init__(
        self,
        y: Any,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        split_seed: int = 42,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 1 or self.y.size == 0:
            raise ValueError("y must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(self.y)) or not np.all(np.isin(self.y, (0.0, 1.0))):
            raise ValueError("y must contain only finite 0 and 1 values")
        if validation_fraction <= 0 or test_fraction < 0:
            raise ValueError(
                "validation_fraction must be positive and test_fraction non-negative"
            )
        if validation_fraction + test_fraction >= 1:
            raise ValueError(
                "validation_fraction and test_fraction must sum to less than 1"
            )
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.train_indices, self.validation_indices, self.test_indices = (
            self._split_indices(
                self.y.size, validation_fraction, test_fraction, split_seed
            )
        )
        self.prior_probability = float(np.mean(self.y[self.train_indices]))
        prior_probability = np.clip(
            self.prior_probability,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
        self.initial_strength = float(
            np.log(prior_probability) - np.log1p(-prior_probability)
        )
        self.strengths = np.full(self.y.shape, self.initial_strength, dtype=float)
        self.coefficients: list[float] = []
        self.features: list[np.ndarray] = []
        self.losses: list[float] = []
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _split_indices(
        size: int, validation_fraction: float, test_fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.default_rng(seed).permutation(size)
        test_size = int(size * test_fraction)
        validation_size = max(1, int(size * validation_fraction))
        if test_fraction > 0:
            test_size = max(1, test_size)
        if validation_size + test_size >= size:
            raise ValueError(
                "dataset is too small for the requested train/validation/test split"
            )
        test_start = size - test_size
        validation_start = test_start - validation_size
        return (
            indices[:validation_start],
            indices[validation_start:test_start],
            indices[test_start:],
        )

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        probabilities = np.empty_like(logits, dtype=float)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return probabilities

    @staticmethod
    def _log_loss_for(logits: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))

    def _fit_coefficient(self, feature: np.ndarray) -> float:
        coefficient = 0.0
        train_feature = feature[self.train_indices]
        train_labels = self.y[self.train_indices]
        train_strengths = self.strengths[self.train_indices]
        for _ in range(self.max_iter):
            logits = train_strengths + coefficient * train_feature
            probabilities = self._sigmoid(logits)
            gradient = float(np.dot(train_feature, probabilities - train_labels))
            hessian = float(
                np.dot(
                    train_feature * train_feature, probabilities * (1.0 - probabilities)
                )
            )
            if hessian <= np.finfo(float).tiny:
                break

            step = gradient / hessian
            if abs(step) <= self.tol:
                break

            current_loss = self._log_loss_for(logits, train_labels)
            step_scale = 1.0
            while step_scale > 1e-12:
                candidate = coefficient - step_scale * step
                candidate_loss = self._log_loss_for(
                    train_strengths + candidate * train_feature, train_labels
                )
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
        baseline_loss = self._log_loss_for(
            self.strengths[self.train_indices], self.y[self.train_indices]
        )
        validation_baseline = self.validation_log_loss

        coefficient = self._fit_coefficient(values)
        candidate_strengths = self.strengths + coefficient * values
        candidate_loss = self._log_loss_for(
            candidate_strengths[self.train_indices], self.y[self.train_indices]
        )
        validation_candidate = self._log_loss_for(
            self.strengths[self.validation_indices]
            + coefficient * values[self.validation_indices],
            self.y[self.validation_indices],
        )

        if not np.isfinite(candidate_loss) or candidate_loss >= baseline_loss:
            return None
        if (
            not np.isfinite(validation_candidate)
            or validation_candidate >= validation_baseline
        ):
            return None

        self.strengths = candidate_strengths
        self.coefficients.append(coefficient)
        self.features.append(values.copy())
        self.losses.append(candidate_loss)
        return coefficient

    @property
    def validation_log_loss(self) -> float:
        """Return the current mean validation Bernoulli negative log-likelihood."""
        return self._log_loss_for(
            self.strengths[self.validation_indices], self.y[self.validation_indices]
        )

    def log_loss(self, split: str = "train") -> float:
        """Return the current mean Bernoulli negative log-likelihood."""
        try:
            indices = {
                "train": self.train_indices,
                "validation": self.validation_indices,
                "test": self.test_indices,
            }[split]
        except KeyError as exc:
            raise ValueError("split must be 'train', 'validation', or 'test'") from exc
        return self._log_loss_for(self.strengths[indices], self.y[indices])

    @property
    def probabilities(self) -> np.ndarray:
        """Return positive-class probabilities for the current strengths."""
        return self._sigmoid(self.strengths)


__all__ = ["LogitFitter"]
