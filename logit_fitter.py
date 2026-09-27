"""Greedy, stagewise fitting of logistic features."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


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
        groups: Any,
        group_order: Any | None = None,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        split_seed: int = 42,
        max_iter: int = 100,
        tol: float = 1e-10,
        objective: str = "log_loss",
        acceptance_metric: str = "log_loss",
        recall_target: float = 0.8,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 1 or self.y.size == 0:
            raise ValueError("y must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(self.y)) or not np.all(np.isin(self.y, (0.0, 1.0))):
            raise ValueError("y must contain only finite 0 and 1 values")
        self.groups = np.asarray(groups, dtype=object)
        if self.groups.shape != self.y.shape:
            raise ValueError(
                f"groups must have shape {self.y.shape}, got {self.groups.shape}"
            )
        self.group_order = (
            None if group_order is None else np.asarray(group_order, dtype=float)
        )
        if self.group_order is not None and self.group_order.shape != self.y.shape:
            raise ValueError(
                f"group_order must have shape {self.y.shape}, got {self.group_order.shape}"
            )
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
        if objective not in {"log_loss", "pairwise_rank"}:
            raise ValueError("objective must be 'log_loss' or 'pairwise_rank'")
        if acceptance_metric not in {"log_loss", "k_at_recall"}:
            raise ValueError("acceptance_metric must be 'log_loss' or 'k_at_recall'")
        if not 0 < recall_target <= 1:
            raise ValueError("recall_target must be greater than 0 and at most 1")
        self.train_indices, self.validation_indices, self.test_indices = (
            self._split_indices(
                self.groups,
                validation_fraction,
                test_fraction,
                split_seed,
                self.group_order,
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
        self.objective = objective
        self.acceptance_metric = acceptance_metric
        self.recall_target = recall_target

    @staticmethod
    def _split_indices(
        groups: np.ndarray,
        validation_fraction: float,
        test_fraction: float,
        seed: int,
        group_order: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        group_names = np.asarray([str(group) for group in groups], dtype=object)
        unique_groups, inverse = np.unique(group_names, return_inverse=True)
        if group_order is None:
            shuffled_groups = np.random.default_rng(seed).permutation(
                len(unique_groups)
            )
        else:
            first_order = np.full(len(unique_groups), np.inf)
            for row_index, group_index in enumerate(inverse):
                first_order[group_index] = min(
                    first_order[group_index], group_order[row_index]
                )
            shuffled_groups = np.argsort(first_order)
        test_size = int(len(unique_groups) * test_fraction)
        validation_size = max(1, int(len(unique_groups) * validation_fraction))
        if test_fraction > 0:
            test_size = max(1, test_size)
        if validation_size + test_size >= len(unique_groups):
            raise ValueError(
                "dataset is too small for the requested train/validation/test split"
            )
        test_start = len(unique_groups) - test_size
        validation_start = test_start - validation_size
        group_slices = (
            shuffled_groups[:validation_start],
            shuffled_groups[validation_start:test_start],
            shuffled_groups[test_start:],
        )
        return (
            np.flatnonzero(np.isin(inverse, group_slices[0])),
            np.flatnonzero(np.isin(inverse, group_slices[1])),
            np.flatnonzero(np.isin(inverse, group_slices[2])),
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

    def _pairwise_pairs(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grouped: dict[str, list[int]] = {}
        for index in indices:
            grouped.setdefault(str(self.groups[index]), []).append(int(index))
        positive_indices = []
        negative_indices = []
        for group_indices in grouped.values():
            positives = [index for index in group_indices if self.y[index] == 1]
            negatives = [index for index in group_indices if self.y[index] == 0]
            for positive in positives:
                for negative in negatives:
                    positive_indices.append(positive)
                    negative_indices.append(negative)
        return np.asarray(positive_indices, dtype=int), np.asarray(
            negative_indices, dtype=int
        )

    @staticmethod
    def _pairwise_loss_for(
        logits: np.ndarray, positive_indices: np.ndarray, negative_indices: np.ndarray
    ) -> float:
        if not len(positive_indices):
            return float("nan")
        margins = logits[positive_indices] - logits[negative_indices]
        return float(np.mean(np.logaddexp(0.0, -margins)))

    def _k_at_recall_for(self, logits: np.ndarray, indices: np.ndarray) -> float:
        grouped: dict[str, list[tuple[float, bool]]] = {}
        probabilities = self._sigmoid(logits)
        for index in indices:
            grouped.setdefault(str(self.groups[index]), []).append(
                (float(probabilities[index]), bool(self.y[index]))
            )
        required = []
        for values in grouped.values():
            total = sum(label for _, label in values)
            if not total:
                continue
            target_hits = int(np.ceil(self.recall_target * total))
            hits = 0
            for rank, (_, label) in enumerate(
                sorted(values, key=lambda item: item[0], reverse=True), start=1
            ):
                hits += label
                if hits >= target_hits:
                    required.append(rank)
                    break
        return float(np.mean(required)) if required else float("nan")

    def _objective_for(self, logits: np.ndarray, indices: np.ndarray) -> float:
        if self.objective == "pairwise_rank":
            positive, negative = self._pairwise_pairs(indices)
            loss = self._pairwise_loss_for(logits, positive, negative)
            if np.isfinite(loss):
                return loss
        return self._log_loss_for(logits[indices], self.y[indices])

    def _fit_coefficient(self, feature: np.ndarray) -> float:
        if self.objective == "pairwise_rank":
            train_indices = self.train_indices
            baseline_strengths = self.strengths.copy()

            def objective(parameter: np.ndarray) -> float:
                logits = baseline_strengths + parameter[0] * feature
                return self._objective_for(logits, train_indices)

            result = minimize(
                objective,
                x0=np.zeros(1),
                method="BFGS",
                options={"gtol": self.tol, "maxiter": self.max_iter},
            )
            return float(result.x[0])

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
        baseline_loss = self._objective_for(self.strengths, self.train_indices)
        validation_baseline = self._acceptance_value(
            self.strengths, self.validation_indices
        )

        coefficient = self._fit_coefficient(values)
        candidate_strengths = self.strengths + coefficient * values
        candidate_loss = self._objective_for(candidate_strengths, self.train_indices)
        validation_candidate = self._acceptance_value(
            candidate_strengths, self.validation_indices
        )

        if not np.isfinite(candidate_loss) or candidate_loss >= baseline_loss:
            return None
        if (
            not np.isfinite(validation_candidate)
            or validation_candidate >= validation_baseline
        ):
            return None

        old_coefficients = self.coefficients.copy()
        old_features = self.features.copy()
        old_strengths = self.strengths.copy()

        self.coefficients.append(coefficient)
        self.features.append(values.copy())
        refit_coefficients = self._refit_coefficients()
        refit_strengths = self.initial_strength + sum(
            coefficient * feature
            for coefficient, feature in zip(refit_coefficients, self.features)
        )
        refit_validation_loss = self._acceptance_value(
            refit_strengths, self.validation_indices
        )
        if (
            not np.isfinite(refit_validation_loss)
            or refit_validation_loss >= validation_baseline
        ):
            self.coefficients = old_coefficients
            self.features = old_features
            self.strengths = old_strengths
            return None

        self.coefficients = refit_coefficients.tolist()
        self.strengths = refit_strengths
        self.losses.append(
            self._log_loss_for(
                refit_strengths[self.train_indices], self.y[self.train_indices]
            )
        )
        return self.coefficients[-1]

    def _acceptance_value(self, logits: np.ndarray, indices: np.ndarray) -> float:
        if self.acceptance_metric == "k_at_recall":
            value = self._k_at_recall_for(logits, indices)
            if np.isfinite(value):
                return value
        return self._log_loss_for(logits[indices], self.y[indices])

    def _refit_coefficients(self) -> np.ndarray:
        """Refit all accepted feature coefficients on the training split."""
        feature_matrix = np.column_stack(
            [feature[self.train_indices] for feature in self.features]
        )
        train_labels = self.y[self.train_indices]
        train_strengths = np.full(train_labels.shape, self.initial_strength)

        def objective(coefficients: np.ndarray) -> float:
            logits = train_strengths + feature_matrix @ coefficients
            if self.objective == "pairwise_rank":
                full_logits = np.empty_like(self.strengths)
                full_logits[self.train_indices] = logits
                return self._objective_for(full_logits, self.train_indices)
            return self._log_loss_for(logits, train_labels)

        result = minimize(
            objective,
            x0=np.asarray(self.coefficients, dtype=float),
            method="BFGS",
            options={"gtol": self.tol, "maxiter": self.max_iter},
        )
        if not np.all(np.isfinite(result.x)):
            raise ValueError("coefficient refit produced non-finite values")
        return result.x

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

    def predict(self, features: Any) -> np.ndarray:
        """Predict positive-class probabilities from accepted feature columns."""
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError("features must be a one- or two-dimensional array")
        if values.shape[0] == self.y.size and values.shape[1] != len(self.coefficients):
            raise ValueError(
                f"features must have {len(self.coefficients)} columns, got {values.shape[1]}"
            )
        if values.shape[1] != len(self.coefficients):
            raise ValueError(
                f"features must have {len(self.coefficients)} columns, got {values.shape[1]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("features must contain only finite values")
        logits = self.initial_strength + values @ np.asarray(self.coefficients)
        return self._sigmoid(logits)


__all__ = ["LogitFitter"]
