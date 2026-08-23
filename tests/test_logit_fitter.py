from unittest.mock import patch

import numpy as np
import pytest

from infcontrol import LogitFitter


def test_logit_fitter_initializes_strengths_and_accepts_feature() -> None:
    targets = np.array([0, 1] * 10)
    feature = np.array([-2.0, 2.0] * 10)
    fitter = LogitFitter(
        targets,
        groups=[f"goal-{index}" for index in range(20)],
        validation_fraction=0.5,
        test_fraction=0,
    )

    assert fitter.prior_probability == pytest.approx(
        np.mean(targets[fitter.train_indices])
    )
    np.testing.assert_allclose(fitter.strengths, fitter.initial_strength)
    coefficient = fitter.evaluate(feature)

    assert coefficient is not None
    assert coefficient > 0
    assert len(fitter.coefficients) == 1
    np.testing.assert_allclose(
        fitter.strengths, fitter.initial_strength + coefficient * feature
    )
    assert fitter.log_loss() < np.log(2.0)
    assert np.isfinite(coefficient)
    assert np.all((fitter.probabilities >= 0) & (fitter.probabilities <= 1))


def test_logit_fitter_rejects_feature_without_improvement() -> None:
    fitter = LogitFitter(
        [0, 1] * 10,
        groups=[f"goal-{index}" for index in range(20)],
        validation_fraction=0.5,
        test_fraction=0,
    )
    fitter.evaluate([-2.0, 2.0] * 10)
    strengths = fitter.strengths.copy()
    coefficients = fitter.coefficients.copy()

    assert fitter.evaluate(np.zeros(20)) is None
    np.testing.assert_array_equal(fitter.strengths, strengths)
    assert fitter.coefficients == coefficients
    assert len(fitter.features) == 1


def test_logit_fitter_refits_all_coefficients_after_acceptance() -> None:
    fitter = LogitFitter(
        [0, 1] * 10,
        groups=[f"goal-{index}" for index in range(20)],
        validation_fraction=0.5,
        test_fraction=0,
    )

    with patch.object(
        fitter, "_refit_coefficients", wraps=fitter._refit_coefficients
    ) as refit:
        coefficient = fitter.evaluate([-2.0, 2.0] * 10)

    assert coefficient is not None
    refit.assert_called_once()


def test_logit_fitter_acceptance_uses_validation_loss() -> None:
    fitter = LogitFitter(
        [0, 1] * 10,
        groups=[f"goal-{index}" for index in range(20)],
        validation_fraction=0.5,
        test_fraction=0,
    )
    strengths = fitter.strengths.copy()

    feature = np.zeros(20)
    feature[fitter.train_indices] = np.where(
        fitter.y[fitter.train_indices] == 1, 1.0, -1.0
    )
    feature[fitter.validation_indices] = np.where(
        fitter.y[fitter.validation_indices] == 1, -1.0, 1.0
    )
    coefficient = fitter.evaluate(feature)

    assert coefficient is None
    np.testing.assert_array_equal(fitter.strengths, strengths)
    assert fitter.validation_log_loss == pytest.approx(np.log(2.0))


def test_logit_fitter_validates_data_shapes_and_labels() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        LogitFitter([0, 2], groups=["a", "b"], validation_fraction=0.5, test_fraction=0)
    with pytest.raises(ValueError, match="shape"):
        LogitFitter(
            [0, 1], groups=["a", "b"], validation_fraction=0.5, test_fraction=0
        ).evaluate([1.0])
