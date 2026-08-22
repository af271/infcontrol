import numpy as np
import pytest

from infcontrol import LogitFitter


def test_logit_fitter_initializes_strengths_and_accepts_feature() -> None:
    targets = np.array([0, 0, 1, 1])
    feature = np.array([-2.0, -1.0, 1.0, 2.0])
    fitter = LogitFitter(targets)

    np.testing.assert_array_equal(fitter.strengths, np.zeros(4))
    coefficient = fitter.evaluate(feature)

    assert coefficient is not None
    assert coefficient > 0
    assert len(fitter.coefficients) == 1
    np.testing.assert_allclose(fitter.strengths, coefficient * feature)
    assert fitter.log_loss < np.log(2.0)
    assert np.isfinite(coefficient)
    assert np.all((fitter.probabilities >= 0) & (fitter.probabilities <= 1))


def test_logit_fitter_rejects_feature_without_improvement() -> None:
    fitter = LogitFitter([0, 0, 1, 1])
    fitter.evaluate([-2.0, -1.0, 1.0, 2.0])
    strengths = fitter.strengths.copy()
    coefficients = fitter.coefficients.copy()

    assert fitter.evaluate([0.0, 0.0, 0.0, 0.0]) is None
    np.testing.assert_array_equal(fitter.strengths, strengths)
    assert fitter.coefficients == coefficients
    assert len(fitter.features) == 1


def test_logit_fitter_validates_data_shapes_and_labels() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        LogitFitter([0, 2])
    with pytest.raises(ValueError, match="shape"):
        LogitFitter([0, 1]).evaluate([1.0])
