import numpy as np
import pytest
from sklearn import metrics as skm

from ml import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def datos(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n) * 10
    pred = y + rng.normal(size=n) * 3
    return y, pred


@pytest.mark.parametrize("n,seed", [(1, 0), (10, 1), (1000, 2)])
def test_mae_mse_igual_que_sklearn(pendiente, n, seed):
    y, pred = datos(n, seed)
    assert pendiente(mean_absolute_error, y, pred) == pytest.approx(skm.mean_absolute_error(y, pred))
    assert pendiente(mean_squared_error, y, pred) == pytest.approx(skm.mean_squared_error(y, pred))


def test_mae_mse_a_mano(pendiente):
    y = [1, 2, 3]
    pred = [2, 2, 5]   # errores: 1, 0, 2
    assert pendiente(mean_absolute_error, y, pred) == pytest.approx(1.0)
    assert pendiente(mean_squared_error, y, pred) == pytest.approx(5 / 3)


@pytest.mark.parametrize("n,seed", [(10, 3), (1000, 4)])
def test_r2_igual_que_sklearn(pendiente, n, seed):
    y, pred = datos(n, seed)
    assert pendiente(r2_score, y, pred) == pytest.approx(skm.r2_score(y, pred))


def test_r2_casos_clave(pendiente):
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert pendiente(r2_score, y, y) == pytest.approx(1.0)                         # perfecto
    assert pendiente(r2_score, y, np.full(4, y.mean())) == pytest.approx(0.0)      # predecir la media
    assert pendiente(r2_score, y, -y) < 0                                          # peor que la media


def test_accuracy(pendiente):
    assert pendiente(accuracy_score, [0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)
    assert pendiente(accuracy_score, ["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    assert pendiente(accuracy_score, y, p) == pytest.approx(skm.accuracy_score(y, p))
