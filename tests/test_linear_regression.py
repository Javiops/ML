import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLinearRegression

from ml import LinearRegression


def datos(n, d, ruido, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d) * 3
    b = rng.normal() * 5
    y = X @ w + b + ruido * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("method", ["normal", "gd"])
def test_recta_exacta(pendiente, method):
    # y = 3x + 2 sin ruido: el modelo tiene que encontrarla exactamente.
    X = np.arange(10, dtype=float).reshape(-1, 1) / 10
    y = 3 * X[:, 0] + 2
    m = pendiente(LinearRegression(method=method).fit, X, y)
    assert m.coef_.shape == (1,)
    assert m.coef_[0] == pytest.approx(3, abs=1e-3)
    assert m.intercept_ == pytest.approx(2, abs=1e-3)


@pytest.mark.parametrize("n,d", [(50, 1), (200, 5), (1000, 20)])
def test_normal_igual_que_sklearn(pendiente, n, d):
    X, y = datos(n, d, ruido=0.5, seed=n + d)
    m = pendiente(LinearRegression(method="normal").fit, X, y)
    sk = SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(m.coef_, sk.coef_, atol=1e-8)
    assert m.intercept_ == pytest.approx(sk.intercept_, abs=1e-8)


@pytest.mark.parametrize("n,d", [(50, 1), (200, 5)])
def test_gd_converge_a_sklearn(pendiente, n, d):
    # Las features son N(0, 1), así que con lr=0.1 GD converge sin escalar.
    X, y = datos(n, d, ruido=0.5, seed=7 * n + d)
    m = pendiente(LinearRegression(method="gd", lr=0.1, n_iters=5000).fit, X, y)
    sk = SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(m.coef_, sk.coef_, atol=1e-4)
    assert m.intercept_ == pytest.approx(sk.intercept_, abs=1e-4)


def test_predict(pendiente):
    X, y = datos(100, 3, ruido=0.1, seed=42)
    m = pendiente(LinearRegression().fit, X, y)
    X_nuevo, _ = datos(20, 3, ruido=0.0, seed=43)
    pred = pendiente(m.predict, X_nuevo)
    assert pred.shape == (20,)
    np.testing.assert_allclose(pred, X_nuevo @ m.coef_ + m.intercept_, atol=1e-10)


def test_fit_devuelve_self(pendiente):
    X, y = datos(30, 2, ruido=0.1, seed=1)
    m = LinearRegression()
    assert pendiente(m.fit, X, y) is m


def test_columnas_duplicadas_no_revientan(pendiente):
    # X^T X es singular si dos columnas son iguales. Con np.linalg.inv esto
    # explota o da basura; hay que usar algo robusto (pista: lstsq / pinv).
    X, y = datos(100, 2, ruido=0.1, seed=3)
    X = np.hstack([X, X[:, :1]])
    m = pendiente(LinearRegression(method="normal").fit, X, y)
    sk = SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(m.predict(X), sk.predict(X), atol=1e-6)


def test_method_invalido():
    with pytest.raises(ValueError):
        LinearRegression(method="magia")
