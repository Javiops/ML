import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor as SkGB

from ml.ensemble import GradientBoostingRegressor
from ml.metrics import mean_absolute_error


def datos(rng, n, d=4):
    X = rng.normal(size=(n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    return X, y


@pytest.fixture
def train_test():
    rng = np.random.default_rng(0)
    return datos(rng, 300), datos(rng, 500)


def suma_abs_minima(a):
    # min_c sum |a_i - c|: la función es convexa y lineal a trozos, así que el
    # mínimo se alcanza en uno de los a_i.
    return min(np.sum(np.abs(a - c)) for c in a)


# ---------- squared_error ----------

def test_squared_init(pendiente, train_test):
    (X, y), _ = train_test
    m = pendiente(GradientBoostingRegressor(n_estimators=1).fit, X, y)
    assert m.init_ == pytest.approx(np.mean(y))


# Con min_samples_leaf=1 y profundidad >= 2 salen nodos pequeños con cortes empatados
# (el mismo punto se puede aislar por dos features) y sklearn los desempata al azar.
# Por eso esas combinaciones no están.
@pytest.mark.parametrize("n_estimators, learning_rate, max_depth, min_samples_leaf", [
    (1, 1.0, 1, 1), (30, 1.0, 1, 1), (50, 0.1, 3, 5), (100, 0.1, 3, 10), (20, 0.5, 2, 5),
    (10, 0.3, 4, 10),
])
def test_squared_igual_que_sklearn(pendiente, train_test, n_estimators, learning_rate,
                                   max_depth, min_samples_leaf):
    (X, y), (Xt, _) = train_test
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    m = pendiente(GradientBoostingRegressor(**params).fit, X, y)
    sk = SkGB(random_state=0, **params).fit(X, y)
    np.testing.assert_allclose(m.predict(Xt), sk.predict(Xt), atol=1e-9)
    np.testing.assert_allclose(m.predict(X), sk.predict(X), atol=1e-9)


# ---------- absolute_error ----------

def test_absolute_init_optimo(pendiente, train_test):
    (X, y), _ = train_test
    for n in (300, 299):   # par e impar
        m = pendiente(GradientBoostingRegressor(n_estimators=1, loss="absolute_error").fit,
                      X[:n], y[:n])
        assert np.sum(np.abs(y[:n] - m.init_)) == pytest.approx(suma_abs_minima(y[:n]))


def test_absolute_hojas_optimas(pendiente, train_test):
    # En cada etapa, el valor de cada hoja tiene que ser un gamma óptimo para esa hoja.
    (X, y), _ = train_test
    m = pendiente(GradientBoostingRegressor(n_estimators=10, learning_rate=0.5,
                                            loss="absolute_error").fit, X, y)
    assert len(m.estimators_) == 10
    F = np.full(len(y), m.init_)
    for tree in m.estimators_:
        hojas = tree.apply(X)
        for h, hoja in enumerate(tree.leaves_):
            res = (y - F)[hojas == h]
            assert np.sum(np.abs(res - hoja.value)) == pytest.approx(suma_abs_minima(res))
        F += m.learning_rate * tree.predict(X)


def test_absolute_error_de_train_no_sube(pendiente, train_test):
    (X, y), _ = train_test
    m = pendiente(GradientBoostingRegressor(n_estimators=30, loss="absolute_error").fit, X, y)
    F = np.full(len(y), m.init_)
    errores = [mean_absolute_error(y, F)]
    for tree in m.estimators_:
        F += m.learning_rate * tree.predict(X)
        errores.append(mean_absolute_error(y, F))
    assert all(b <= a + 1e-12 for a, b in zip(errores, errores[1:]))
    assert errores[-1] < 0.8 * errores[0]


def test_absolute_parecido_a_sklearn(pendiente, train_test):
    # Con pseudo-residuos que valen -1, 0 o 1 hay muchos cortes empatados y sklearn
    # los desempata al azar, así que aquí comparamos el error en test.
    (X, y), (Xt, yt) = train_test
    m = pendiente(GradientBoostingRegressor(n_estimators=100, loss="absolute_error").fit, X, y)
    sk = SkGB(n_estimators=100, loss="absolute_error", random_state=0).fit(X, y)
    e1 = mean_absolute_error(yt, m.predict(Xt))
    e2 = mean_absolute_error(yt, sk.predict(Xt))
    assert 0.9 < e1 / e2 < 1.1
