import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor as SkTree

from ml.tree import DecisionTreeRegressor, best_split


def sse(a):
    a = np.asarray(a, dtype=float)
    return float(np.sum((a - a.mean()) ** 2)) if len(a) else 0.0


# ---------- best_split ----------

def test_best_split_dos_grupos(pendiente):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 1.0, 5.0, 5.0])
    j, t, coste = pendiente(best_split, X, y, 1)
    assert (j, t) == (0, 2.5)
    assert coste == pytest.approx(0.0)


def test_best_split_elige_la_feature_buena(pendiente):
    # La feature 0 es ruido; la 1 separa y perfectamente.
    X = np.array([[5.0, 1.0], [1.0, 2.0], [4.0, 10.0], [2.0, 11.0]])
    y = np.array([0.0, 0.0, 7.0, 7.0])
    j, t, coste = pendiente(best_split, X, y, 1)
    assert (j, t) == (1, 6.0)
    assert coste == pytest.approx(0.0)


def test_best_split_min_samples_leaf(pendiente):
    # Ejemplo 2 del problema del árbol de regresión.
    X = np.arange(1, 6, dtype=float).reshape(-1, 1)
    y = np.array([1.0, 2.0, 100.0, 101.0, 3.0])
    j, t, coste = pendiente(best_split, X, y, 2)
    assert (j, t) == (0, 2.5)
    assert coste == pytest.approx(sse([1, 2]) + sse([100, 101, 3]))


def test_best_split_sin_corte_valido(pendiente):
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    assert pendiente(best_split, X, y, 2) is None          # no caben 2 y 2
    X = np.array([[7.0], [7.0], [7.0], [7.0]])
    assert pendiente(best_split, X, np.arange(4.0), 1) is None   # todos iguales


def test_best_split_fuerza_bruta(pendiente):
    rng = np.random.default_rng(0)
    for _ in range(30):
        n, d, M = rng.integers(2, 25), rng.integers(1, 4), rng.integers(1, 4)
        X = rng.normal(size=(n, d)).round(1)   # con repetidos
        y = rng.normal(size=n)
        mejor = None
        for j in range(d):
            vals = np.unique(X[:, j])
            for a, b in zip(vals[:-1], vals[1:]):
                t = (a + b) / 2
                L, R = y[X[:, j] <= t], y[X[:, j] > t]
                if len(L) >= M and len(R) >= M:
                    c = sse(L) + sse(R)
                    if mejor is None or c < mejor[2] - 1e-12:
                        mejor = (j, t, c)
        res = pendiente(best_split, X, y, M)
        if mejor is None:
            assert res is None
        else:
            assert res is not None
            assert res[2] == pytest.approx(mejor[2], abs=1e-9)


# ---------- árbol completo ----------

def test_profundidad_0_es_la_media(pendiente):
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0) ** 2
    m = pendiente(DecisionTreeRegressor(max_depth=0).fit, X, y)
    np.testing.assert_allclose(m.predict([[0.0], [100.0]]), [y.mean(), y.mean()])


def test_ejemplo_2_del_problema(pendiente):
    X = np.arange(1, 6, dtype=float).reshape(-1, 1)
    y = np.array([1.0, 2.0, 100.0, 101.0, 3.0])
    m = pendiente(DecisionTreeRegressor(max_depth=3, min_samples_leaf=2).fit, X, y)
    np.testing.assert_allclose(m.predict([[0.0], [3.0], [10.0]]), [1.5, 68.0, 68.0])


def test_y_constante_es_una_hoja(pendiente):
    X = np.random.default_rng(1).normal(size=(20, 3))
    m = pendiente(DecisionTreeRegressor().fit, X, np.full(20, 4.0))
    assert m.root_.feature is None and m.root_.value == pytest.approx(4.0)


@pytest.mark.parametrize("max_depth,min_samples_leaf", [(1, 1), (3, 1), (3, 5), (6, 2), (None, 10)])
def test_igual_que_sklearn(pendiente, max_depth, min_samples_leaf):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 3))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=300)
    X_new = rng.normal(size=(200, 3))

    mio = pendiente(DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf).fit, X, y)
    sk = SkTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0).fit(X, y)
    np.testing.assert_allclose(mio.predict(X_new), sk.predict(X_new), atol=1e-9)
    np.testing.assert_allclose(mio.predict(X), sk.predict(X), atol=1e-9)


def test_sin_limite_memoriza_train(pendiente):
    # Sin max_depth ni min_samples_leaf, el árbol parte hasta que cada hoja es pura:
    # acierta todos los puntos de train. (Con puntos nuevos no comparamos con sklearn
    # porque en las hojas pequeñas hay empates entre cortes y cada uno elige uno distinto.)
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    m = pendiente(DecisionTreeRegressor().fit, X, y)
    np.testing.assert_allclose(m.predict(X), y, atol=1e-9)


def test_xor_parte_con_ganancia_0(pendiente):
    # Ningún corte de la raíz baja el SSE (media 0.5 a cada lado), pero como sklearn
    # partimos igualmente y en el nivel siguiente se aciertan los 4 puntos.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    m = pendiente(DecisionTreeRegressor().fit, X, y)
    np.testing.assert_allclose(m.predict(X), y)
    np.testing.assert_allclose(m.predict(X), SkTree(random_state=0).fit(X, y).predict(X))
