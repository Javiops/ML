import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor as SkForest

from ml.ensemble import RandomForestRegressor
from ml.metrics import mean_squared_error
from ml.tree import DecisionTreeRegressor, n_features_por_nodo


def datos(rng, n, d=5):
    X = rng.normal(size=(n, d))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.5 * rng.normal(size=n)
    return X, y


@pytest.fixture
def train_test():
    rng = np.random.default_rng(0)
    return datos(rng, 300), datos(rng, 1000)


# ---------- max_features en el árbol ----------

@pytest.mark.parametrize("mf, d, esperado", [
    (None, 7, 7), (1.0, 7, 7), (0.5, 7, 3), (0.01, 7, 1),
    ("sqrt", 10, 3), ("log2", 10, 3), ("sqrt", 1, 1), (2, 7, 2), (20, 7, 7),
])
def test_n_features_por_nodo(mf, d, esperado):
    assert n_features_por_nodo(mf, d) == esperado


def test_max_features_cambia_la_raiz(train_test):
    (X, y), _ = train_test
    raices = {DecisionTreeRegressor(max_features=1, random_state=s).fit(X, y).root_.feature
              for s in range(20)}
    assert len(raices) > 1


def test_features_constantes_no_cuentan():
    # Las columnas 0 y 1 son constantes: con max_features=1 siempre se elige la 2,
    # así que el árbol es el mismo que probando todas.
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(50), np.zeros(50), rng.normal(size=50)])
    y = rng.normal(size=50)
    completo = DecisionTreeRegressor().fit(X, y).predict(X)
    for s in range(5):
        m = DecisionTreeRegressor(max_features=1, random_state=s).fit(X, y)
        assert m.root_.feature == 2
        np.testing.assert_allclose(m.predict(X), completo)


# ---------- el bosque ----------

def test_un_arbol_sin_bootstrap_es_el_arbol(train_test):
    (X, y), (Xt, _) = train_test
    rf = RandomForestRegressor(n_estimators=1, bootstrap=False, max_features=None).fit(X, y)
    np.testing.assert_allclose(rf.predict(Xt), DecisionTreeRegressor().fit(X, y).predict(Xt))


def test_predice_la_media_de_los_arboles(train_test):
    (X, y), (Xt, _) = train_test
    rf = RandomForestRegressor(n_estimators=7, max_features="sqrt", random_state=0).fit(X, y)
    assert len(rf.estimators_) == 7
    media = np.mean([t.predict(Xt) for t in rf.estimators_], axis=0)
    np.testing.assert_allclose(rf.predict(Xt), media)


def test_random_state(train_test):
    (X, y), (Xt, _) = train_test
    a = RandomForestRegressor(n_estimators=5, random_state=3).fit(X, y).predict(Xt)
    b = RandomForestRegressor(n_estimators=5, random_state=3).fit(X, y).predict(Xt)
    c = RandomForestRegressor(n_estimators=5, random_state=4).fit(X, y).predict(Xt)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_bosque_mejor_que_un_arbol(train_test):
    (X, y), (Xt, yt) = train_test
    arbol = mean_squared_error(yt, DecisionTreeRegressor().fit(X, y).predict(Xt))
    bosque = mean_squared_error(yt, RandomForestRegressor(n_estimators=50, random_state=0)
                                .fit(X, y).predict(Xt))
    assert bosque < 0.8 * arbol


@pytest.mark.parametrize("max_features", [1.0, "sqrt"])
def test_parecido_a_sklearn(train_test, max_features):
    # El azar no es el mismo que el de sklearn, así que no comparamos predicciones
    # punto a punto sino el error en test.
    (X, y), (Xt, yt) = train_test
    nuestro = RandomForestRegressor(n_estimators=50, max_features=max_features, random_state=0)
    suyo = SkForest(n_estimators=50, max_features=max_features, random_state=0)
    e1 = mean_squared_error(yt, nuestro.fit(X, y).predict(Xt))
    e2 = mean_squared_error(yt, suyo.fit(X, y).predict(Xt))
    assert 0.85 < e1 / e2 < 1.15
