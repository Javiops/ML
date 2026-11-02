"""Ensembles de árboles.

Enunciado: problemas/04-random-forest.md
"""

import numpy as np

from ml.tree import DecisionTreeRegressor


class RandomForestRegressor:
    """Media de n_estimators árboles. Cada árbol se entrena con un bootstrap de los
    puntos (n puntos sacados con reemplazo) y en cada nodo solo prueba max_features
    features elegidas al azar."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 max_features=1.0, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features          # 1.0 = todas, como en sklearn
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng,
            )
            self.estimators_.append(tree.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([tree.predict(X) for tree in self.estimators_], axis=0)


class GradientBoostingRegressor:
    """F(x) = init_ + learning_rate * (h_1(x) + ... + h_K(x)), con K = n_estimators.

    Enunciado: problemas/05-gradient-boosting.md
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, loss="squared_error"):
        if loss not in ("squared_error", "absolute_error"):
            raise ValueError("loss tiene que ser 'squared_error' o 'absolute_error'")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate       # eta
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)   # matriz (n, d)
        y = np.asarray(y, dtype=float)   # array de n
        # Tienes que dejar:
        #   self.init_       = F_0 (un número)
        #   self.estimators_ = lista con los K árboles h_1, ..., h_K, con el valor
        #                      de cada hoja ya cambiado por gamma (paso 3)
        # TODO (tú)
        raise NotImplementedError("GradientBoostingRegressor.fit")

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F
