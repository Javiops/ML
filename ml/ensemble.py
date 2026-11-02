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
