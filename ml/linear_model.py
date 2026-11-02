"""Modelos lineales.

Enunciado: problemas/01-regresion-lineal.md
"""

import numpy as np


class LinearRegression:
    """Regresión lineal por mínimos cuadrados.

    Parámetros
    ----------
    method : "normal" o "gd"
        "normal": resuelve la ecuación normal (solución cerrada).
        "gd": gradient descent sobre el MSE.
    lr : float
        Learning rate, solo para method="gd".
    n_iters : int
        Número de iteraciones, solo para method="gd".

    Atributos tras fit
    ------------------
    coef_ : ndarray de forma (d,)
    intercept_ : float
    """

    def __init__(self, method="normal", lr=0.1, n_iters=5000):
        if method not in ("normal", "gd"):
            raise ValueError(f"method debe ser 'normal' o 'gd', no {method!r}")
        self.method = method
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        """Ajusta el modelo. X: (n, d), y: (n,). Devuelve self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # TODO: rellenar self.coef_ y self.intercept_
        raise NotImplementedError("LinearRegression.fit")

    def predict(self, X):
        """Devuelve las predicciones, un ndarray de forma (n,)."""
        X = np.asarray(X, dtype=float)
        # TODO
        raise NotImplementedError("LinearRegression.predict")
