"""Partición de datos.

Enunciado: problemas/02-metricas-y-split.md
"""

import numpy as np


def train_test_split(X, y, test_size=0.25, random_state=None):
    """Devuelve X_train, X_test, y_train, y_test."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.default_rng(random_state)   # generador con semilla
    # TODO
    raise NotImplementedError("train_test_split")
