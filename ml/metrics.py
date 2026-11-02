"""Métricas.

Enunciado: problemas/02-metricas-y-split.md
"""

import numpy as np


def mean_absolute_error(y_true, y_pred):
    """MAE. Devuelve un número."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # TODO
    raise NotImplementedError("mean_absolute_error")


def mean_squared_error(y_true, y_pred):
    """MSE. Devuelve un número."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # TODO
    raise NotImplementedError("mean_squared_error")


def r2_score(y_true, y_pred):
    """Coeficiente R^2. Devuelve un número."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # TODO
    raise NotImplementedError("r2_score")


def accuracy_score(y_true, y_pred):
    """Fracción de aciertos. Devuelve un número entre 0 y 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # TODO
    raise NotImplementedError("accuracy_score")
