import numpy as np


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)   # array de n
    y_pred = np.asarray(y_pred, dtype=float)   # array de n
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 1 - np.sum((y_true - y_pred) ** 2)/np.sum((y_true - np.mean(y_true)) ** 2)


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)   # sin float: pueden ser etiquetas como "a", "b"
    y_pred = np.asarray(y_pred)
    return np.mean(y_pred == y_true)
