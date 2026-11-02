import numpy as np
import pytest

from ml import train_test_split


def datos(n, d=3):
    # La fila i de X empieza por i, e y[i] = 10 * i: así se puede comprobar
    # que X e y se barajan juntos.
    X = np.hstack([np.arange(n).reshape(-1, 1), np.ones((n, d - 1))])
    y = 10 * np.arange(n)
    return X, y


@pytest.mark.parametrize("n,test_size,n_test", [(100, 0.25, 25), (10, 0.3, 3), (7, 0.5, 4), (1000, 0.2, 200)])
def test_tamanos(pendiente, n, test_size, n_test):
    X, y = datos(n)
    X_tr, X_te, y_tr, y_te = pendiente(train_test_split, X, y, test_size=test_size, random_state=0)
    assert X_te.shape == (n_test, 3) and y_te.shape == (n_test,)
    assert X_tr.shape == (n - n_test, 3) and y_tr.shape == (n - n_test,)


def test_particion_y_filas_juntas(pendiente):
    X, y = datos(50)
    X_tr, X_te, y_tr, y_te = pendiente(train_test_split, X, y, test_size=0.2, random_state=1)
    ids = np.concatenate([X_tr[:, 0], X_te[:, 0]])
    assert sorted(ids) == list(range(50))              # cada fila exactamente una vez
    np.testing.assert_array_equal(y_tr, 10 * X_tr[:, 0])   # X e y siguen emparejados
    np.testing.assert_array_equal(y_te, 10 * X_te[:, 0])


def test_baraja(pendiente):
    X, y = datos(100)
    X_tr, X_te, _, _ = pendiente(train_test_split, X, y, random_state=2)
    assert not np.array_equal(X_te[:, 0], np.arange(75, 100))   # no coge simplemente las últimas


def test_semilla_reproducible(pendiente):
    X, y = datos(100)
    a = pendiente(train_test_split, X, y, random_state=42)
    b = pendiente(train_test_split, X, y, random_state=42)
    c = pendiente(train_test_split, X, y, random_state=43)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(a[1], c[1])
