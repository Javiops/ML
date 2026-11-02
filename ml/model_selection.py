import numpy as np


def train_test_split(X, y, test_size=0.25, random_state=None):
    X = np.asarray(X)   # matriz (n, d)
    y = np.asarray(y)   # array de n
    n = X.shape[0]
    rng = np.random.default_rng(random_state)   # generador con semilla, como mt19937 rng(seed)
    perm = rng.permutation(n)

    n_test = int(np.ceil(n * test_size))
    test, train = perm[:n_test], perm[n_test:]

    return X[train], X[test], y[train], y[test]
