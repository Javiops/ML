"""Modelos lineales.

Enunciado: problemas/01-regresion-lineal.md
Demostración: problemas/01-regresion-lineal-demostracion.md
"""

import numpy as np


class LinearRegression:
    def __init__(self, method="normal", lr=0.1, n_iters=5000):
        if method not in ("normal", "gd"):
            raise ValueError(f"method debe ser 'normal' o 'gd', no {method!r}")
        self.method = method
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)   # matriz (n, d)
        y = np.asarray(y, dtype=float)   # vector (n,)
        n, d = X.shape

        if self.method == "normal":
            mx = X.mean(axis=0)   # E[x]: array de d
            my = y.mean()         # E[y]: un número
    
            S = np.zeros((d, d))   # Sigma
            c = np.zeros(d)
    
            for j in range(d):
                for k in range(d):
                    # Cov(x_j, x_k) = E[x_j x_k] - E[x_j] E[x_k]
                    S[j, k] = np.mean(X[:, j] * X[:, k]) - mx[j] * mx[k]
                # Cov(x_j, y) = E[x_j y] - E[x_j] E[y]
                c[j] = np.mean(X[:, j] * y) - mx[j] * my
    
            w = np.linalg.lstsq(S, c, rcond=None)[0]
    
            b = my - (mx @ w)
    
            self.coef_ = w          # array de d
            self.intercept_ = b     # número
            return self
        if self.method == "gd":
            w = np.zeros(d)
            b = 0.0

            for _ in range(self.n_iters):
                r = (X @ w) + b - y          # error de cada punto: array de n
                bp = 2 * np.mean(r)          # dMSE/db: un número
                g = 2 / n * (X.T @ r)        # dMSE/dw_j para cada j: array de d

                b -= self.lr * bp
                w -= self.lr * g

            self.coef_ = w
            self.intercept_ = b
            return self
            

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        return (X @ self.coef_) + self.intercept_


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression as SkLR

    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 3 + 0.1 * rng.normal(size=n)

    mio = LinearRegression().fit(X, y)
    sk = SkLR().fit(X, y)
    print("w mio:", mio.coef_)
    print("w sk: ", sk.coef_)
    print("b mio:", mio.intercept_, " b sk:", sk.intercept_)
    print("OK" if np.allclose(mio.coef_, sk.coef_) and np.isclose(mio.intercept_, sk.intercept_) else "WA")
