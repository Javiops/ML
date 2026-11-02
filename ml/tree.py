"""Árboles de decisión.

Enunciado: problemas/03-arbol-de-regresion.md
"""

import numpy as np


class Node:
    """Un nodo del árbol.

    Si es hoja: feature es None y la predicción es value.
    Si no: los puntos con x[feature] <= threshold van a left, el resto a right.
    """

    def __init__(self, value):
        self.value = value        # media de y en este nodo
        self.feature = None       # j del corte (None si es hoja)
        self.threshold = None     # t del corte
        self.left = None          # Node
        self.right = None         # Node


def best_split(X, y, min_samples_leaf):
    """Mejor corte de un nodo.

    X: matriz (n, d), y: array de n.
    Devuelve (j, t, coste) del corte válido de menor coste, o None si no hay
    ningún corte válido. coste = SSE(izquierda) + SSE(derecha).
    """
    n, d = X.shape
    y = y - np.mean(y)            # no cambia ningún SSE y evita errores de redondeo
    tot1, tot2 = np.sum(y), np.sum(y ** 2)
    best = None
    for i in range(d):
        p = np.argsort(X[:, i])   # O(n log n)
        x_ord = X[p, i]
        y_ord = y[p]

        s1, s2 = 0.0, 0.0         # suma de y y de y^2 del prefijo
        for k in range(1, n):     # prefijo = y_ord[:k], sufijo = y_ord[k:], O(1) por k
            s1 += y_ord[k - 1]
            s2 += y_ord[k - 1] ** 2
            if k < min_samples_leaf or n - k < min_samples_leaf:
                continue
            if x_ord[k - 1] == x_ord[k]:
                continue
            pref = s2 - s1 ** 2 / k
            suf = (tot2 - s2) - (tot1 - s1) ** 2 / (n - k)

            if best is None or pref + suf < best[2]:
                t = (x_ord[k - 1] + x_ord[k]) / 2
                best = (i, t, pref + suf)

    return best


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth                # None = sin límite
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)   # matriz (n, d)
        y = np.asarray(y, dtype=float)   # array de n
        self.root_ = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        """Construye recursivamente el subárbol con los puntos (X, y) a profundidad depth.
        Devuelve su Node raíz."""
        # TODO (tú)
        raise NotImplementedError("_build")

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.empty(X.shape[0])
        for i, x in enumerate(X):
            node = self.root_
            while node.feature is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            pred[i] = node.value
        return pred
