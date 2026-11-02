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


def best_split(X, y, min_samples_leaf, features=None):
    """Mejor corte de un nodo.

    X: matriz (n, d), y: array de n.
    features: índices de las features que se prueban (None = todas).
    Devuelve (j, t, coste) del corte válido de menor coste, o None si no hay
    ningún corte válido. coste = SSE(izquierda) + SSE(derecha).
    """
    n, d = X.shape
    y = y - np.mean(y)            # no cambia ningún SSE y evita errores de redondeo
    tot1, tot2 = np.sum(y), np.sum(y ** 2)
    best = None
    if features is None:
        features = range(d)
    for i in features:
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


def n_features_por_nodo(max_features, d):
    """Cuántas features se prueban en cada nodo, con las mismas reglas que sklearn."""
    if max_features is None:
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    if isinstance(max_features, float):
        return max(1, int(max_features * d))
    return min(int(max_features), d)


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth                # None = sin límite
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features          # None = todas (ver n_features_por_nodo)
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)   # matriz (n, d)
        y = np.asarray(y, dtype=float)   # array de n
        self.rng_ = np.random.default_rng(self.random_state)
        self.m_ = n_features_por_nodo(self.max_features, X.shape[1])
        self.root_ = self._build(X, y, 0)
        self.leaves_ = []                # las hojas; la hoja h tiene leaf_id = h
        pila = [self.root_]
        while pila:
            node = pila.pop()
            if node.feature is None:
                node.leaf_id = len(self.leaves_)
                self.leaves_.append(node)
            else:
                pila += [node.right, node.left]
        return self

    def _features_al_azar(self, X):
        """Features que se prueban en este nodo: None (todas) o self.m_ elegidas al azar.
        Como sklearn, las que son constantes en el nodo no cuentan (no se pueden cortar)."""
        d = X.shape[1]
        if self.m_ >= d:
            return None
        elegidas = []
        for j in self.rng_.permutation(d):
            if X[:, j].min() < X[:, j].max():
                elegidas.append(j)
                if len(elegidas) == self.m_:
                    break
        return elegidas

    def _build(self, X, y, depth):
        """Construye recursivamente el subárbol con los puntos (X, y) a profundidad depth.
        Devuelve su Node raíz."""
        node = Node(np.mean(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if np.var(y) <= np.finfo(float).eps:     # nodo puro: todos los y iguales
            return node

        split = best_split(X, y, self.min_samples_leaf, self._features_al_azar(X))
        if split is None:                        # no hay corte válido
            return node

        # Como sklearn: se parte aunque el corte no baje el SSE (ganancia 0).
        j, t, coste = split
        node.feature, node.threshold = j, t
        m = X[:, j] <= t
        node.left = self._build(X[m], y[m], depth + 1)
        node.right = self._build(X[~m], y[~m], depth + 1)
        return node

    def _hoja(self, x):
        node = self.root_
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def apply(self, X):
        """Para cada punto, el índice h de la hoja a la que llega (su nodo es self.leaves_[h])."""
        X = np.asarray(X, dtype=float)
        return np.array([self._hoja(x).leaf_id for x in X], dtype=int)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.empty(X.shape[0])
        for i, x in enumerate(X):
            node = self.root_
            while node.feature is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            pred[i] = node.value
        return pred
