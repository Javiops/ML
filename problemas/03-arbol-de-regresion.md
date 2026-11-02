# 03. Árbol de regresión

Es el mismo problema que el árbol de regresión estilo CF que ya tienes, con dos cambios para que dé exactamente lo mismo que `sklearn.tree.DecisionTreeRegressor`: los umbrales son **puntos medios** y no hace falta aritmética exacta.

## Notación

| Símbolo | Qué es | Tamaño |
|---|---|---|
| $n$ | número de puntos | número |
| $d$ | número de features | número |
| $X$ | matriz de datos: la fila $i$ es el punto $i$, la columna $j$ es la feature $j$ | $n \times d$ |
| $x_{ij}$ | valor de la feature $j$ en el punto $i$ | número |
| $y_i$ | objetivo del punto $i$ | número |
| $S$ | conjunto de puntos (índices $i$) que llegan a un nodo | conjunto |
| $\bar{y}_S$ | media de los $y_i$ con $i \in S$ | número |
| $\mathrm{SSE}(S)$ | $\sum_{i \in S} \left( y_i - \bar{y}_S \right)^2$ | número |
| $D$ | `max_depth` (`None` = sin límite) | número |
| $M$ | `min_samples_leaf` | número |

Convenio: $\mathrm{SSE}$ de un conjunto vacío es $0$.

## El algoritmo

Se construye recursivamente, empezando con todos los puntos en la raíz (profundidad $0$). Para un nodo con puntos $S$ y profundidad $h$:

1. Su **predicción** (`value`) es $\bar{y}_S$.
2. Si $D$ no es `None` y $h \geq D$, es una hoja.
3. Si no, se prueban todos los **cortes** $(j, t)$:
   - $j$ es una feature, $0 \leq j < d$.
   - Sean $v_1 < v_2 < \dots < v_m$ los valores **distintos** de la columna $j$ entre los puntos de $S$. Los umbrales candidatos son los puntos medios $t = \frac{v_r + v_{r+1}}{2}$, para $r = 1, \dots, m - 1$.
   - $S_L = \left\{ i \in S : x_{ij} \leq t \right\}$ y $S_R = S \setminus S_L$.
   - El corte es **válido** si $|S_L| \geq M$ y $|S_R| \geq M$.
   - Su **coste** es $\mathrm{SSE}(S_L) + \mathrm{SSE}(S_R)$.
4. Si no hay ningún corte válido, o si el de menor coste **no** es estrictamente menor que $\mathrm{SSE}(S)$, es una hoja.
5. Si no, el nodo guarda `feature` $= j$ y `threshold` $= t$ del mejor corte, y sus hijos son el nodo de $S_L$ (`left`) y el de $S_R$ (`right`), los dos con profundidad $h + 1$.

Para predecir un punto $z$ (vector de $d$ números): empiezas en la raíz, vas a `left` si $z_j \leq t$ y a `right` si no, hasta llegar a una hoja, y devuelves su `value`.

## Qué te toca a ti (`ml/tree.py`)

El esqueleto ya trae hecho: la clase `Node`, `fit`, `predict` y la lectura de datos. Tú escribes dos funciones.

**`best_split(X, y, min_samples_leaf)`**

- Entrada: `X` de tamaño $n \times d$ e `y` de tamaño $n$, que son solo los puntos de **un** nodo (los de $S$).
- Salida: la tupla `(j, t, coste)` del corte válido de menor coste, o `None` si no hay ninguno válido.
- Empates de coste: da igual cuál devuelvas.

**`_build(self, X, y, depth)`**

- Entrada: los puntos de un nodo y su profundidad.
- Salida: el `Node` raíz del subárbol, con `feature`, `threshold`, `left` y `right` rellenos si se parte, o como vienen (`None`) si es hoja.
- Tienes `self.max_depth` y `self.min_samples_leaf`.

Con fuerza bruta basta para pasar los tests. Si quieres que vaya rápido con $n$ grande, piensa cuánto cuesta tu `best_split` y si puedes bajarlo.

## NumPy que te hace falta

| Qué | NumPy |
|---|---|
| columna $j$ de `X` | `X[:, j]` |
| valores distintos de un array, ordenados | `np.unique(a)` |
| máscara de los puntos que van a la izquierda | `m = X[:, j] <= t` (array de `True`/`False`) |
| quedarte con esos puntos / con los otros | `X[m]`, `y[m]` / `X[~m]`, `y[~m]` |
| cuántos `True` hay | `np.sum(m)` |
| media de un array | `np.mean(a)` o `a.mean()` |
| $\mathrm{SSE}$ de un array `a` | `np.sum((a - a.mean()) ** 2)` (ojo: con `a` vacío, `mean` da aviso) |
| crear un nodo | `Node(valor)` |
| comparar floats con margen | `a < b - 1e-12` |

## Cómo saber si está bien

```bash
python -m pytest tests/test_tree.py -v
```

Los tests comprueban `best_split` en casos pequeños y contra fuerza bruta, el árbol en el ejemplo 2 del problema CF, y que tus predicciones coinciden con las de sklearn con varias combinaciones de `max_depth` y `min_samples_leaf`.

## Para pensar

1. Sin `max_depth` y con `min_samples_leaf = 1`, ¿cuánto vale el MSE en **train**? ¿Y en test? Pruébalo con `train_test_split`.
2. ¿Por qué un árbol nunca puede predecir un valor mayor que el máximo de los $y_i$ de train? Compáralo con la regresión lineal.
