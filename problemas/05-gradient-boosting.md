# 05. Gradient Boosting

Random Forest entrena árboles **independientes** y hace la media. Boosting los entrena **uno detrás de otro**: cada árbol nuevo intenta arreglar lo que los anteriores hacen mal. Se suelen usar muchos árboles poco profundos (`max_depth=3`).

## Notación

| Símbolo | Qué es | Tamaño |
|---|---|---|
| $n$ | número de puntos | número |
| $X$ | matriz de datos: la fila $i$ es el punto $x_i$ | $n \times d$ |
| $y_i$ | objetivo del punto $i$ | número |
| $L(y, F)$ | **pérdida**: lo que cuesta predecir $F$ cuando la verdad es $y$ | número |
| $K$ | número de árboles (`n_estimators`) | número |
| $\eta$ | `learning_rate`, con $0 < \eta \leq 1$ | número |
| $F_k(x)$ | predicción del modelo después de $k$ árboles | número |
| $h_k(x)$ | predicción del árbol $k$ | número |
| $R$ | una hoja de un árbol, vista como el conjunto de puntos de train que caen en ella | conjunto |

## Las dos pérdidas

- `loss="squared_error"`:

$$L(y, F) = (y - F)^2$$

- `loss="absolute_error"`:

$$L(y, F) = \left| y - F \right|$$

En el punto donde no es derivable ($F = y$), toma la derivada como $0$.

## El algoritmo

1. Empieza con la mejor **constante**:

$$F_0 = \arg\min_{c} \sum_{i=1}^{n} L(y_i, c)$$

2. Para $k = 1, \dots, K$:

   a. Calcula para cada punto el **pseudo-residuo**, menos la derivada de la pérdida respecto a la predicción, en la predicción actual:

$$r_i = - \left. \frac{\partial L(y_i, F)}{\partial F} \right|_{F = F_{k-1}(x_i)}$$

   b. Entrena un árbol de regresión del problema 03 (con `max_depth` y `min_samples_leaf`) con $X$ y objetivo $r$.

   c. Para cada hoja $R$ de ese árbol, cambia su valor por

$$\gamma_R = \arg\min_{\gamma} \sum_{i \in R} L\left( y_i, F_{k-1}(x_i) + \gamma \right)$$

   El árbol con esos valores es $h_k$.

   d. Actualiza: $F_k(x) = F_{k-1}(x) + \eta\ h_k(x)$.

3. La predicción es $F_K(x)$.

Si un $\arg\min$ tiene varios valores óptimos, vale cualquiera.

## La idea

$F_{k-1}(x_i)$ es lo que predice el modelo para el punto $i$ antes de añadir el árbol $k$.

- El **pseudo-residuo** $r_i$ dice hacia dónde mover esa predicción para que la pérdida del punto $i$ baje, y cuánto. Si $r_i > 0$ hay que subirla y si $r_i < 0$ hay que bajarla. Lleva un signo menos porque la derivada apunta hacia donde la pérdida **sube**. Es el gradient descent del módulo 1, pero moviendo las predicciones en vez de los pesos.
- Las predicciones de train no se pueden mover una a una, porque eso no dice nada de un punto nuevo. Por eso se entrena un árbol que aprende $r$ en función de $x$: así la corrección vale para cualquier $x$.
- El paso c decide cuánto moverse en cada hoja, y $\eta$ hace que el paso sea pequeño, como el learning rate.

## Ejemplo de una vuelta

Con otra pérdida, $L(y, F) = (y - F)^4$, para no destriparte las tuyas. Hay 4 puntos con una feature, $\eta = 0.5$ y `max_depth=1`.

| $i$ | $x_i$ | $y_i$ | $F_0(x_i)$ | $r_i = 4 (y_i - F_0(x_i))^3$ | hoja | $\gamma$ | $F_1(x_i)$ |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 5 | $-256$ | izq | $-3.5$ | 3.25 |
| 2 | 2 | 2 | 5 | $-108$ | izq | $-3.5$ | 3.25 |
| 3 | 3 | 8 | 5 | $108$ | der | $3.5$ | 6.75 |
| 4 | 4 | 9 | 5 | $256$ | der | $3.5$ | 6.75 |

- **$F_0$:** la derivada de $\sum_i (y_i - c)^4$ es $-4 \sum_i (y_i - c)^3$, que en $c = 5$ vale $-4 \left( (-4)^3 + (-3)^3 + 3^3 + 4^3 \right) = 0$. Como es convexa, $F_0 = 5$.
- **a.** $\partial L / \partial F = -4 (y - F)^3$, así que $r_i = 4 (y_i - F_0(x_i))^3$. Los dos primeros puntos piden bajar y los dos últimos subir.
- **b.** El árbol entrenado con objetivo $r$ corta en $x \leq 2.5$. Sus hojas valdrían la media de $r$, que es $\pm 182$.
- **c.** En la hoja izquierda los residuos $y_i - F_0(x_i)$ son $-4$ y $-3$, y $\gamma = \arg\min_\gamma \left( (-4 - \gamma)^4 + (-3 - \gamma)^4 \right) = -3.5$ por simetría. En la derecha, $3.5$. Las hojas pasan a valer $\pm 3.5$, no $\pm 182$.
- **d.** $F_1(x) = 5 + 0.5\ h_1(x)$. Por ejemplo, un punto nuevo con $x = 10$ predice $6.75$.

## Lo que tienes que sacar

Para **cada una** de las dos pérdidas: quién es $F_0$, quién es $r_i$ y quién es $\gamma_R$. Con papel, antes de programar.

## Qué te toca a ti (`ml/ensemble.py`)

`GradientBoostingRegressor.fit(X, y)`. Tiene que dejar `self.init_` $= F_0$ y `self.estimators_` = la lista de los $K$ árboles con los valores de las hojas ya cambiados (paso c). `predict` ya está hecho.

## Python que te hace falta

| Qué | Python |
|---|---|
| entrenar un árbol con objetivo `r` | `tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(X, r)` |
| hoja a la que cae cada punto | `hojas = tree.apply(X)` (array de $n$ enteros) |
| recorrer las hojas | `for h, hoja in enumerate(tree.leaves_):` |
| puntos de la hoja `h` | máscara `hojas == h`, y luego `a[hojas == h]` |
| cambiar el valor de una hoja | `hoja.value = ...` |
| predicción del árbol en todos los puntos | `tree.predict(X)` |
| array de $n$ copias de un número | `np.full(n, valor)` |
| añadir a una lista | `lista.append(tree)` |

## Cómo saber si está bien

```bash
python -m pytest tests/test_boosting.py -v
```

- `squared_error`: $F_0$, y que las predicciones coinciden con las de `sklearn.ensemble.GradientBoostingRegressor` en 6 combinaciones de parámetros.
- `absolute_error`: que $F_0$ es óptimo (con $n$ par e impar), que el valor de cada hoja es un $\gamma_R$ óptimo, que el error de train nunca sube de un árbol al siguiente, y que el error en test está a menos de un 10 % del de sklearn.

## Para pensar

1. ¿Por qué con `absolute_error` el error de train nunca sube?
2. Mete un 5 % de outliers en $y$ (súmales 100). ¿Qué pérdida aguanta mejor en test? ¿Por qué?
3. Random Forest baja la varianza. ¿Qué baja el boosting? ¿Por qué aquí se usan árboles poco profundos y en Random Forest no?
