# 01. Regresión lineal

*Dificultad estimada: ~1600 (la idea es fácil; el reto está en hacerlo bien
con NumPy y en entender por qué funciona).*

## Enunciado

Te dan `n` puntos. El punto `i` tiene `d` features `x_i = (x_i1, ..., x_id)` y
un valor objetivo `y_i`. Encuentra un vector `w` de `d` reales y un real `b`
que minimicen el error cuadrático medio:

```
MSE(w, b) = (1/n) * sum_i (w · x_i + b - y_i)^2
```

Después, para puntos nuevos, predice `w · x + b`.

Es exactamente lo que hace `sklearn.linear_model.LinearRegression`, y lo que
usarás en Kaggle para cualquier baseline de regresión (por ejemplo, House
Prices).

## Qué hay que implementar

`ml/linear_model.py`, clase `LinearRegression`:

- `fit(X, y)`: `X` tiene forma `(n, d)`, `y` forma `(n,)`. Guarda
  `self.coef_` (forma `(d,)`) y `self.intercept_` (un float). Devuelve `self`.
- `predict(X)`: devuelve un array de forma `(n,)`.
- Dos métodos, elegidos con `method`:
  - `"normal"`: solución cerrada.
  - `"gd"`: gradient descent con `lr` y `n_iters`.

Nada de bucles `for` sobre los puntos: todo vectorizado con NumPy. En `"gd"`
el bucle sobre las iteraciones sí está permitido.

## Pistas (léelas solo si te atascas)

<details>
<summary>Pista 1: quitarte el <code>b</code> de encima</summary>

Añade a `X` una columna de unos. Entonces `w · x + b` es un único producto
escalar con un vector de `d + 1` pesos, y el problema queda como
"minimizar `||Xw - y||^2`". El último peso es el `intercept_`.
</details>

<details>
<summary>Pista 2: la ecuación normal</summary>

`||Xw - y||^2` es una función convexa de `w` (un paraboloide). El mínimo
está donde el gradiente es cero:

```
grad = 2 X^T (Xw - y) = 0   =>   X^T X w = X^T y
```

Es un sistema lineal de `(d+1) × (d+1)`. Coste `O(n d^2 + d^3)`.

Ojo: si dos columnas son linealmente dependientes, `X^T X` es singular y
`np.linalg.inv` falla o devuelve basura. Hay un test que lo comprueba.
`np.linalg.lstsq` o `np.linalg.pinv` lo resuelven bien.
</details>

<details>
<summary>Pista 3: gradient descent</summary>

En lugar de resolver el sistema, empieza con `w = 0` y repite `n_iters` veces:

```
w -= lr * grad,   con grad = (2/n) X^T (Xw - y)
```

Analogía CF: es como una búsqueda ternaria, pero en `d` dimensiones y dando
pasos en la dirección de máxima bajada. Funciona porque la función es
convexa: no hay mínimos locales que no sean el global.

Coste `O(n_iters · n · d)`. Si `d` es enorme (miles de features), esto gana a
la ecuación normal, que tiene un `d^3`.

Si `lr` es demasiado grande, diverge (los pesos explotan a `inf`/`nan`).
Prueba con `lr = 10` para verlo.
</details>

## Cómo saber si está bien

```bash
pytest tests/test_linear_regression.py -v
```

Los tests comparan tus `coef_` e `intercept_` con los de sklearn sobre datos
aleatorios. Mientras tu código lance `NotImplementedError`, salen como
*skipped*.

## Para pensar después

1. ¿Por qué `"gd"` necesita que las features estén en escalas parecidas y
   `"normal"` no? Prueba a multiplicar una columna de `X` por 1000.
2. ¿Qué pasa si `n < d`? ¿Cuántas soluciones hay? ¿Cuál devuelve `lstsq`?
3. Sube un `LinearRegression` hecho con tu librería a House Prices de Kaggle
   y compara el score con el de sklearn. Deberían ser idénticos.
