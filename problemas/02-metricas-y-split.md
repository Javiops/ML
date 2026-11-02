# 02. Métricas y train/test split

## Por qué

Hasta ahora medíamos el modelo con los mismos datos con los que lo entrenamos. Eso es como probar tu solución solo con los samples del enunciado: puede pasarlos y dar WA en los tests ocultos. En Kaggle pasa lo mismo: lo que importa es el error sobre datos que el modelo **no ha visto**.

Así que necesitamos dos cosas:

1. **Métricas**: un número que diga lo bueno que es un modelo.
2. **`train_test_split`**: separar los datos en una parte para entrenar y otra, escondida, para medir.

## Notación

$y$ son los valores reales e $\hat{y}$ las predicciones, los dos arrays de $n$ elementos. $\bar{y}$ es la media de $y$.

## Parte 1: métricas (`ml/metrics.py`)

**Error absoluto medio:**

$$\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

**Error cuadrático medio** (el que minimizamos en el problema 01):

$$\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

**Coeficiente $R^2$:**

$$R^2 = 1 - \frac{\sum_{i} \left( y_i - \hat{y}_i \right)^2}{\sum_{i} \left( y_i - \bar{y} \right)^2}$$

**Accuracy** (para clasificación, donde $y$ son etiquetas): la fracción de puntos con $\hat{y}_i = y_i$.

Las cuatro reciben `(y_true, y_pred)` y devuelven un número.

## Parte 2: `train_test_split` (`ml/model_selection.py`)

`train_test_split(X, y, test_size=0.25, random_state=None)` devuelve `X_train, X_test, y_train, y_test`.

- El test tiene $n_{test} = \lceil \mathrm{test\_size} \cdot n \rceil$ filas, y el train el resto.
- Las filas se reparten **al azar**: no vale coger las últimas.
- Cada fila acaba exactamente en uno de los dos lados.
- La fila $i$ de $X$ y el elemento $i$ de $y$ tienen que ir juntos al mismo lado.
- Con el mismo `random_state`, el resultado tiene que ser siempre el mismo. El generador ya está creado en el esqueleto: `rng`.

## NumPy que te hace falta

| Qué | NumPy |
|---|---|
| valor absoluto, elemento a elemento | `np.abs(a)` |
| cuadrado, elemento a elemento | `a ** 2` |
| suma / media de un array | `np.sum(a)`, `np.mean(a)` |
| comparar elemento a elemento | `a == b` (da un array de `True`/`False`; `True` cuenta como 1) |
| redondear hacia arriba | `np.ceil(x)`, y luego `int(...)` |
| permutación aleatoria de $0, \dots, n-1$ | `rng.permutation(n)` |
| primeros $k$ / el resto de un array | `a[:k]`, `a[k:]` |
| quedarte con las filas de unos índices | `X[indices]` |

## Cómo saber si está bien

```bash
python -m pytest tests/test_metrics.py tests/test_model_selection.py -v
```

## Para pensar

1. ¿Qué vale $R^2$ si siempre predices $\bar{y}$? Compáralo con lo que demostraste en el problema 01: la constante que minimiza el error cuadrático es la media.
2. ¿Puede $R^2$ ser negativo? ¿Qué significaría?
3. Entrena tu `LinearRegression` con `X_train` y mide el MSE en train y en test. ¿Cuál es mayor? ¿Por qué?
