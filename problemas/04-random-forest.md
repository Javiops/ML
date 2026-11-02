# 04. Random Forest

Un solo árbol profundo memoriza el train: cambias unos pocos puntos y el árbol sale muy distinto. Es decir, tiene mucha **varianza**. Un Random Forest entrena muchos árboles distintos y hace la media de sus predicciones.

## Notación

| Símbolo | Qué es | Tamaño |
|---|---|---|
| $n$ | número de puntos | número |
| $d$ | número de features | número |
| $X$ | matriz de datos: la fila $i$ es el punto $i$, la columna $j$ es la feature $j$ | $n \times d$ |
| $y_i$ | objetivo del punto $i$ | número |
| $B$ | número de árboles (`n_estimators`) | número |
| $m$ | features que se prueban en cada nodo (sale de `max_features`) | número |
| $\hat{f}_b(z)$ | predicción del árbol $b$ en el punto $z$ | número |

## El algoritmo

Para cada árbol $b = 1, \dots, B$:

1. **Bootstrap.** Saca $n$ índices $i_1, \dots, i_n$ al azar **con reemplazo** de $\lbrace 0, \dots, n - 1 \rbrace$. El árbol $b$ se entrena con las filas $X_{i_1}, \dots, X_{i_n}$ y sus $y$. Algunos puntos salen repetidos y otros no salen. Con `bootstrap=False` se usan las $n$ filas tal cual.
2. **Árbol con features al azar.** Es el árbol del problema 03 con un cambio: en cada nodo, en vez de probar las $d$ features, se prueban solo $m$ elegidas al azar sin reemplazo. Las features que son constantes en ese nodo no cuentan, porque no se pueden cortar. Si hay menos de $m$ no constantes, se prueban todas esas.

La predicción del bosque es la media:

$$\hat{f}(z) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(z)$$

**Cuánto vale $m$** (como en sklearn, salvo que sklearn da error con un entero $k > d$):

| `max_features` | $m$ |
|---|---|
| `None` o `1.0` | $d$ |
| `"sqrt"` | $\max\left(1, \lfloor \sqrt{d} \rfloor\right)$ |
| `"log2"` | $\max\left(1, \lfloor \log_2 d \rfloor\right)$ |
| un float $f$ | $\max\left(1, \lfloor f d \rfloor\right)$ |
| un entero $k$ | $\min(k, d)$ |

## Por qué funciona

Supón que cada árbol tiene varianza $\sigma^2$ y que dos árboles distintos tienen correlación $\rho$. Entonces la media de $B$ árboles tiene varianza

$$\mathrm{Var}\left( \frac{1}{B} \sum_{b} \hat{f}_b \right) = \frac{1}{B^2} \left( B \sigma^2 + B (B - 1) \rho \sigma^2 \right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

El segundo término se va a $0$ al subir $B$. El primero no: solo baja si los árboles se parecen poco. Para eso sirven el bootstrap y las features al azar, para que los árboles salgan distintos y $\rho$ baje.

## Qué hay que implementar

- En `ml/tree.py`: los parámetros `max_features` y `random_state` del árbol, y la elección de las $m$ features en cada nodo (`_features_al_azar`). `best_split` recibe la lista de features que tiene que probar.
- En `ml/ensemble.py`: `RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1, max_features=1.0, bootstrap=True, random_state=None)`, con `fit`, `predict` y la lista de árboles en `estimators_`.

*Resuelto en esos dos ficheros.*

**Diferencias con sklearn.** El azar no es el mismo, así que las predicciones no coinciden punto a punto: los tests comparan el error en test. Además, sklearn no duplica las filas del bootstrap sino que les da peso, así que su `min_samples_leaf` cuenta puntos distintos y el nuestro cuenta repetidos.

## Cómo saber si está bien

```bash
python -m pytest tests/test_ensemble.py -v
```

Los tests comprueban las reglas de `max_features`, que las features constantes no cuentan, que un bosque de un árbol sin bootstrap es el árbol del problema 03, que la predicción es la media de los árboles, que el bosque gana a un solo árbol y que el error en test está a menos de un 15 % del de sklearn.

## Para pensar

1. Con `max_features=None` y `bootstrap=False`, ¿en qué se diferencian los $B$ árboles? ¿Qué predice el bosque?
2. ¿Qué probabilidad tiene un punto de no salir en el bootstrap de un árbol? Calcula el límite de $\left( 1 - \frac{1}{n} \right)^n$. Con los árboles en los que no sale se puede estimar el error en test sin separar datos (el *out-of-bag error*).
3. Prueba $B = 1, 10, 100$ y mide el MSE en test. ¿Puede empeorar al poner más árboles?
