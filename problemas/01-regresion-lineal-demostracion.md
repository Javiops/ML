# 01. Regresión lineal: demostración de la solución cerrada

Cómo se llega a la fórmula que usa el método `"normal"`, paso a paso.

## Notación

Todos los vectores son **columna**. Una $^T$ convierte un vector columna en fila.

| Símbolo | Qué es | Tamaño |
|---|---|---|
| $n$ | número de puntos | número |
| $d$ | número de features | número |
| $X$ | matriz de datos: la fila $i$ es el punto $i$, la columna $j$ es la feature $j$ | $n \times d$ |
| $x_{ij}$ | valor de la feature $j$ en el punto $i$ | número |
| $x_i$ | el punto $i$ (la fila $i$ de $X$, escrita como columna) | $d \times 1$ |
| $y_i$ | objetivo del punto $i$ | número |
| $w$ | pesos, uno por feature | $d \times 1$ |
| $b$ | término independiente (`intercept_`) | número |
| $w^T x_i$ | producto escalar $\sum_{k=1}^{d} w_k x_{ik}$ | número |

Medias sobre los puntos (todas se calculan solo con los datos):

| Símbolo | Definición | Tamaño |
|---|---|---|
| $E[y]$ | $\frac{1}{n}\sum_i y_i$ | número |
| $E[x_j]$ | $\frac{1}{n}\sum_i x_{ij}$, media de la columna $j$ | número |
| $E[x]$ | vector con componentes $E[x_1], \dots, E[x_d]$ | $d \times 1$ |
| $E[x_j x_k]$ | $\frac{1}{n}\sum_i x_{ij}\  x_{ik}$ | número |
| $M$ | matriz con $M_{jk} = E[x_j x_k]$ | $d \times d$ |
| $E[x_j y]$ | $\frac{1}{n}\sum_i x_{ij}\  y_i$ | número |
| $E[xy]$ | vector con componentes $E[x_1 y], \dots, E[x_d y]$ | $d \times 1$ |
| $\mathrm{Cov}(u, v)$ | $E[uv] - E[u]\ E[v]$ | número |
| $\Sigma$ | **matriz de covarianzas** (no es un sumatorio): $\Sigma_{jk} = \mathrm{Cov}(x_j, x_k)$ | $d \times d$ |
| $c$ | vector con $c_j = \mathrm{Cov}(x_j, y)$ | $d \times 1$ |

## 1. El problema

Buscamos $w$ y $b$ que minimicen

$$\mathrm{MSE}(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( w^T x_i + b - y_i \right)^2$$

Es una función de $d + 1$ variables: $w_1, \dots, w_d, b$.

## 2. Basta con anular todas las derivadas parciales

**Lema.** El MSE es convexo en $(w, b)$.

*Demostración.* Para cada $i$, la expresión $w^T x_i + b - y_i$ es afín en $(w, b)$ (una combinación lineal de las variables más una constante). La función $t \mapsto t^2$ es convexa, y una función convexa compuesta con una afín es convexa. Una suma de funciones convexas multiplicada por $\frac{1}{n} > 0$ sigue siendo convexa. $\square$

**Hecho.** Si una función convexa y derivable tiene todas sus derivadas parciales iguales a $0$ en un punto, ese punto es un mínimo global. (Es la versión en varias variables de "en una parábola hacia arriba, donde la derivada es 0 está el mínimo".)

Por tanto, $(w, b)$ es óptimo si y solo si

$$\frac{\partial\  \mathrm{MSE}}{\partial w_j} = 0 \quad \text{para } j = 1, \dots, d, \qquad \frac{\partial\  \mathrm{MSE}}{\partial b} = 0$$

Son $d + 1$ ecuaciones con $d + 1$ incógnitas.

## 3. Las derivadas parciales

Una derivada parcial se calcula derivando respecto a esa variable y tratando las demás como constantes. Por la regla de la cadena, y como $w_j$ solo aparece multiplicando a $x_{ij}$:

$$\frac{\partial\  \mathrm{MSE}}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} \left( w^T x_i + b - y_i \right) x_{ij}$$

$$\frac{\partial\  \mathrm{MSE}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} \left( w^T x_i + b - y_i \right)$$

## 4. Las ecuaciones en términos de medias

Igualamos a $0$, dividimos entre $2$ y repartimos la suma.

**Ecuación de $w_j$.** Escribiendo $w^T x_i = \sum_k w_k x_{ik}$:

$$\frac{1}{n}\sum_{i} \sum_{k} w_k\  x_{ik}\  x_{ij} + b\  \frac{1}{n}\sum_i x_{ij} - \frac{1}{n}\sum_i x_{ij}\  y_i = 0$$

Intercambiando el orden de las sumas en el primer término:

$$\sum_{k=1}^{d} E[x_j x_k]\  w_k + b\  E[x_j] = E[x_j y] \qquad \text{para } j = 1, \dots, d$$

**Ecuación de $b$.** Del mismo modo:

$$\sum_{k=1}^{d} E[x_k]\  w_k + b = E[y]$$

**Forma matricial.** La componente $j$ de $M w$ es $\sum_k M_{jk} w_k = \sum_k E[x_j x_k]\  w_k$, y $E[x]^T w = \sum_k E[x_k]\  w_k$. Así que el sistema es:

$$M w + b\  E[x] = E[xy] \qquad (d \text{ ecuaciones})$$

$$E[x]^T w + b = E[y] \qquad (1 \text{ ecuación})$$

## 5. Eliminar $b$

De la segunda ecuación:

$$b = E[y] - E[x]^T w$$

Sustituyendo en la primera:

$$M w + \left( E[y] - E[x]^T w \right) E[x] = E[xy]$$

Repartiendo (el paréntesis es un número que multiplica al vector $E[x]$):

$$M w + E[y]\  E[x] - \left( E[x]^T w \right) E[x] = E[xy]$$

**El término $\left( E[x]^T w \right) E[x]$ es una matriz por $w$.** Su componente $j$ es

$$E[x_j] \sum_{k=1}^{d} E[x_k]\  w_k = \sum_{k=1}^{d} E[x_j]\  E[x_k]\  w_k$$

que es la componente $j$ de $\left( E[x]\  E[x]^T \right) w$, donde $E[x]\  E[x]^T$ es la matriz $d \times d$ (columna por fila) con elemento $(j, k)$ igual a $E[x_j]\  E[x_k]$. Por tanto:

$$\left( M - E[x]\  E[x]^T \right) w = E[xy] - E[y]\  E[x]$$

## 6. Aparecen las covarianzas

Elemento $(j, k)$ de la matriz de la izquierda:

$$E[x_j x_k] - E[x_j]\  E[x_k] = \mathrm{Cov}(x_j, x_k) = \Sigma_{jk}$$

Componente $j$ del lado derecho:

$$E[x_j y] - E[y]\  E[x_j] = \mathrm{Cov}(x_j, y) = c_j$$

**Resultado.** $(w, b)$ minimiza el MSE si y solo si

$$\Sigma\  w = c, \qquad b = E[y] - E[x]^T w$$

## 7. Resolver el sistema

$\Sigma$ y $c$ solo dependen de los datos, así que $\Sigma\  w = c$ es un sistema lineal de $d$ ecuaciones con $d$ incógnitas.

- **Si $\Sigma$ es invertible:** hay una única solución, $w = \Sigma^{-1} c$, y luego $b = E[y] - E[x]^T w$.
- **Si $\Sigma$ no es invertible:** hay infinitas soluciones. Todas cumplen las condiciones del paso 2, así que todas son mínimos globales y dan el mismo MSE. Basta con elegir una.

**Cuándo $\Sigma$ no es invertible.** Para cualquier vector $v$, desarrollando la definición:

$$v^T \Sigma\  v = \sum_{j,k} v_j v_k \mathrm{Cov}(x_j, x_k) = \mathrm{Var}\left( \sum_j v_j x_j \right) \geq 0$$

$\Sigma$ es singular si y solo si existe $v \neq 0$ con $\Sigma v = 0$. En ese caso $v^T \Sigma v = 0$, es decir, la combinación de columnas $\sum_j v_j x_j$ tiene varianza $0$: vale lo mismo en todos los puntos. Por ejemplo, dos columnas iguales ($v = e_1 - e_2$) o una columna constante.

## 8. Comprobación con $d = 1$

Con una sola feature, $\Sigma$ es el número $\mathrm{Var}(x)$ y $c$ es el número $\mathrm{Cov}(x, y)$:

$$w = \frac{\mathrm{Cov}(x, y)}{\mathrm{Var}(x)}, \qquad b = E[y] - w\  E[x]$$

que es la fórmula clásica de la recta de regresión. El caso singular es $\mathrm{Var}(x) = 0$: todos los $x_i$ iguales.

## 9. El algoritmo

1. Calcular $E[x]$, $E[y]$, $\Sigma$ y $c$. Coste $O(n d^2)$.
2. Resolver $\Sigma\  w = c$. Coste $O(d^3)$. Como $\Sigma$ puede ser singular, en NumPy se usa `np.linalg.lstsq` (que devuelve una solución aunque haya infinitas) y no `np.linalg.inv`.
3. Calcular $b = E[y] - E[x]^T w$.
