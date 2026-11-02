# ML desde cero

Librería de Machine Learning escrita desde cero con Python + NumPy, en paralelo
a aprender Kaggle. La API imita a scikit-learn (`fit` / `predict`) para poder
comparar cada modelo contra sklearn sobre los mismos datos: sklearn hace de
"brute force" y nuestra implementación de "solución".

## Cómo trabajar

Cada módulo tiene un enunciado estilo Codeforces en `problemas/` y unos tests en
`tests/`. El código empieza como un esqueleto que lanza `NotImplementedError`;
mientras un módulo no esté implementado, sus tests salen como *skipped*. Cuando
lo implementas, pasan a ejecutarse y tienen que dar verde.

```bash
pip install -e ".[dev]"
pytest            # todos los tests
pytest -rs        # además lista qué módulos siguen pendientes
```

## Hoja de ruta

| # | Módulo | Enunciado | Estado |
|---|--------|-----------|--------|
| 1 | Regresión lineal (ecuación normal + gradient descent) | [01](problemas/01-regresion-lineal.md) | normal ✅, gd ✅ |
| 2 | Métricas (MAE, MSE, accuracy) + `train_test_split` | — | — |
| 3 | Árbol de decisión (regresión y clasificación) | — | — |
| 4 | Random Forest | — | — |
| 5 | Gradient Boosting | — | — |

## Estructura

```
ml/               # el paquete
  linear_model.py
problemas/        # enunciados estilo CF, uno por módulo
tests/            # tests con pytest, comparando contra sklearn
```
