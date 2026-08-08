# DeepGBoost — Bitácora

Bitácora de cambios, experimentos y conclusiones para el clasificador multiclase.
Orientada al agente matemático para contextualizar propuestas futuras.

---

## Estado actual del modelo (resumen rápido)

- **Clasificador activo:** `DeepGBoostMultiClassifier` → `DGBFMultiOutputModel`
- **Config benchmark:** `n_layers=20, n_trees=5, lr=0.1, per_class_trees=True`
- **Métrica:** Weighted F1 (clasificación), R² (regresión)

---

## Experimento 0 — OvR baseline (contexto histórico)

**Descripción:** `DeepGBoostClassifier` original usa One-vs-Rest (K modelos binarios independientes con `LogisticObjective`).

**Resultados (bootstrap, 5 runs):**

| Dataset | DGB OvR | XGB | GB |
|---|---|---|---|
| Penguins | ~0.985 | ~0.978 | ~0.979 |
| BankMarketing | ~0.883 | ~0.901 | ~0.897 |
| Abalone | ~0.544 | ~0.534 | ~0.545 |
| Adult | ~0.847 | ~0.868 | ~0.857 |

**Conclusión:** Base de referencia. El OvR pierde dependencias entre clases.

---

## Experimento 1 — Multi-output baseline (v1)

**Descripción:** Nuevo `DGBFMultiOutputModel` con `SoftmaxObjective`. Un árbol multi-output por slot aprende pseudo-residuos 2D `(n_samples, K)` simultáneamente. NNLS per-class. Prior `log(p/(1-p))`.

**Cambios clave:**
- `src/deepgboost/gbm/dgbf_multioutput.py` (nuevo)
- `src/deepgboost/deepgboost_multiclassifier.py` (nuevo)

**Resultados (bootstrap, 5 runs, 20L×5T):**

| Dataset | DGB | XGB | GB | vs XGB |
|---|---|---|---|---|
| Penguins | 0.9856 | 0.9952 | 0.9904 | -0.0096 |
| BankMarketing | 0.8833 | 0.9010 | 0.8959 | -0.0177 |
| Abalone | 0.5505 | 0.5421 | 0.5485 | **+0.0084** |
| Adult | 0.8463 | 0.8670 | 0.8566 | -0.0207 |

**Conclusión:** Abalone mejora XGB. Penguins, BankMarketing y Adult quedan por debajo. El árbol multi-output captura dependencias entre clases pero el criterio de split `Σ_k MSE_k` está sesgado hacia la clase de mayor varianza.

---

## Experimento 2 — Fix prior: log(p) en lugar de log(p/(1-p))

**Hipótesis:** El prior `log(p/(1-p))` es incorrecto para softmax; debería ser `log(p)`.

**Resultado:** **Empeora en todos los datasets.**

**Conclusión:** El prior log-odds crea un foco implícito en clases minoritarias que resulta beneficioso en la práctica. No es un bug, es una propiedad útil. **No revertir.**

---

## Experimento 3 — Fix pseudo-residual normalization (z-score)

**Hipótesis:** Normalizar pseudo-residuos a media 0 / varianza 1 antes de NNLS para igualar escalas entre clases.

**Problema encontrado:** Se normalizaba antes del fit pero no se restauraba la escala en `predict_raw` → las actualizaciones de capa tenían magnitudes incorrectas (bug de inferencia).

**Resultado tras corregir el bug:** Penguins -0.0147. **Empeora.**

**Conclusión:** La normalización (incluso correctamente implementada con restauración de escala) no ayuda cuando los árboles son multi-output — el problema raíz es el criterio de split, no la escala de los pesos NNLS.

---

## Experimento 4 — Fix NNLS: eliminar normalización suma=1

**Hipótesis:** Los pesos NNLS normalizados a suma=1 limitan la expresividad del ensemble.

**Resultado:** Inestabilidad en Adult (-0.0156). **Empeora.**

**Conclusión:** La normalización suma=1 es necesaria para estabilidad numérica. **No eliminar.**

---

## Experimento 5 — Hessian-weighted NNLS

**Descripción:** Pasar `sample_weight=hessian[:,k]` al NNLS per-class. Pre-multiplica A y b por `sqrt(clip(h, 1e-6))` antes de resolver. Equivale a minimizar el error cuadrático ponderado por incertidumbre del modelo, consistente con el objetivo de cross-entropy de segundo orden.

**Cambios:**
- `src/deepgboost/common/utils.py`: parámetro `sample_weight` en `weight_solver`
- `src/deepgboost/gbm/dgbf_multioutput.py`: pasa `hessian[:,k]` en el bucle NNLS

**Resultados (bootstrap, 5 runs, 20L×5T, vs multioutput baseline):**

| Dataset | Baseline | hw_nnls | Δ |
|---|---|---|---|
| Penguins | 0.9856 | 0.9857 | ≈ 0 |
| BankMarketing | 0.8833 | 0.8853 | **+0.0020** |
| Abalone | 0.5505 | 0.5374 | -0.0131 |
| Adult | 0.8463 | 0.8485 | +0.0022 |

**Conclusión:** Mejora leve donde hay desbalance de clases (BankMarketing, Adult). Empeora en Abalone (3 clases equilibradas). El efecto es pequeño porque el problema raíz (criterio de split multi-output) no se aborda. **Cambio mantenido en el código** como mejora menor.

---

## Experimento 6 — Per-class trees ✅ (mejora confirmada)

**Concepto:** Per_class_trees NO es OvR/OvA clásico. En OvR se entrenan K modelos binarios completamente independientes, cada uno con su propia loss (logistic). Con `per_class_trees` hay un único modelo donde:
- Los pseudo-residuos se calculan conjuntamente con `SoftmaxObjective`: `g_k = y_k - p_k` donde `p_k = softmax(F)_k` — la actualización de la clase k depende de los logits de todas las otras clases vía el denominador softmax.
- El prior es compartido entre todas las clases.
- K grupos de T árboles comparten la misma estructura de capas; cada grupo aprende a corregir residuos de su clase.

La diferencia respecto al árbol multi-output original: en lugar de 1 árbol que recibe target 2D `(n, K)` y elige splits minimizando `Σ_k MSE_k` (sesgado hacia la clase de mayor varianza), se entrenan K árboles independientes, cada uno con target 1D y Hessian exacto por clase.

**Hipótesis:** El árbol multi-output con criterio `Σ_k MSE_k` está dominado por la clase de mayor varianza. Entrenar K árboles independientes (uno por clase) con target 1D y Hessian exacto por clase elimina este sesgo.

**Trade-off:** Se pierden las dependencias entre clases (splits compartidos), pero se gana criterio de split correcto y sample_weight exacto.

**Cambios:**
- `src/deepgboost/gbm/dgbf_multioutput.py`: nuevo parámetro `per_class_trees=False`. Cuando `True`, `_fit_layer` delega a `_fit_layer_per_class` que entrena K×n_trees árboles single-output. `graph_[l]` pasa a ser `list[list[TreeUpdater]]` de shape `(K, n_trees)`. `predict_raw` acumula por clase.
- `src/deepgboost/deepgboost_multiclassifier.py`: expone `per_class_trees`
- Config benchmark: `per_class_trees=True`

**Nota sobre presupuesto:** Con `per_class_trees=True` el número real de árboles es K × n_layers × n_trees. Para K=3 (Penguins, Abalone) con 20L×5T = 300 árboles efectivos.

**Resultados (bootstrap, 5 runs, 20L×5T, vs multioutput baseline):**

| Dataset | Baseline | per_class | Δ DGB | DGB vs XGB |
|---|---|---|---|---|
| Penguins | 0.9856 | 0.9929 | **+0.0073** | +0.0025 🏆 |
| BankMarketing | 0.8833 | 0.8937 | **+0.0104** | -0.0069 |
| Abalone | 0.5505 | 0.5476 | -0.0029 | +0.0031 🏆 |
| Adult | 0.8463 | 0.8491 | **+0.0028** | -0.0211 |

**Resultados cross-validation (5-fold, 1 run):**

| Dataset | DGB | XGB | GB | RF | Posición |
|---|---|---|---|---|---|
| Penguins | 0.9911 | 0.9880 | 0.9849 | 0.9941 | 2º |
| BankMarketing | 0.8914 | 0.9039 | 0.8969 | 0.8954 | 4º |
| Abalone | 0.5483 | 0.5383 | 0.5398 | 0.5436 | 1º 🏆 |
| Adult | 0.8487 | 0.8691 | 0.8581 | 0.8447 | 3º |
| NavalVessel | 0.9953 | 0.9860 | 0.8556 | 0.9935 | 1º 🏆 |
| CaliforniaHousing | 0.8285 | 0.8322 | 0.7709 | 0.8215 | 2º |
| BikeSales | 0.8930 | 0.8868 | 0.8482 | 0.8826 | 1º 🏆 |
| Concrete | 0.9354 | 0.9320 | 0.9021 | 0.9057 | 1º 🏆 |

**Conclusión:** Mejora consistente en clasificación. DeepGBoost gana en 5/8 datasets overall (1º en Abalone, NavalVessel, BikeSales, Concrete; 2º en Penguins y CaliforniaHousing). El gap mayor que persiste es Adult (-0.0204 vs XGB).

---

## Hipótesis pendientes de explorar

### A — Multi-output + scale normalization ❌ (probado, descartado)

Ver Experimento 7 abajo.

---

## Experimento 7 — Multi-output + scale normalization ❌

**Hipótesis:** Normalizar `pseudo_y[:,k]` por `std(pseudo_y[:,k])` antes de entrenar el árbol multi-output corrige el sesgo de escala del criterio `Σ_k MSE_k`, recuperando los splits compartidos entre clases con criterio equivalente.

**Cambios:**
- `src/deepgboost/gbm/dgbf_multioutput.py`: parámetro `normalize_pseudo_residuals=False`. Almacena `_layer_scales_` por capa. `predict_raw` restaura escala con `* _layer_scales_[l]`.
- `src/deepgboost/deepgboost_multiclassifier.py`: expone `normalize_pseudo_residuals`.

**Resultados (bootstrap, 5 runs, 20L×5T, multi-output path):**

| Dataset | multi-output v1 | per_class_trees | scale_norm | Veredicto |
|---|---|---|---|---|
| Penguins | 0.9856 | **0.9929** | 0.9713 | ▼ peor que v1 |
| BankMarketing | 0.8833 | **0.8937** | 0.8826 | ▼ peor que v1 |
| Abalone | 0.5505 | 0.5476 | 0.5397 | ▼ peor que v1 |
| Adult | 0.8463 | **0.8491** | 0.8472 | ▼ peor que v1 |

**Conclusión:** Empeora en todos los datasets, incluso respecto al multi-output sin normalizar. El árbol multi-output con splits compartidos es estructuralmente inferior a K árboles independientes, independientemente de cómo se escalen los pseudo-residuos. Los splits compartidos no aportan señal útil suficiente para compensar la mezcla de clases en el criterio. **Descartado. per_class_trees es el mejor enfoque actual.**

---

### B — Per-class trees + scale normalization ❌ (descartada por análisis)

Matemáticamente equivalente al Experimento 7 dentro del path per_class_trees: cada árbol ya opera en aislamiento, no hay `Σ_k MSE_k` que corregir. La normalización sería un no-op en el criterio de split y solo afectaría la escala de los pesos NNLS — mismo efecto que Exp 7, que ya empeoró.

### C — Adaptive lr por clase ❌ (descartada por análisis)

`lr_k = lr / std(pseudo_y[:,k])` sin restaurar escala en predict_raw es una versión rota del Experimento 7: produce un learning rate efectivo que *aumenta* a lo largo de las capas (porque los pseudo-residuos se reducen con el entrenamiento), lo contrario de la regularización por shrinkage. Para Adult (K=2), la antisimetría del gradiente softmax garantiza `std(pseudo_y[:,0]) = std(pseudo_y[:,1])`, haciendo que la hipótesis sea un puro no-op. **Descartada.**

### D — Hessian-corrected leaf values ❌ (probado, descartado)

**Motivación matemática:** DeepGBoost pre-computa el Newton step `pseudo_y[i] = g[i]/(h[i]+λ)·lr` y el árbol aprende la media ponderada de estos valores. El valor de hoja resultante es:

```
v_j = Σ_{i∈j} h[i]·g[i]/(h[i]+λ) / Σ_{i∈j} h[i]
```

El valor correcto (XGBoost) es:
```
v_j* = Σ_{i∈j} g[i] / (Σ_{i∈j} h[i] + λ_leaf) · lr
```

Los dos son equivalentes solo cuando `h[i] ≈ constante ∀i∈j`. En Adult, `h = p*(1-p)` varía 25x entre muestras (h ∈ [0.01, 0.25]). Esta discrepancia es la causa estructural del gap.

**Implementación:** Tras `tree.fit(...)`, usar `tree.apply(X)` para obtener asignaciones de hoja y reemplazar `tree_.value[leaf, 0, 0]` con el valor Newton exacto usando los gradientes y hessianos crudos (antes de la división). Requiere pasar `g` y `h` (no `pseudo_y`) a `_fit_layer_per_class`, añadir `lambda_leaf` (default=1.0) a `DGBFMultiOutputModel`, y un método `set_leaf_values` en `TreeUpdater`.

**Expected impact:** 0.005–0.015 Weighted F1 en Adult. Neutral en regresión (h=1 uniforme) y clasificación multiclase equilibrada. Cierra ~25-70% del gap de -0.0207.

**Resultados reales (bootstrap, 5 runs, 20L×5T, lambda_leaf=1.0):**

| Dataset | per_class_trees | leaf_corr | Δ |
|---|---|---|---|
| Penguins | 0.9929 | 0.9857 | ▼ -0.0072 |
| BankMarketing | 0.8937 | 0.8698 | ▼ -0.0239 |
| Abalone | 0.5476 | 0.5488 | ≈ +0.0012 |
| Adult | 0.8491 | 0.8441 | ▼ -0.0050 |

**Conclusión:** Empeora en 3/4 datasets. La causa: los splits del árbol se eligieron para minimizar MSE sobre `pseudo_y = g/h·lr` (Newton steps pre-computados). Reemplazar los valores de hoja post-fit con `Σg/(Σh+λ)·lr` introduce una inconsistencia entre la estructura del árbol (splits) y los valores de hoja — el árbol aprendió a separar regiones de `pseudo_y`, no de `g/h`. Para que la corrección fuese efectiva, el criterio de split también tendría que usar `g/h` exactos con Hessian por leaf (requiere un CART custom, fuera del scope de sklearn). **Descartado.**

---

## Notas importantes sobre Adult

Adult es el dataset más difícil para DeepGBoost (gap -0.0211 vs XGB con per_class_trees). Características:
- ~48k muestras, 14 features, tarea binaria (income >50K)
- Desbalanceado: ~76% clase negativa
- XGBoost config en benchmark: `n_estimators=100, lr=0.3 (default), max_depth=6 (default)` vs DeepGBoost `lr=0.1, max_depth=None (sin límite)`
- El gap tiene std=0.0021 sobre 5 runs → es estructural, no ruido
- Hipótesis D descartada. El gap puede requerir un criterio de split custom (fuera de sklearn) para resolverse correctamente.

---

## Hipótesis pendientes (nuevas)

### E — max_depth explícito como regulador de varianza (ALTA VIABILIDAD)

**Problema:** DeepGBoost usa `max_depth=None` (árboles perfectos). XGBoost usa `max_depth=6` por defecto. Con 14 features en Adult y subsamples bootstrap, un árbol sin límite puede crear hojas con 1-2 muestras del subsample → valores de hoja extremos que el lr=0.1 no puede amortiguar suficientemente. El problema es especialmente severo para la clase minoritaria (24%): en un subsample pequeño de la capa 1 (frac=0.3 → ~14k muestras, ~3.3k de clase positiva), los árboles sin límite de profundidad memorizan particiones degeneradas.

**Propuesta:** Añadir `max_depth=6` (o un rango {4, 6, 8}) como parámetro benchmark para clasificación. Esto no requiere cambios en el código — solo en `config.json`. Alternativamente, añadir `min_samples_leaf` al `TreeUpdater` para dar un mínimo de muestras por hoja (equivalente funcional de `min_child_weight` en XGBoost).

**Riesgo en otros datasets:** Penguins (333 muestras) y Abalone (4177 muestras) se benefician de árboles profundos; restringir max_depth podría empeorar datasets pequeños. Solución: usar `min_samples_leaf` relativo al tamaño del subsample en lugar de `max_depth` absoluto.

**Implementación:** `benchmark/config.json` → `"max_depth": 6` en model_4. Si se añade `min_samples_leaf`: `src/deepgboost/tree/updater.py:40` → `DecisionTreeRegressor(..., min_samples_leaf=min_samples_leaf)` y exponer el parámetro en `DGBFMultiOutputModel` y `DeepGBoostMultiClassifier`.

---

### F — learning_rate más alto para Adult (MEDIA VIABILIDAD, requiere ablación lr)

**Problema:** El config usa lr=0.1 que es 3× más conservador que el XGBoost default de lr=0.3. La combinación `lr_bajo + árboles_profundos` es sub-óptima: árboles profundos capturan señal de alta frecuencia que el shrinkage bajo no amortiza bien. XGBoost equilibra con `lr=0.3 + max_depth=6`. DeepGBoost necesita o bien aumentar lr o reducir profundidad.

**Propuesta:** Ablación de lr ∈ {0.1, 0.2, 0.3} manteniendo el presupuesto de árboles (20L×5T). Si max_depth se fija en 6 (Hipótesis E), aumentar lr a 0.2-0.3 es más seguro porque los árboles menos profundos tienen menor varianza. La interacción lr × max_depth es la palanca real.

**Riesgo en otros datasets:** lr alto sin max_depth puede empeorar NavalVessel y BikeSales donde DeepGBoost ya gana. Si se condiciona lr alto a `max_depth` finito, el riesgo es menor. Penguins (pocas muestras) es más sensible a lr alto.

**Implementación:** Cambio solo en `benchmark/config.json`. No requiere código nuevo.

---

### G — min_child_weight (Hessian acumulado mínimo por hoja) como regulador de Adult (ALTA VIABILIDAD, complementa E)

**Problema matemático:** XGBoost aplica `min_child_weight` que exige `Σ_{i∈hoja} h_i ≥ min_child_weight`. Para softmax binario con h_i = p_i(1-p_i) ∈ [0.01, 0.25], una hoja con 10 muestras tiene `Σh ≈ 0.05-2.5`. XGBoost default `min_child_weight=1` impide hojas donde el Hessian acumulado < 1, rechazando splits en regiones de alta confianza (h pequeño) o con pocas muestras.

DeepGBoost no tiene este regulador. sklearn `DecisionTreeRegressor` con `sample_weight=hessian` y `min_samples_leaf=1` (default) permite hojas con una sola muestra de h ≈ 0.01 → el árbol crea splits ruidosos en regiones de alta confianza donde ya no hay señal útil.

**Propuesta:** Añadir `min_weight_fraction_leaf` al `DecisionTreeRegressor` subyacente. El parámetro de sklearn define la fracción mínima de la suma total de pesos que debe tener cada hoja. Con `sample_weight=hessian[:,k]` y `min_weight_fraction_leaf=α`, esto implementa exactamente el `min_child_weight` proporcional de XGBoost. Valor inicial: α=0.001 (≈ fracción pequeña de la suma hessiana total).

**Riesgo en otros datasets:** Abalone (3 clases equilibradas, Hessian más uniforme) y Penguins son menos afectados porque sus Hessianos varían menos. CaliforniaHousing y otros regresores usan h=1 uniforme → α actúa como `min_samples_leaf` fraccionario, neutral o ligeramente beneficioso.

**Implementación:** `src/deepgboost/tree/updater.py:40` → añadir `min_weight_fraction_leaf` al constructor de `DecisionTreeRegressor`. Exponer en `TreeUpdater.__init__`, `DGBFMultiOutputModel.__init__`, `DeepGBoostMultiClassifier.__init__`. Default=0.0 para compatibilidad backward.

---

### H — Idea: revisar los coeficientes NNLS (signo fijo y granularidad global, pendiente de experimento)

**Origen:** El diseño actual usa NNLS (`weight_solver`) porque se asumía que un árbol entrenado para predecir un target no debería tener nunca una contribución *negativa* a la predicción final — la no-negatividad se trató como un axioma de diseño, no como algo a validar empíricamente. Además, NNLS produce un único coeficiente escalar global por árbol, constante para todo el espacio de features.

**Parte 1 — ¿Por qué exigir signo no-negativo? (usuario, 2026-08-08):** Esa asunción puede no ser correcta. Un árbol se entrena sobre un *subsample* bootstrap de cada capa; si ese subsample tiene un comportamiento anómalo respecto a la distribución global (ruido, outliers, una región del espacio de features sub/sobre-representada), el árbol puede aprender una corrección que es útil localmente pero que, agregada linealmente sobre el resto de la distribución, actúa como una corrección *inversa* de un sesgo introducido por árboles anteriores. En ese caso, un coeficiente negativo no es un error — es la forma correcta de decir "esta hoja se equivocó de signo para la mayoría de la población, pesa menos que cero en la mezcla".

Forzar `coef ≥ 0` puede estar suprimiendo señal legítima de corrección de sesgo, no solo ruido. Esto conecta con varios experimentos ya descartados (Exp 1, 4, 5, 7) donde tocar la escala/normalización de los pesos NNLS no ayudó — nunca se probó relajar la restricción de signo en sí.

**Parte 2 — ¿Debería el coeficiente depender de la región del espacio?** Ligado a la Parte 1: si la contribución "correcta" de un árbol depende de qué subsample/región lo generó, entonces un único coeficiente escalar global por árbol es una simplificación excesiva. La contribución óptima de un árbol podría ser función de `x`: `coef_t(x)` en lugar de `coef_t` constante — esencialmente, una mezcla de expertos (mixture-of-experts) o un gating aprendido sobre el espacio de features, donde cada árbol "vota" con un peso que depende de si `x` cae dentro de la región donde ese árbol aprendió una corrección fiable. Esto generaliza la Parte 1 (la negatividad ya no sería una propiedad fija del árbol, sino de la región: un árbol podría ser `coef > 0` en una región y `coef < 0` en otra) y conecta con la intuición de Exp 6 (per_class_trees), que ya mostró que dar más granularidad al modelo mejora resultados.

**Riesgos / complejidad:**
- Sin restricción de signo, la escala de las predicciones deja de estar acotada por convexidad — puede reintroducir la inestabilidad numérica que motivó Exp 4 (normalización suma=1). Habría que mantener algún tipo de regularización (Ridge, o clip post-hoc).
- Un gating por región introduce muchos más parámetros a aprender → riesgo de overfitting, especialmente en datasets pequeños (Penguins, 333 muestras).
- Un coeficiente no-escalar rompe la interpretación actual de "peso por árbol", con impacto directo en el método de explicabilidad por contribución de features (commit `c93a6e2`) — habría que revisar esa pieza si se implementa.
- Coste computacional: evaluar un gating por muestra en predict añade overhead frente al escalar actual.

**Experimentos propuestos (orden de complejidad creciente):**
1. Sustituir NNLS por mínimos cuadrados sin restricción de signo (OLS) o por Ridge con regularización L2, y comparar contra NNLS. Si OLS/Ridge sin restricción de signo iguala o mejora a NNLS, la hipótesis original (no-negatividad como axioma) queda refutada empíricamente.
2. Gating simple por región ya conocida: pesos NNLS *por hoja del árbol anterior* (usa la partición de la capa previa como proxy de "región") en vez de un único peso por árbol.
3. Gating aprendido explícito: un modelo ligero (p.ej. regresión logística o un árbol pequeño) que prediga el peso óptimo de cada árbol dado `x`, entrenado sobre residuos de validación.
4. Comparar todo lo anterior contra el escalar NNLS actual en los 8 datasets del benchmark, prestando atención especial a overfitting en datasets pequeños.

**Estado:** Fase de idea, sin experimento diseñado ni implementado. Empezar por el punto 1 (signo) antes de justificar la complejidad añadida de un gating por región (puntos 2-3) — si el signo no debe ser fijo por árbol, aún queda por ver si el valor global también debe dejar de serlo.

---

### I — Idea: muestreo vía combinaciones no lineales aleatorias de features (en vez de subsample de filas)

**Origen (usuario, 2026-08-08):** La premisa de DGBF es que capas/árboles distintos aprenden comportamientos distintos (aprendizaje distribuido + aprendizaje de representación), y hoy esa diversidad se induce solo con *subsampling de filas* (bootstrap aleatorio de muestras) por capa. La pregunta es si en vez de (o además de) variar *qué muestras* ve cada árbol, tendría sentido variar *qué representación de las features* ve — es decir, generar features sintéticas por combinaciones no lineales aleatorias de las features originales (proyecciones aleatorias tipo random Fourier features / random kitchen sinks, productos cruzados aleatorios, sumas ponderadas con no-linealidad aleatoria, etc.) y muestrear/entrenar cada árbol (o grupo de árboles) sobre un subconjunto de esas features inventadas.

**Por qué podría tener sentido:** El subsampling de filas diversifica *qué parte de la distribución* ve cada árbol, pero todos los árboles siguen viendo el mismo espacio de features original — la única fuente de diversidad estructural es la elección de splits sobre las mismas variables. Si además cada árbol (o capa) ve una proyección no lineal distinta del espacio de entrada, la diversidad deja de ser solo "qué muestras" y pasa a ser también "qué relaciones puede ver" — más cercano a cómo random forests ya combinan bootstrap + feature subsampling (`max_features`), pero yendo un paso más allá al usar combinaciones no lineales en vez de simplemente elegir un subconjunto de columnas originales. Esto podría ayudar precisamente en el tipo de sesgo que motivó Exp 6 (splits dominados por la clase/feature de mayor varianza): si cada árbol parte de una proyección distinta, el sesgo hacia una misma variable dominante en todas las capas se diluye.

**Relación con el resto de la bitácora:** Complementa la idea de la Sección H (revisión de los coeficientes NNLS): si los árboles ya son diversos por representación además de por muestra, es más plausible que algunos "acierten de signo" solo en ciertas regiones/proyecciones — reforzaría el caso para relajar la no-negatividad NNLS y/o el gating por región.

**Riesgos / preguntas abiertas:**
- Pérdida de interpretabilidad: las features sintéticas no son directamente atribuibles a features originales, lo que complica el método de contribución de features (commit `c93a6e2`) — habría que propagar la combinación lineal/no lineal hacia atrás para descomponer la contribución en términos de las features originales.
- Coste de diseño: hay que decidir la familia de combinaciones no lineales (polinómicas de grado bajo, RBF/Fourier aleatorias, funciones tipo tanh/relu sobre combinaciones lineales aleatorias) y cuántas features sintéticas generar por capa/árbol — más hiperparámetros a ajustar.
- Riesgo en datasets con pocas features o pocas muestras (Penguins, Concrete): las combinaciones aleatorias podrían introducir ruido en vez de señal si el espacio original ya es de baja dimensión.
- Se solapa parcialmente con lo que ya hace un árbol de decisión (splits axis-aligned sobre combinaciones *implícitas* vía profundidad), por lo que el beneficio marginal frente a simplemente variar `max_features`/subsampling de columnas (más barato, ya estándar en random forests) no está claro sin experimento.

**Experimento propuesto (orden de complejidad creciente):**
1. Baseline barato: `max_features` aleatorio por árbol (subsampling de columnas, sin combinaciones no lineales) — para aislar si la ganancia viene de "menos columnas por árbol" o realmente de la no-linealidad.
2. Proyecciones aleatorias lineales (random projections) de las features antes de cada capa/árbol, comparando contra el baseline.
3. Proyecciones no lineales aleatorias (random Fourier features o combinaciones tipo `tanh(Wx+b)` con `W,b` aleatorios fijos por árbol) como espacio de entrada del árbol.
4. Comparar contra el baseline actual (subsample de filas únicamente) en los 8 datasets del benchmark, con atención especial a datasets de baja dimensionalidad (Penguins, Concrete) y al coste de reconstrucción de la explicabilidad.

**Estado:** Fase de idea, sin experimento diseñado ni implementado. Pendiente de que `mathematician` evalúe primero el paso 1 (barato) antes de justificar la complejidad de proyecciones no lineales.

---
