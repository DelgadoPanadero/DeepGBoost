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

## Análisis de suites de datasets (2026-04-18)

### Suite Dev — evaluación de los 8 datasets actuales

Análisis de sensibilidad de cada dataset para detectar cambios en el algoritmo:

| Dataset | Tamaño | Task | Discriminatividad | Justificación |
|---|---|---|---|---|
| Penguins | 333, 3C | classif | BAJA | Demasiado pequeño: la varianza de bootstrap domina sobre diferencias algorítmicas. DGB ya está cerca del techo (~0.99). No discrimina. |
| BankMarketing | 45k, binario | classif | MEDIA | Desbalanceado (12% positivo). Sensible a cambios en NNLS y regularización. El gap -0.007 vs XGB es real pero no dramático. Mantener. |
| Abalone | 4177, 3C | classif | ALTA | DGB gana (+0.009 vs XGB). Benchmark de 3 clases equilibradas. Único caso multiclase donde DGB lidera. Mantener. |
| Adult | 48k, binario | classif | ALTA | Mayor gap (-0.020 vs XGB). Muy sensible a max_depth, min_weight_fraction_leaf, lr. Barómetro de mejoras en regulación. Mantener. |
| NavalVessel | 11k | regresión | ALTA | DGB gana (+0.009 vs XGB). Features con alta correlación lineal, terreno natural de DGBF. Mantener. |
| CaliforniaHousing | 20k | regresión | MEDIA | DGB 2º (-0.004 vs XGB). Mezcla numérica/geoespacial. Útil como regresión de tamaño medio. Mantener. |
| BikeSales | 8k | regresión | ALTA | DGB gana (+0.006 vs XGB). Series temporales tabulares, estacionalidad. Mantener. |
| Concrete | 1k | regresión | MEDIA | DGB gana (+0.003 vs XGB). Muy pequeño: fluctúa bastante entre runs. Útil como smoke test rápido. |

**Conclusión Dev:** Mantener los 8 datasets actuales. Si hay presión de tiempo, Penguins puede suprimirse (poca discriminatividad, DGB ya en techo). Si se necesita un 9º dataset para reforzar clasificación multiclase, `jannis` (OpenML 45021, 57k muestras, 55 features) sería el candidato natural.

### Suite Academic — subconjunto de Grinsztajn et al. (NeurIPS 2022)

Los cuatro OpenML suites del paper son:
- Suite 336: regresión numérica (19 datasets)
- Suite 337: clasificación numérica (16 datasets)
- Suite 335: regresión categórica (17 datasets) — requiere encoding, más complejo
- Suite 334: clasificación categórica (7 datasets) — requiere encoding

Para DeepGBoost se propone el subconjunto de suites 336 + 337 (features numéricas) que evitan la complejidad de encoding categórico y son directamente comparables. Ver propuesta detallada en la respuesta del agente.

**Datasets excluidos por tamaño excesivo:**
- nyc-taxi-green-dec-2016 (did=44143): 581k filas → demasiado lento
- delays_zurich_transport (did=45034): 5.46M filas → impracticable
- Higgs (did=44129): 940k filas → impracticable
- covertype (did=44121): 566k filas → impracticable
- MiniBooNE (did=44128): 73k filas — borderline, posible incluir
- medical_charges (did=44146): 163k filas — borderline

---

## Análisis del benchmark académico (2026-04-20) — 18 datasets Grinsztajn, 10-fold CV

### Diagnóstico de pérdidas: clasificación por significancia estadística

Antes de proponer mejoras es necesario distinguir qué pérdidas son reales (sistemáticas) de cuáles son ruido estadístico. Con 10 folds se puede calcular el SNR = |mean_gap| / std_gap:

| Dataset | DGB-XGB mean | std | SNR | Diagnóstico |
|---|---|---|---|---|
| Electricity | -0.041 | 0.005 | 8.2 | PÉRDIDA REAL |
| Pol_Clf | -0.006 | 0.003 | 2.0 | PÉRDIDA REAL (borderline) |
| Elevators | -0.009 | 0.006 | 1.5 | BORDERLINE |
| Eye_Movements | -0.012 | 0.013 | 0.9 | RUIDO ESTADÍSTICO |
| CPU_Act | -0.002 | 0.002 | 1.0 | RUIDO ESTADÍSTICO |
| Superconduct | +0.003 | 0.003 | 1.0 | DGB GANA en 7/10 folds (tabla original incorrecta) |

**Conclusión:** Solo Electricity (-0.041) y Pol_Clf (-0.006) representan pérdidas reales. Eye_Movements, CPU_Act y Superconduct son estadísticamente indistinguibles de empate.

---

### Causa raíz 1: Degeneration NNLS por n_trees=5 en clasificación (CRÍTICO)

**Observación clave — la asimetría Pol:**
- `Pol_Reg` (26 features, 10082 muestras, regresión): DGB **gana** +0.005 vs XGB
- `Pol_Clf` (mismas features, mismas muestras, clasificación): DGB **pierde** -0.006 vs XGB

Las features y los datos son idénticos. La única diferencia es el config:
- Regresión: `5L × 20T, lr=0.8` → capas anchas
- Clasificación: `20L × 5T, lr=0.1` → capas estrechas

**Mecanismo matemático:** El NNLS resuelve `min ||A·w - pseudo_y||²` donde `A ∈ R^(n,T)`. Con `T=5` (clasificación) la matriz es muy estrecha. Las 5 columnas de `A` provienen de 5 árboles entrenados sobre bootstraps del mismo conjunto con `max_features=None` (todas las features): la única fuente de diversidad entre columnas es el ruido de bootstrap. Con `T=20` (regresión) el NNLS tiene 20 columnas con más diversidad estructural → selección de señal efectiva.

Cuando las 5 columnas son casi colineales, `cond(A) >> 1` y NNLS converge a pesos ~uniformes `1/5` → equivale a un RF de 5 árboles por capa, mucho peor que un RF de 100 árboles.

**Cuantificación:** Para Electricity (8 features), `sqrt(8) ≈ 2.8` → con `max_features=None` y 5 árboles, los splits son casi idénticos. El NNLS sobre un `A(45312, 5)` casi singular produce pesos uniformes → el modelo es un RF de 5 árboles secuenciados 20 veces, no un DGBF.

**Conclusión:** El config `20L×5T` para clasificación es subóptimo para el mecanismo NNLS. El config `5L×20T` (regresión) extrae mucho más valor del NNLS.

---

### Causa raíz 2: Electricity — concept drift temporal + baja diversidad (ESTRUCTURAL)

**Observación:** DGB-RF = -0.049 en Electricity (mayor gap que DGB-XGB = -0.041). Incluso XGB pierde vs RF (-0.009). El RF es el mejor modelo en este dataset.

**Mecanismo:** OpenML split 168 (electricity) usa folds contiguos temporalmente. La distribución P(y|X) cambia a lo largo del tiempo (precios de electricidad → concept drift). El RF es naturalmente robusto al drift porque sus árboles son independientes y pueden capturar distintas épocas de la distribución. Los métodos boosting (DGB y XGB) amplifican el drift: cada capa secuencial corrige residuos de la capa anterior, que se ajustaron sobre datos históricos → el sesgo temporal se acumula en las capas.

**Por qué DGB pierde más que XGB ante RF:** XGB con 100 estimadores secuenciales tiene más iteraciones para corregir el drift. DGB con 20 capas de 5 árboles (≡ 5-árbol RF secuenciado) tiene peor diversidad por capa debido a `n_trees=5`.

**Conclusión:** El gap de Electricity es parcialmente estructural (drift → RF beats boosting) y parcialmente por config (`n_trees=5` baja diversidad). La mejora máxima posible con configuración es reducir la mitad del gap; la otra mitad es inherente al paradigma boosting secuencial.

---

### Hipótesis pendientes (nuevas tras análisis del benchmark académico)

### H — Cambio de config clasificación: de 20L×5T a 10L×10T con lr=0.3 (ALTA PRIORIDAD)

**Fundamento:** La asimetría Pol (mismos datos, DGB gana en regresión y pierde en clasificación) señala directamente al config como causa. El NNLS con 10 árboles por capa tiene matrix `A ∈ R^(n,10)` mucho mejor condicionada que `A ∈ R^(n,5)`. Con `lr=0.3` (=XGBoost default) la comparación es más justa.

**Riesgo:** Reducir `n_layers` de 20 a 10 podría empeorar datasets donde la profundidad es clave (Bank-Marketing, Credit, Heloc). Pero como `n_trees` sube de 5 a 10, el NNLS gana calidad compensando.

**Implementación:** Solo `benchmark/config.json` → `"n_layers": 10, "n_trees": 10, "learning_rate": 0.3`. Zero-code change.

### I — max_features="sqrt" en config clasificación (ALTA PRIORIDAD, bajo riesgo)

**Fundamento:** Con `n_trees=5` y `max_features=None`, la diversidad entre los 5 árboles proviene solo del bootstrap noise. Añadir `max_features="sqrt"` garantiza diversidad estructural (cada árbol usa subconjunto distinto de features). Para Electricity (8f): `sqrt(8)≈3` → 3 features por split → árboles estructuralmente diferentes.

**Riesgo bajo:** `max_features="sqrt"` es el default de RF y está ampliamente validado. No requiere cambio de código, solo en `config.json`.

**Nota importante:** `max_features="sqrt"` con presupuesto fijo reduce la profundidad efectiva de cada árbol (más splits aleatorios), lo que puede afectar negativamente datasets donde la señal está concentrada en pocas features. Sin embargo, con `max_depth=6` ya activo en el config actual, el riesgo de pérdida de precisión por features incompletas es bajo.

### J — Disjoint feature assignment entre árboles de una misma capa (MEDIA PRIORIDAD, requiere código)

**Concepto:** En lugar de `max_features="sqrt"` (aleatorio), asignar particiones disjuntas de features a los T árboles de cada capa. El árbol `t` usa features `{t*F//T, ..., (t+1)*F//T - 1}`. Esto garantiza correlación cero entre columnas de `A` para features no solapadas, condición número mínimo.

**Problema potencial:** Particiones disjuntas pueden perder interacciones entre features de distintos grupos. En datasets donde la señal está en interacciones (Ailerons, Pol), esto puede empeorar.

**Solución alternativa:** Particiones solapadas al 50% (`max_features=2*F//T`) — más diversidad que aleatorio, menos pérdida que disjunto puro.

**Implementación:** `_fit_layer` en `dgbf_multioutput.py` y `dgbf.py`: sustituir `bootstrap_sampler` de features por un generador determinista de subconjuntos de features asignados por `t` (índice del árbol dentro de la capa).

---

## Análisis 1 — Impacto del config H+I (2026-04-20)

**Hipótesis:** El config H+I (10L×10T, lr=0.3, max_features="sqrt") mejora la clasificación respecto al baseline per_class_trees (20L×5T, lr=0.1).
**Datos usados:** `benchmark/results/*_cross_validation_test_scores.json` (8 datasets dev suite, 5-fold CV). Comparación histórica contra BITACORA Experimento 6.
**Script:** `benchmark/analysis/config_hi_analysis.py`

**Resultado:**

| Dataset | Task | DGB viejo | DGB nuevo | ΔDGB | Gap viejo | Gap nuevo |
|---|---|---|---|---|---|---|
| Penguins | clf | 0.9911 | 0.9940 | +0.0029 ▲ | +0.0031 | +0.0059 |
| Abalone | clf | 0.5483 | 0.5502 | +0.0019 ▲ | +0.0100 | +0.0127 |
| Adult | clf | 0.8487 | 0.8563 | +0.0076 ▲ | −0.0204 | −0.0116 |
| BankMarketing | clf | 0.8914 | 0.8781 | −0.0133 ▼ | −0.0125 | −0.0246 |
| NavalVessel | reg | 0.9953 | 0.9953 | ≈0 | +0.0093 | +0.0093 |
| BikeSales | reg | 0.8930 | 0.8907 | −0.0023 | +0.0062 | +0.0020 |
| CaliforniaHousing | reg | 0.8285 | 0.8270 | −0.0015 | −0.0037 | −0.0047 |
| Concrete | reg | 0.9354 | 0.9286 | −0.0068 | +0.0034 | +0.0068 |

Media gap clasificación: −0.0050 → −0.0044 (+0.0005). Media gap regresión: +0.0038 → +0.0034 (−0.0004).

**Conclusión:** El config H+I tiene efecto **asimétrico**. Mejora 3/4 datasets de clasificación — especialmente Adult (+0.0076, mejor resultado histórico en este dataset) y multiclase (Penguins, Abalone). Pero BankMarketing empeora −0.013, lo que cancela la mejora neta. El efecto negativo en BankMarketing sugiere que `lr=0.3` es demasiado agresivo para datasets binarios muy desbalanceados (12% positivo) donde el shrinkage conservador era necesario. La regresión no se ve afectada significativamente.

**Acción recomendada:** Explorar config diferenciada por tipo de desbalance: mantener lr=0.3 para multiclase y datasets equilibrados, pero reducir a lr=0.1 o lr=0.15 para binario desbalanceado. Alternativamente, explorar Hipótesis K: lr adaptativo por clase basado en frecuencia de clase (sin romper la filosofía DGBF).
