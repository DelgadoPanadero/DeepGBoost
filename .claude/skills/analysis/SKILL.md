---
name: analysis
description: Exploratory analysis of DeepGBoost benchmark results. Use when the user wants to investigate a hypothesis about the algorithm's behavior, compare models across datasets, understand why a model wins or loses, or extract insights from benchmark results. Trigger on phrases like "analiza los resultados", "investiga por qué", "compara los modelos", "explora el benchmark", "qué conclusiones sacas de", "analiza la hipótesis", "benchmark analysis", "explora los datos del benchmark", or when the mathematician agent needs to validate a claim with empirical data.
version: 1.1.0
---

# Benchmark Analysis

## Step 1 — Formulate hypothesis
State clearly: hypothesis, operationalization, expected outcome. If the request is vague, ask one clarifying question before continuing.

## Step 2 — Audit data
Check `benchmark/results/` (JSON scores), `benchmark/data/` (cached datasets), and `benchmark/config.json` / `config_academic.json`. Assess explicitly: are the existing results sufficient? If not, tell the user what's missing and ask permission before continuing.

## Step 3 — Script
Check `benchmark/analysis/` for existing scripts — reuse or extend before creating a new one. New scripts go in `benchmark/analysis/<name>.py`. Rules:
- Load from `benchmark/results/` or `benchmark/data/`
- Save all outputs (plots, intermediates) to `/tmp/` — never inside the project
- Print a summary table to stdout
- If it exists but doesn't do exactly what you need, extend it.
- Run with `.venv/bin/python benchmark/analysis/<name>.py`

If the script does not exist, write it under the name `benchmark/analysis/<name>.py`. Rules:
- Try to write generic code
- Test on a small subset of data first — don't run a full benchmark analysis without testing the script first.

Fix any errors before moving on.

## Step 4 — Interpret
Write: key numbers, YES/NO/PARTIALLY verdict, 2–4 sentence explanation, caveats. Be honest about ambiguity.

## Step 5 — Update BITACORA.md
Only update if the result is clearly informative (confirmed/refuted hypothesis, new root cause, new hypothesis). Append only — never rewrite. Format:

```
## Análisis N — Title (YYYY-MM-DD)
**Hipótesis:** ...
**Datos usados:** ...
**Script:** benchmark/analysis/<name>.py
**Resultado:** ...
**Conclusión:** ...
**Acción recomendada:** ...
```

Stop and ask the user before continuing if data is insufficient, a long benchmark run is needed, or you are about to write anywhere except `benchmark/analysis/` and `benchmark/BITACORA.md`.
