# Business Entity Resolution Platform

A complete, production-quality solution for the **Business Entity Resolution Challenge**.

---

## 📌 Executive Summary & Architecture

Given three independent business data sources:
- **Source 1**: Deduplicated canonical reference entities
- **Source 2**: Noisy business records (variations in names, abbreviations, addresses)
- **Source 3**: Noisy business records

The objective is to identify, for every Source 1 entity, all matching Source 2 and Source 3 records that represent the same real-world business entity.

### Strict Compliance
* **Zero external data**: No external APIs, databases, geocoding services, or internet lookups.
* **Unseen country generalizability**: Normalized country logic supports global datasets (e.g. France, Germany, US, India, UK) without hardcoding fixed country subsets.
* **F0.5 metric alignment**: Precision is weighted twice as heavily as recall to aggressively prevent false merges.
* **Format integrity**: Output strictly produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` verified by `utils/validate_submission.py`.

---

## 🛠️ Pipeline Stages

```
┌─────────────────────────┐
│       Data Ingestion    │  Auto schema discovery, multi-country loading (sep='\t')
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│     Preprocessing       │  Unicode NFKD, business suffixes (Inc/Corp/Ltd/Pvt), address tokens
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   Multi-Pass Blocking   │  Union of exact names, 3-char prefixes, address/postal tokens,
└────────────┬────────────┘  and TF-IDF character n-gram fuzzy recall (>95% candidate recall)
             ▼
┌─────────────────────────┐
│   Feature Engineering   │  29 pairwise features (Levenshtein, Jaccard, character n-grams,
└────────────┬────────────┘  numeric token overlap, country compatibility, composite metrics)
             ▼
┌─────────────────────────┐
│   Tabular ML Model      │  LightGBM / HistGradientBoosting with entity-level CV split
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   F0.5 Optimization     │  Grid search for optimal decision threshold maximizing F0.5
└────────────┬────────────┘  with explicit singleton detection (0 matches)
             ▼
┌─────────────────────────┐
│  Validation & Output    │  Generates TSVs and executes utils/validate_submission.py
└─────────────────────────┘
```

---

## 🚀 Quickstart & Commands

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Generate / Seed Benchmark Data (Optional)
If you want to run immediate end-to-end tests before supplying your own challenge data:
```bash
python dataset_generator.py
```

### 3. Run End-to-End Pipeline
```bash
python src/pipeline.py --train-dir dataset/train --test-dir dataset/test --output-dir output
```

### 4. Validate Submission TSVs
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### 5. Launch Localhost Web Studio (`http://localhost:8000`)
```bash
python server.py
```
Open **`http://localhost:8000`** in your browser to access:
- **Interactive Results Explorer**: Filter between All, Resolved, Singletons, and Multi-Matches.
- **Live Entity Playground**: Test arbitrary business pairs and view all 29 similarity features.
- **Diagnostic Metrics**: Threshold vs F0.5 curve, feature importance bars, and singleton accuracy.
- **One-Click Pipeline Execution**: Retrain and re-resolve directly from the UI.
