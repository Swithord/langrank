## Evaluating language distances with LangRank

### Introduction

LangRank ranks transfer languages for cross-lingual transfer across four tasks: dependency parsing, entity linking, machine translation and part-of-speech tagging.
LangRank uses URIEL distances and dataset-dependent features (statistical features of the specific corpora used in each task, e.g. word overlap between task and transfer languages in the TED talk corpora for machine translation), and ranks transfer language for a given task and task language using LightGBM.

### URIEL(+) Evaluation

We use LangRank as a downstream task to evaluate URIEL(+) distances. The file `run_evaluation.py` contains the code to supply language distances to LangRank and evaluate its performance.
Key functions:
- `replace(file_path)` given a path to a CSV file with language features, replaces URIEL distances (syntactic, etc.) with new distances (computed from your dataset) in the task-specific datasets used to train the LGBM models. **Notes:** the CSV file should have language features as columns, and language codes as rows. Column names must begin with `"S_"` for syntactic, `"P_"` for phonological, `"M_"` for morphological features, and `"INV_"` for phonetic inventory features.
- `dep(file_path, features)`, `el(file_path, features)`, `mt(file_path, features)`, `pos(file_path, features)` evaluate the transfer language ranking for dependency parsing, entity linking, machine translation and part-of-speech tagging tasks respectively, returning a NDCG score. The `features` argument is a list of features used from the task-specific datasets. You should not have to change the 'file_path' argument.

### Metric

LangRank performance is evaluated using NDCG@3, which is the normalized discounted cumulative gain at rank 3. It measures how well the top 3 ranked languages match the best transfer language for a given task and task language.
The score ranges from 0 to 100, where 100 refers to perfect ranking.