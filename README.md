## Evaluating language distances with LangRank

### Background

LangRank ranks transfer languages for NLP tasks. By default, it supports four NLP tasks: dependency parsing, entity linking, machine translation and part-of-speech tagging.
After training some model on an NLP task, and evaluating the model's performance, LangRank trains a classifier on URIEL distances and dataset-dependent features to rank transfer language for each **(task language, transfer language)** pair.

Specifically, LangRank trains a LightGBM classifier on a dataset of **(task language, transfer language, model performance, dataset-dependent features, URIEL distances)** corresponding to a specific model and task *(e.g. machine translation with an LLM on the TED talk corpus)*. This dataset is collected after training the model on the task in each transfer language separately, and evaluating the model's performance on each task language with transfer from each transfer language. "Dataset-dependent features" refer to statistical features of the training dataset used in the task *(e.g. word overlap between task and transfer languages in the TED talk corpora for machine translation)*. The classifier takes in **(task language, dataset-dependent features, URIEL distances)** and outputs a ranking of the best transfer languages, based on its predictions of model performance.

### URIEL(+) Evaluation

We use LangRank as a downstream task to evaluate URIEL(+) distances. The file `run_evaluation.py` contains the code to supply language distances to LangRank and evaluate its performance.
Key functions:
- `replace(file_path)` given a path to a CSV file with language features, replaces URIEL distances (syntactic, etc.) with new distances (computed from your dataset) in the task-specific datasets used to train the LGBM models. **Notes:** the CSV file should have language features as columns, and language codes as rows. Column names must begin with `"S_"` for syntactic, `"P_"` for phonological, `"M_"` for morphological features, and `"INV_"` for phonetic inventory features.
- `dep(file_path, features)`, `el(file_path, features)`, `mt(file_path, features)`, `pos(file_path, features)` evaluate the transfer language ranking for dependency parsing, entity linking, machine translation and part-of-speech tagging tasks respectively, returning a NDCG score. The `features` argument is a list of features used from the task-specific datasets. You should not have to change the 'file_path' argument.

### Metric

LangRank performance is evaluated using NDCG@3, which is the normalized discounted cumulative gain at rank 3. It measures how well the top 3 ranked languages match the actual best transfer language for a given task and task language.
The score ranges from 0 to 100, where 100 refers to perfect ranking.
