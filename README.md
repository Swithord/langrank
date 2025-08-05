## Evaluating language distances with LangRank

### Background

LangRank ranks transfer languages for NLP tasks. By default, it supports four NLP tasks: dependency parsing, entity linking, machine translation and part-of-speech tagging.
After training some model on an NLP sub-task, and evaluating the model's performance, LangRank trains a ranker on URIEL distances and dataset-dependent features to rank the best transfer language for each task language.

Specifically, LangRank trains a LightGBM ranker on a dataset of **task language, transfer language, model performance, dataset-dependent features, URIEL distances** corresponding to a specific model and sub-task *(e.g. machine translation with an LLM on the TED talk corpus)*. This dataset is obtained after evaluating the model's performance on the sub-task, which is done separately. Cross-validation is performed by leaving out one task language at a time, training on the remaining dataset, and predicting the best transfer language for the task language that is left out.

The ranker takes in **dataset-dependent features, URIEL distances** for all possible transfer languages and outputs a ranking of the best transfer languages, based on its predictions of model performance. "Dataset-dependent features" refer to statistical features of the training dataset used in the task *(e.g. word overlap between task and transfer languages in the TED talk corpora for machine translation)*.

For more details, refer to the [original paper](https://aclanthology.org/P19-1301.pdf).

### URIEL(+) Evaluation

We use LangRank as a downstream task to evaluate URIEL(+) distances. In each sub-task that we want to evaluate LangRank on, the general pipeline is to (1) add/replace URIEL distances in the sub-task dataset, and (2) run LangRank on the new dataset. All relevant classes are found in the `evaluator.py` file.

The `DistanceCalculator` *abstract* class provides a public method, `calculate_distance(lang1: str, lang2: str) -> float` for computing distance of a particular type (e.g. syntactic) between `lang1` and `lang2`. For extendability and customizability, it is up to you to implement the class and method however you'd like. You should decide for yourself whether this function will take in ISO codes or Glottocodes.

The `LangRankEvaluator` class is responsible for running LangRank and evaluating its performance. When initializing this class, you will need to pass in a `str` -> `DistanceCalculator` dictionary, which will define all distance types your evaluator will support. You may optionally pass in a path to the ISO-Glottocode mapping file (but depending on the sub-task dataset and your calculator implementations, you may not need this.)

Key methods of the `LangRankEvaluator` are:
- `replace_distances`: Add/replace distances in the sub-task dataset. Parameters:
  * `dataset_path`: path to the sub-task dataset.
  * `distance_types`: list of distance types to calculate. These should be keys of the dictionary passed in to `LangRankEvaluator`. For example, if `distance_types=['syntactic']`, this method will add/replace values in the "syntactic" column of the dataset CSV with values calculated from the `DistanceCalculator` corresponding to "syntactic".
  * `task_col_name`: name of the column containing the target language codes. Default is "target_lang".
  * `transfer_col_name`: name of the column containing the transfer language codes. Default is "transfer_lang".
  * `iso_conversion`: whether to convert language codes from ISO to Glottocode before calculating distances (i.e. if `True`, language codes are converted to Glottocodes before being passed into *all* `DistanceCalculator`s.). This will use the file path passed into `LangRankEvaluator` (or the default file path).
- `evaluate`: Run LangRank and returns the average NDCG@3 score. Parameters:
  * `dataset_path`
  * `features`: list of dataset columns to be used when training the ranker.
  * `performance_col_name`: name of the column containing the performance scores (e.g., 'f1_score' in Taxi1500, 'BLEU' in machine translation)
  * `task_col_name`
  * `transfer_col_name`
  * `baseline_ndcg_scores`: (optional) list of NDCG scores from the baseline. If provided, a p-value will additionally be returned.

See `evaluate.ipynb` for a commented code example.


### Metric

LangRank performance is evaluated using NDCG@3, which is the normalized discounted cumulative gain at rank 3. It measures how well the top 3 ranked languages match the actual best transfer language for a given task and task language.
The score ranges from 0 to 100, where 100 refers to perfect ranking.
