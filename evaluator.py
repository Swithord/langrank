from typing import Tuple, List, Any
from lightgbm import LGBMRanker
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
print("Test")
from urielplus.urielplus import URIELPlus
print("Test1.25")
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
import pickle
from Latent_islands_URIEL.src.functions import distance_vector_input



class DistanceCalculator:
    """
    Base class for calculating language distances for some distance type.
    """
    def __init__(self):
        print("Test1.5")
        self.uriel = URIELPlus()
        self.df = None
        print("Test2")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate the distance between two languages.
        :param lang1: code for language 1
        :param lang2: code for language 2
        :return: float
        """
        pass

    def _vector_distance(self, lang1: str, lang2: str) -> float:
        """
        Computes language distance by indexing into self.df and calculating the angular distance
        :param lang1: code for language 1
        :param lang2: code for language 2
        :return: float
        """
        if self.df is None:
            raise ValueError("Dataframe is not initialized.")
        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))
        if not intersection:
            return np.nan
        return self.uriel._angular_distance(self.df.loc[lang1].iloc[intersection].to_numpy(),
                                            self.df.loc[lang2].iloc[intersection].to_numpy())


class SyntacticCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("S_")]]
        print("Test3")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class InventoryCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("INV_")]]
        print("Test4")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class PhonologicalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("P_")]]
        print("Test5")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class FeaturalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)
        print("Test6")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class MorphologicalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("M_")]]
        print("Test7")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class GenericCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)
        print("Test8")

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)

class GeographicCalculator(DistanceCalculator):
    def __init__(self, language_centroid_style, dataset_path: str = None, ):
        super().__init__()
        self.uriel.integrate_ethnologue_geo(language_centroid_style)
        print("Test8")
    
    def calculate_distance(self, lang1: str, lang2: str) -> float:

        return self.uriel.new_distance("geographic_w1_normalized", [lang1,lang2])


class IslandCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)

        island_path = os.path.join("Latent_islands_URIEL", "outputs", "bridged_island_full.pkl")
        with open(island_path, 'rb') as f:
            island_full = pickle.load(f)
        self.islands = island_full

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        if self.df is None:
            raise ValueError("Dataframe is not initialized.")
        # Using imputed data, so no need to check for -1
        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))
        # print("shared features: ", len(intersection))
        if not intersection:
            return np.nan
        result = distance_vector_input(self.islands, self.df.loc[lang1].to_numpy(), self.df.loc[lang2].to_numpy())
        return result

class GeographicCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self.uriel.new_distance()


class LangRankEvaluator:
    """
    Evaluate LangRank performance in predicting transfer language based on language distances.
    """
    def __init__(self, calculators: dict[str, DistanceCalculator], iso_map_file: str = 'data/code_mapping.csv'):
        self.calculators = calculators
        self.iso_map = pd.read_csv(iso_map_file, index_col=0).to_dict()['glottocode']
        print("Test9")

    def replace_distances(self, dataset_path: str, distance_types: list[str], task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang', iso_conversion: bool = True) -> None:
        """
        Replace distances in the task dataset based on specified distance types.
        :param dataset_path: path to the task dataset (containing task, transfer, and performance columns)
        :param distance_types: list of distance types to calculate (e.g., ['syntactic', 'inventory', 'phonological']). These should match the keys in self.calculators.
        :param task_col_name: name of the column containing the target language codes
        :param transfer_col_name: name of the column containing the transfer language codes
        :param iso_conversion: whether to convert language codes from ISO to Glottocode using self.iso_map
        :return: None
        """
        if any(distance_type not in self.calculators for distance_type in distance_types):
            raise ValueError("Invalid distance types provided.")
        df = pd.read_csv(dataset_path)
        for index, row in df.iterrows():
            if iso_conversion:
                target_lang = self.iso_map[row[task_col_name]]
                transfer_lang = self.iso_map[row[transfer_col_name]]
            else:
                target_lang = row[task_col_name]
                transfer_lang = row[transfer_col_name]
            for distance_type in distance_types:
                if (distance_type == "geographic"):
                   code_replacements  = pd.read_csv("glottocode_replacements.csv") 
                   for index,row in code_replacements.iterrows():
                       if (target_lang == row["bad_iso"]):
                           target_lang = row["replacement_iso"]
                       if (transfer_lang == row["bad_iso"]):
                           transfer_lang = row["replacement_iso"]
                distance = self.calculators[distance_type].calculate_distance(target_lang, transfer_lang)
                df.at[index, distance_type] = distance
        path_split = dataset_path.split('.')
        df.to_csv(f"{''.join(path_split[:-1])}_updated.{path_split[-1]}", index=False)

    def evaluate(self, dataset_path: str, features: list[str], performance_col_name: str, task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang',
                 baseline_ndcg_scores: list | None = None) -> \
    tuple[int, list[Any], float] | tuple[int, list[Any]]:
        """
        Run LangRank and collect NDCG@3 scores for the given task.
        :param dataset_path: path to the task dataset (containing task, transfer, and performance columns)
        :param features: list of feature names to train the ranker on (e.g., ['syntactic', 'inventory', 'phonological']). These should be columns in the dataset.
        :param performance_col_name: name of the column containing the performance scores (e.g., 'f1_score')
        :param task_col_name: name of the column containing the task language codes (e.g., 'task_lang')
        :param transfer_col_name: name of the column containing the transfer language codes (e.g., 'transfer_lang')
        :param baseline_ndcg_scores: optional list of baseline NDCG scores to compare against for statistical significance testing
        :return: average NDCG@3 score across all leave-one-out iterations
        """
        if baseline_ndcg_scores is None:
            baseline_ndcg_scores = []
        data = pd.read_csv(dataset_path)
        logo = LeaveOneGroupOut()
        data['relevance'] = 0

        for source_lang in data[task_col_name].unique():
            source_lang_data = data[data[task_col_name] == source_lang].copy()
            source_lang_data['rank'] = source_lang_data[performance_col_name].rank(method='min', ascending=False)
            top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
            data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

        groups = data[task_col_name]
        ndcg_scores = []

        ranker = LGBMRanker(
            boosting_type='gbdt',
            objective='lambdarank',
            metric='lambdarank',

            # New parameters
            # n_estimators=50,
            # min_child_samples=5,
            # num_leaves=8,
            # learning_rate=0.05,

            # Old parameters
            n_estimators=100,
            min_data_in_leaf=5,
            num_leaves=16,
            learning_rate=0.1,

            verbose=-1
        )

        for train_idx, test_idx in logo.split(data, groups=groups):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            train_X = train_data[features]
            train_y = train_data['relevance']
            test_X = test_data[features]
            test_y = test_data['relevance']
            train_group_sizes = train_data.groupby(transfer_col_name).size().tolist()

            train_dataset = lgb.Dataset(train_X, label=train_y,group=train_group_sizes)
            test_dataset = lgb.Dataset(test_X, label=test_y, group=[len(test_y)], reference=train_dataset)

            ranker.fit(train_X, train_y, group=train_group_sizes, eval_group=[[len(test_y)]], eval_set=[(test_X, test_y)], eval_at=[3])
            y_pred = ranker.predict(test_X)
            ndcg = ndcg_score([test_y.values], [y_pred], k=3)
            ndcg_scores.append(ndcg)

        # lgb.plot_importance(ranker, importance_type='split')
        # plt.show()

        average_ndcg = np.mean(ndcg_scores) * 100

        if baseline_ndcg_scores and len(baseline_ndcg_scores) == len(ndcg_scores):
            p_value = self._statistical_significance(ndcg_scores, baseline_ndcg_scores)
            return average_ndcg, ndcg_scores, p_value

        return average_ndcg, ndcg_scores

    def _statistical_significance(self, ndcg_scores_1: list[float], ndcg_scores_2: list[float]) -> float:
        """
        Perform a paired t-test to determine if the difference between two sets of NDCG scores is statistically significant.
        :param ndcg_scores_1: first set of NDCG scores
        :param ndcg_scores_2: second set of NDCG scores
        :return: p-value from the t-test
        """
        _, p_value = ttest_rel(ndcg_scores_1, ndcg_scores_2)
        return p_value
