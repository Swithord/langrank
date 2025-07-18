from lightgbm import LGBMRanker
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
from urielplus.urielplus import URIELPlus
import pandas as pd
import numpy as np


class DistanceCalculator:
    """
    Base class for calculating language distances for some distance type.
    """
    def __init__(self):
        self.uriel = URIELPlus()
        self.df = None

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

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class InventoryCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("INV_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class PhonologicalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("P_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class FeaturalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class MorphologicalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("M_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class ScripturalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class LangRankEvaluator:
    """
    Evaluate LangRank performance in predicting transfer language based on language distances.
    """
    def __init__(self, calculators: dict[str, DistanceCalculator], iso_map_file: str = 'code_mapping.csv'):
        self.calculators = calculators
        self.iso_map = pd.read_csv(iso_map_file, index_col=0).to_dict()['glottocode']

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
                distance = self.calculators[distance_type].calculate_distance(target_lang, transfer_lang)
                df.at[index, distance_type] = distance
        path_split = dataset_path.split('.')
        df.to_csv(f"{''.join(path_split[:-1])}_updated.{path_split[-1]}", index=False)

    def evaluate(self, dataset_path: str, features: list[str], performance_col_name: str, task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang') -> float:
        """
        Run LangRank and collect NDCG@3 scores for the given task.
        :param dataset_path: path to the task dataset (containing task, transfer, and performance columns)
        :param features: list of feature names to train the ranker on (e.g., ['syntactic', 'inventory', 'phonological']). These should be columns in the dataset.
        :param performance_col_name: name of the column containing the performance scores (e.g., 'f1_score')
        :param task_col_name: name of the column containing the task language codes (e.g., 'task_lang')
        :param transfer_col_name: name of the column containing the transfer language codes (e.g., 'transfer_lang')
        :return: average NDCG@3 score across all leave-one-out iterations
        """
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
            n_estimators=100,
            metric='lambdarank',
            num_leaves=16,
            min_data_in_leaf=5,
            random_state=0,
            # feature_fraction=0.8,
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

        average_ndcg = np.mean(ndcg_scores)
        return average_ndcg * 100
