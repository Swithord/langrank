from evaluator import DistanceCalculator, SyntacticCalculator, MorphologicalCalculator, InventoryCalculator, PhonologicalCalculator, FeaturalCalculator, ScripturalCalculator, LangRankEvaluator
import pandas as pd
import os
import pickle
import numpy as np
import sys

sys.path.append(os.path.abspath("Latent_islands_URIEL/src/"))
from functions import *

class islandCalculator(DistanceCalculator):
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

print("start")
# Change csv names to correspond to the desired ablation
syn = SyntacticCalculator('URIEL_original_integrate.csv')
morph = MorphologicalCalculator('URIEL_original_integrate.csv')
inv = InventoryCalculator('URIEL_original_integrate.csv')
phon = PhonologicalCalculator('URIEL_original_integrate.csv')
feat = FeaturalCalculator('URIEL_original_integrate.csv')
script = ScripturalCalculator('URIEL_Scriptural.csv')
islands = islandCalculator('URIELPlus_Union_SoftImpute2.csv')

evaluator = LangRankEvaluator({
    'syntactic': syn,
    'morphological': morph,
    'inventory': inv,
    'phonological': phon,
    'featural': feat,
    'scriptural': script,
    'islands': islands
})

features = ['syntactic', 'morphological', 'inventory', 'phonological', 'featural', 'scriptural', 'islands']
# features = ['featural']
tasks = ['taxi1500', 'dep', 'el', 'mt', 'pos']
# tasks = ['pos']
# task = 'taxi1500'
task_col_name = 'task_lang'
transfer_col_name = 'transfer_lang'

# replace/add distances in the `distance_types` columns of the dataset CSV using the corresponding calculators passed into the evaluator.
for task in tasks:
    if task == 'taxi1500':
        iso_conversion = False
    else:
        iso_conversion = True
    print("task is: ", task)
    evaluator.replace_distances(
        dataset_path=f'data/{task}.csv', # path to the task dataset (the one containing columns for task lang, transfer lang, performance)
        distance_types=features, # list of distance types to replace in the dataset. these should match the keys of the dict passed into the evaluator.
        task_col_name=task_col_name, # name of the task language column in your dataset
        transfer_col_name=transfer_col_name, # name of the transfer language column in your dataset
        iso_conversion=iso_conversion # indicate whether to convert lang codes in your dataset from ISO to Glottocode using the file in self.iso_map_file
    )
    print("replacing distances for: ", task)

# run LangRank and evaluate task performance

performance_names = ['f1_score', 'accuracy', 'accuracy', 'BLEU', 'accuracy']

for i in range(len(tasks)):
    task = tasks[i]
    performance_name = performance_names[i]
    additional_features = ['geographic', 'genetic']
    result = evaluator.evaluate(
        dataset_path=f'data/{task}_updated.csv',
        features=features+additional_features, # list of columns in the dataset CSV to use for evaluation
        performance_col_name=performance_name, # name of the column in the dataset CSV containing task performance scores
        task_col_name=task_col_name,
        transfer_col_name=transfer_col_name
    )
    print(f"result of {task}: ", result)