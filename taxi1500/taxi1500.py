import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
import letor_metrics
from sklearn.metrics import ndcg_score

def taxi1500(filename, features):
    # Load the data
    data = pd.read_csv(filename)
    logo = LeaveOneGroupOut()
    # Define feature columns and target column
    # features = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']

    data['relevance'] = 0

    # assign relevances from 10 to 0
    for source_lang in data['task_lang'].unique():
        source_lang_data = data[data['task_lang'] == source_lang].copy()
        source_lang_data['rank'] = source_lang_data['f1_score'].rank(method='dense', ascending=False)
        top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
        data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

    groups = data['task_lang']
    ndcg_scores = []

    # Parameters for ranker
    ranker = LGBMRanker(
        boosting_type='gbdt',
        objective ='lambdarank',
        n_estimators=200,
        metric='lambdarank',
        num_leaves=31,
        min_data_in_leaf=5,
        verbose=-1
    )

    #leave one language out
    for train_idx, test_idx in logo.split(data, groups=groups):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        train_X = train_data[features]
        train_y = train_data['relevance']
        test_X = test_data[features]
        test_y = test_data['relevance']

        train_group_sizes = train_data.groupby('transfer_lang').size().tolist()
        train_dataset = lgb.Dataset(train_X, label=train_y,group=train_group_sizes)
        test_dataset = lgb.Dataset(test_X, label=test_y, group=[len(test_y)], reference=train_dataset)


        # Train the model
        ranker.fit(train_X, train_y, group=train_group_sizes, eval_set=[(test_X, test_y)], eval_group=[[len(test_y)]], eval_at=[3])

        # Predict and evaluate NDCG@3
        y_pred = ranker.predict(test_X)
        score = ndcg_score([test_y],[y_pred],k=3)
        # score = letor_metrics.ndcg_score(test_y.values, y_pred, k=3)
        ndcg_scores.append(score)


    #final model
    group = data.groupby('task_lang').size().tolist()
    ranker.fit(data[features], data['relevance'], group=group)
    # print(ranker.feature_importances_)
    # lgb.plot_importance(ranker, importance_type='split')
    # plt.show()
    # Calculate the average NDCG@3 score
    average_ndcg = np.mean(ndcg_scores)
    # print(f'Average NDCG@3: {round(average_ndcg*100,1)}')
    # print([round(float(x), 3) for x in ndcg_scores])
    return average_ndcg * 100


