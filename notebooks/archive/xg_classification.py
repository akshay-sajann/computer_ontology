# importing packages
from xgboost import XGBClassifier
import time
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from computer_ontology.config import*
from sklearn.feature_selection import RFECV
from computer_ontology.custom_funcs import *
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
import statsmodels.stats.weightstats as stests
from sklearn.ensemble import RandomForestClassifier
from computer_ontology.featurizer import get_mordred
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# fetching the datasets
train = pd.read_csv(alldesc_train_path)
test = pd.read_csv(alldesc_test_path)

train = train.set_index('CID')
test = test.set_index('CID')

train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)

train_x = get_mordred(train_x)
test_x = get_mordred(test_x)

train_x = train_x[selected_features]
test_x = test_x[selected_features]        

# Scaling 

scaler = MinMaxScaler()

scaler.fit(train_x)

train_x = pd.DataFrame(data=scaler.transform(train_x), columns=train_x.columns)
test_x = pd.DataFrame(data=scaler.transform(test_x), columns=test_x.columns)


cv = IterativeStratification(n_splits=3, order=2)
scorer = make_scorer(roc_auc_score, average = 'macro')

parameters = {
    'tree_method': ['approx', 'hist'],  # Categorical values
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Integer range
    'min_child_weight': [1, 50, 100, 150, 200, 250],  # Integer range
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Float range
    'colsample_bynode': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Float range
    'reg_lambda': [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 25],  # Float, log-scaled values
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8],  # You can specify a list of learning rates if needed
    'n_estimators':[100, 200, 400, 600, 800, 1000, 2000, 4000, 5000, 10000]
}


clf =  XGBClassifier(random_state=0) 

random_search = BayesSearchCV(estimator = clf, search_spaces = parameters, n_iter=100, scoring=scorer, n_jobs = -1, random_state = 0, cv=cv, refit = True, verbose = 1)
random_search.fit(train_x, train_y)

print(random_search.best_params_)

parameters = random_search.best_params_

y_hat = random_search.predict(test_x)

f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

print(f"F1 Score Macro: {f1score_macro}")
print(f"AUROC Macro: {auroc_macro}")
print(f"Precision Macro: {precision_macro}")
print(f"Recall Macro: {recall_macro}")
print("=====================================")

