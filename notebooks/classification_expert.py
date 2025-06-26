# importing packages
import time
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from computer_ontology.config import*
from sklearn.feature_selection import RFECV
from computer_ontology.custom_funcs import *
import statsmodels.stats.weightstats as stests
from sklearn.ensemble import RandomForestClassifier
from computer_ontology.featurizer import get_mordred
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import make_scorer, roc_auc_score
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall


# fetching the datasets
train = pd.read_csv(expert_train_path)
test = pd.read_csv(expert_test_path)

train = train.set_index('CID')
test = test.set_index('CID')

train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)

train_x = get_mordred(train_x)
test_x = get_mordred(test_x)

# Find common columns
common_columns = test_x.columns.intersection(train_x.columns)

# Filter dataframes to keep only common columns
test_x = test_x[common_columns]
train_x = train_x[common_columns]

cv = IterativeStratification(n_splits=5, order=2)
scorer = make_scorer(roc_auc_score, average = 'macro')

clf = RandomForestClassifier(random_state=0, n_jobs = 2)

start = time.time()

rfecv = RFECV(
    estimator=clf,
    step=0.2,
    cv=cv,
    scoring=scorer,
    min_features_to_select=1,
    n_jobs= -1,
    verbose = 0,
)
rfecv.fit(train_x, train_y)

print(list(train_x.columns[rfecv.support_]))

train_x = rfecv.transform(train_x)
test_x = rfecv.transform(test_x)

print(f"Optimal number of features: {rfecv.n_features_}")


parameters = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['log2', 'sqrt'],
 'min_samples_leaf': [1, 2, 3, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [50, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

'''
parameters = {'bootstrap': [True, False],
 'min_samples_leaf': [1, 2, 3, 4],
 'min_samples_split': [2, 5, 10],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [50,150, 250, 350, 450]}
'''
clf = RandomForestClassifier(random_state=0, n_jobs = 2)

random_search = BayesSearchCV(estimator = clf, search_spaces = parameters, n_iter=6, scoring=scorer, n_jobs = -1, random_state = 0, cv=cv, refit = True, verbose = 1)
random_search.fit(train_x, train_y)

print(random_search.best_params_)

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

end = time.time()
print(end - start)
