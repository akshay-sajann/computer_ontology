# importing packages
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


class RFExplainer(RandomForestClassifier):
    def fit(self, X,y):
        X_train, y_train, X_val, y_val = iterative_train_test_split(
            X, y, test_size=0.25, 
        )
        super().fit(X_train,y_train)
        
        self.perm_feature_importances_ = permutation_importance(
            self, X_val, y_val, 
            n_repeats=5, random_state=0,
        )['importances_mean']
        
        return super().fit(X,y)

# fetching the datasets
train = pd.read_csv(alldesc_train_path_2025)
test = pd.read_csv(alldesc_test_path_2025)

train = train.set_index('CID')
test = test.set_index('CID')

train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)

train_x = get_mordred(train_x)
test_x = get_mordred(test_x)

train_x = train_x[selected_features_2025]
test_x = test_x[selected_features_2025]

# Scaling 

scaler = MinMaxScaler()

scaler.fit(train_x)

train_x = pd.DataFrame(data=scaler.transform(train_x), columns=train_x.columns)
test_x = pd.DataFrame(data=scaler.transform(test_x), columns=test_x.columns)

cv = IterativeStratification(n_splits=3, order=2)
scorer = make_scorer(roc_auc_score, average = 'macro')

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

random_search = BayesSearchCV(estimator = clf, search_spaces = parameters, n_iter=100, scoring=scorer, n_jobs = -1, random_state = 0, cv=cv, refit = True, verbose = 1)
random_search.fit(train_x, train_y)

print(random_search.best_params_)

print('rf')

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
