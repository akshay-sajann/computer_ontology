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
train = pd.read_csv(computer_train_path_wo_leff)
test = pd.read_csv(computer_test_path_wo_leff)

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

# Scaling 

scaler = MinMaxScaler()

scaler.fit(train_x)

train_x = pd.DataFrame(data=scaler.transform(train_x), columns=train_x.columns)
test_x = pd.DataFrame(data=scaler.transform(test_x), columns=test_x.columns)

# Feature selection

selector = VarianceThreshold()

train_x = pd.DataFrame(data=selector.fit_transform(train_x),
                       columns=selector.get_feature_names_out(train_x.columns)
                       )

relevant_features = []

for i in train_y.columns:
  selector = SelectKBest(score_func=f_classif, k=3)
  selector.fit(train_x, train_y[i])
  selected_features = selector.get_support()
  relevant_features = relevant_features + train_x.columns[selected_features].values.tolist()

relevant_features = list(set(relevant_features))

train_x = train_x[relevant_features]

# Find common columns
common_columns = test_x.columns.intersection(train_x.columns)

# Filter dataframes to keep only common columns
test_x = test_x[common_columns]
train_x = train_x[common_columns]

print(len(train_x.columns))

cv = IterativeStratification(n_splits=5, order=2)
scorer = make_scorer(roc_auc_score, average = 'macro')

clf = RFExplainer(random_state=0, n_jobs = 2)

start = time.time()

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    importance_getter='perm_feature_importances_',
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

y_hat = random_search.predict(test_x)

f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

print("Computer")
print(f"F1 Score Macro: {f1score_macro}")
print(f"AUROC Macro: {auroc_macro}")
print(f"Precision Macro: {precision_macro}")
print(f"Recall Macro: {recall_macro}")
print("=====================================")

end = time.time()
print(end - start)
