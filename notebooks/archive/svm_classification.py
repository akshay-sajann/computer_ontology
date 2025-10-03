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

# fetching the datasets
train = pd.read_csv(alldesc_train_path)
test = pd.read_csv(alldesc_test_path)

train = train.set_index('CID')
test = test.set_index('CID')

train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)

train_x = get_mordred(train_x)
test_x = get_mordred(test_x)

selected_features = ['nS', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTare-1l', 'BalabanJ', 'SpMAD_Dzp', 'RNCG', 'RPCG', 'C2SP1', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C4SP3', 'Xch-7dv', 'Xc-6d', 'Xpc-4d', 'Xpc-4dv', 'Xpc-6dv', 'Xp-1dv', 'NdsCH', 'NsssCH', 'NddC', 'NssNH', 'NaaN', 'SssCH2', 'SdsCH', 'SdssC', 'SaasN', 'SsSH', 'ETA_shape_y', 'AETA_beta_ns', 'AETA_beta_ns_d', 'AETA_eta_FL', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_4', 'ETA_dEpsilon_B', 'AETA_dBeta', 'fMF', 'IC0', 'TIC0', 'SIC0', 'MIC0', 'PEOE_VSA3', 'PEOE_VSA7', 'PEOE_VSA11', 'PEOE_VSA12', 'SMR_VSA3', 'SMR_VSA9', 'SlogP_VSA3', 'SlogP_VSA8', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'AMID_h', 'AMID_N', 'piPC1', 'piPC7', 'piPC8', 'TpiPC10', 'naHRing', 'n6ARing', 'n7ARing', 'nG12ARing', 'nAHRing', 'n6AHRing', 'n10FHRing', 'n10FaHRing', 'n7FARing', 'nRot', 'GGI5', 'GGI6', 'GGI9', 'JGI7', 'JGI8', 'MWC02', 'SRW10', 'AMW']

train_x = train_x[selected_features]
test_x = test_x[selected_features]        

cv = IterativeStratification(n_splits=3, order=2)
scorer = make_scorer(roc_auc_score, average = 'macro')

parameters = {'estimator__C':[0.01,0.1,1,10, 10, 100, 1000],
    'estimator__kernel' : ["linear","poly","rbf","sigmoid"],
    'estimator__degree' : [1,2,3,4,5,6,7],
    'estimator__gamma' : [0.0001, 0.001, 0.01,1,10,500]}

clf = OneVsRestClassifier(svm.SVC(random_state=0))

random_search = BayesSearchCV(estimator = clf, search_spaces = parameters, n_iter=3, scoring=scorer, n_jobs = -1, random_state = 0, cv=cv, refit = True, verbose = 1)
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

