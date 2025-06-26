# importing packages
import time
import shap
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from xgboost import XGBClassifier
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
from sklearn.feature_selection import VarianceThreshold
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall

# fetching the datasets
train = pd.read_csv(expert_train_path_2025)
test = pd.read_csv(expert_test_path_2025)

train = train.set_index('CID')
test = test.set_index('CID')

train_x, train_y = x_y_split(train)
test_x, test_y = x_y_split(test)

train_x = get_mordred(train_x)
test_x = get_mordred(test_x)

train_x = train_x[selected_features_2025]
test_x = test_x[selected_features_2025]

# scaling 

scaler = MinMaxScaler()

scaler.fit(train_x)

train_x = pd.DataFrame(data=scaler.transform(train_x), columns=train_x.columns)
test_x = pd.DataFrame(data=scaler.transform(test_x), columns=test_x.columns)

# Load the model
with open(exp_xg_path_2025, "rb") as f:
    clf = pickle.load(f)
    print("Model loaded successfully")

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(test_x)

# Pickle the SHAP explainer
with open(exp_explainer_path_2025, "wb") as f:
    pickle.dump(explainer, f)
    print('explainer saved successfully')

# Pickle the SHAP values
with open(exp_shap_path_2025, "wb") as f:
    pickle.dump(shap_values, f)
    print('values saved successfully to file')
