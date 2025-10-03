import time
import shap
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
import xgboost as xgb
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from computer_ontology.featurizer import get_mordred
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall

alldesc_dict = {"train": alldesc_train_path, "test": alldesc_test_path, "model": {"logreg": alldesc_logreg_path, "rf": alldesc_rf_path, "xg": alldesc_xg_path}}
comp_dict = {"train": computer_train_path, "test": computer_test_path, "model": {"logreg": comp_logreg_path, "rf": comp_rf_path}}
exp_dict = {"train": expert_train_path, "test": expert_test_path, "model": {"logreg": exp_logreg_path, "rf": exp_rf_path}}


path_dict ={"alldesc": alldesc_dict, "comp": comp_dict, "exp": exp_dict}

models= {
    "logreg": OneVsRestClassifier(LogisticRegression(**logreg_params)),
    "rf": RandomForestClassifier(**rf_params, n_jobs=-1),
    "xg": xgb.XGBClassifier(**xg_params, n_jobs=-1)
}

for dataset in path_dict.keys():
    train = pd.read_csv(path_dict[dataset]["train"])
    test = pd.read_csv(path_dict[dataset]["test"])

    train = train.set_index('CID')
    test = test.set_index('CID')

    train_x, train_y = x_y_split(train)
    test_x, test_y = x_y_split(test)

    train_x = get_mordred(train_x)
    test_x = get_mordred(test_x)

    train_x = train_x[selected_features]
    test_x = test_x[selected_features]

    train_x.shape

    scaler = MinMaxScaler()

    scaler.fit(train_x)

    train_x = pd.DataFrame(data=scaler.transform(train_x), columns=train_x.columns)
    test_x = pd.DataFrame(data=scaler.transform(test_x), columns=test_x.columns)

    for model in path_dict[dataset]["model"].keys():
        print(dataset)
        print(model)
        
        clf = models[model]

        clf.fit(train_x, train_y)

        y_hat = clf.predict(test_x)

        f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
        auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
        precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
        recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

        print(f"F1 Score Macro: {f1score_macro}")
        print(f"AUROC Macro: {auroc_macro}")
        print(f"Precision Macro: {precision_macro}")
        print(f"Recall Macro: {recall_macro}")

        # Pickle the model
        with open(path_dict[dataset]["model"][model], "wb") as f:
            pickle.dump(clf, f)
            print('model saved successfully')
