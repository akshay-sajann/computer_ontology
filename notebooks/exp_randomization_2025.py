# importing packages
from collections import OrderedDict
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
from computer_ontology.config import*
from computer_ontology.custom_funcs import *
from sklearn.preprocessing import MinMaxScaler
import statsmodels.stats.weightstats as stests
from sklearn.ensemble import RandomForestClassifier
from computer_ontology.featurizer import get_mordred
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall
from xgboost import XGBClassifier
from collections import defaultdict

# dictionary to store umbrella scores
umbrella_scores = {}
umbrella_scores['f1_macro'] = []
umbrella_scores['auroc_macro'] = []
umbrella_scores['precision_macro'] = []
umbrella_scores['recall_macro'] = []

# fetching the datasets
alldesc_train = pd.read_csv(alldesc_train_path_2025)
expert_train = pd.read_csv(expert_train_path_2025)

alldesc_test = pd.read_csv(alldesc_test_path_2025)
expert_test = pd.read_csv(expert_test_path_2025)

# Setting CIDs as indices
alldesc_train.set_index('CID', inplace=True)
expert_train.set_index('CID', inplace=True)
alldesc_test.set_index('CID', inplace=True)
expert_test.set_index('CID', inplace=True)


alldesc_train['Descriptors'] = alldesc_train['Descriptors'].apply(format_list)
alldesc_train = alldesc_train[['IsomericSMILES', 'Descriptors']]

alldesc_test['Descriptors'] = alldesc_test['Descriptors'].apply(format_list)
alldesc_test = alldesc_test[['IsomericSMILES', 'Descriptors']]

train_x, train_y = x_y_split(alldesc_train)
test_x, test_y = x_y_split(alldesc_test)

train_x = get_mordred(train_x)
test_x = get_mordred(test_x)

train_x = train_x[selected_features_2025]
test_x = test_x[selected_features_2025]

# scaling
scaler = MinMaxScaler()

scaler.fit(train_x)

train_x = pd.DataFrame(data=scaler.transform(train_x), columns=train_x.columns)
test_x = pd.DataFrame(data=scaler.transform(test_x), columns=test_x.columns)

X, train_y = x_y_split(expert_train)
X, test_y = x_y_split(expert_test)

clf = XGBClassifier(**xg_params_2025)
clf.fit(train_x, train_y)

y_hat = clf.predict(test_x)

f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

umbrella_scores['f1_macro'].append(f1score_macro)
umbrella_scores['auroc_macro'].append(auroc_macro)
umbrella_scores['precision_macro'].append(precision_macro)
umbrella_scores['recall_macro'].append(recall_macro)

rand_scores = {}
rand_scores['f1_macro'] = []
rand_scores['auroc_macro'] = []
rand_scores['precision_macro'] = []
rand_scores['recall_macro'] = []

labels_df = pd.read_excel(expert_tax_path, index_col=0)
labels_to_remove = labels_df[labels_df['Umbrella Terms'].isna()]['Original Descriptors']
labels_df = labels_df.dropna()

labels_df = labels_df.reset_index(drop=True)
labels_df["Umbrella Terms"] = labels_df["Umbrella Terms"].astype(str)

trials = 1000 

class_counts_dict = {}

for count in range(trials):
    replace = labels_df.copy()
    replace['Umbrella Terms'] = np.random.permutation(replace['Umbrella Terms'])
    replace = replace.values.tolist()
    
    # Changing to Umbrella terms and normalizing it once again
    rand_train = alldesc_train.copy()
    rand_test = alldesc_test.copy()
    
    # -------
    rand_train['Descriptors'] = rand_train['Descriptors'].apply(lambda x: ';'.join([item for item in x.split(';') if item not in labels_to_remove.index]))
    rand_train = rand_train[rand_train['Descriptors'] != '']
    rand_train['Descriptors'] = rand_train['Descriptors'].apply(check_and_replace, replace = replace)
    rand_train['Descriptors'] = rand_train['Descriptors'].apply(make_unique)
    rand_train['Descriptors'] = rand_train['Descriptors'].dropna()
    rand_train = rand_train[rand_train['Descriptors'] != '']
    #rand_train['Descriptor Count'] = rand_train['Descriptors'].apply(lambda x: len(x.split(';')))
    rand_train['Descriptors'] = rand_train['Descriptors'].apply(lambda x: x.split(';'))
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(rand_train['Descriptors'])
    rand_train = rand_train.join(pd.DataFrame.sparse.from_spmatrix(mlb.transform(rand_train['Descriptors']), index=rand_train.index, columns=mlb.classes_))
    X, train_y = x_y_split(rand_train)
    print("train y shape is ",train_y.shape) 

    sorted_class_counts = train_y.sum(axis=0).sort_values(ascending = False).reset_index(drop=True)
    for idx, count in sorted_class_counts.items():
        if idx not in class_counts_dict:
            class_counts_dict[idx] = []  # Initialize a list for the index if it doesn't exist
        class_counts_dict[idx].append(count)  # Append the count to the list
    # -------
    rand_test['Descriptors'] = rand_test['Descriptors'].apply(lambda x: ';'.join([item for item in x.split(';') if item not in labels_to_remove.index]))
    rand_test = rand_test[rand_test['Descriptors'] != '']
    rand_test['Descriptors'] = rand_test['Descriptors'].apply(check_and_replace, replace = replace)
    rand_test['Descriptors'] = rand_test['Descriptors'].apply(make_unique)
    rand_test['Descriptors'] = rand_test['Descriptors'].dropna()
    rand_test = rand_test[rand_test['Descriptors'] != '']
    #rand_test['Descriptor Count'] = rand_test['Descriptors'].apply(lambda x: len(x.split(';')))
    rand_test['Descriptors'] = rand_test['Descriptors'].apply(lambda x: x.split(';'))
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(rand_test['Descriptors'])
    rand_test = rand_test.join(pd.DataFrame.sparse.from_spmatrix(mlb.transform(rand_test['Descriptors']), index=rand_test.index, columns=mlb.classes_))
    X, test_y = x_y_split(rand_test)
    
    # --------
    clf =  XGBClassifier(**xg_params_2025)
    clf.fit(train_x, np.asarray(train_y, dtype=np.float64))

    y_hat = clf.predict(test_x)

    f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
    auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

    rand_scores['f1_macro'].append(float(f1score_macro))
    rand_scores['auroc_macro'].append(float(auroc_macro))
    rand_scores['precision_macro'].append(float(precision_macro))
    rand_scores['recall_macro'].append(float(recall_macro))


for key in umbrella_scores:
   z_statistic, p_value = stests.ztest(rand_scores[key], value=umbrella_scores[key])
   print(key)
   print("Umbrella score", umbrella_scores[key])
   print("Random scores mean", statistics.mean(rand_scores[key]))
   print("Random scores stdev", statistics.stdev(rand_scores[key]))
   print("Z-statistic:", z_statistic)
   print("P-value:", p_value)

with open(exp_rand_xg_path_2025, 'wb') as fp:
    pickle.dump(rand_scores, fp)
    print('scores saved successfully to file')

with open(exp_class_dist_path, 'wb') as fp:
    pickle.dump(class_counts_dict, fp)
    print('Distributions saved successfully to file')

