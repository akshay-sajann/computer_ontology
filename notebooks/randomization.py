# importing packages
import torch
import pandas as pd
from computer_ontology.custom_funcs import *
from sklearn.ensemble import RandomForestClassifier
from computer_ontology.featurizer import get_mordred
from sklearn.preprocessing import MultiLabelBinarizer
from computer_ontology.config import raw_path, computer_dataset_path, computer_path
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, MultilabelRecall

# dictionary to store umbrella scores
umbrella_scores = {}
umbrella_scores['f1_macro'] = []
umbrella_scores['auroc_macro'] = []
umbrella_scores['precision_macro'] = []
umbrella_scores['recall_macro'] = []
umbrella_scores['f1_micro'] = []
umbrella_scores['auroc_micro'] = []
umbrella_scores['precision_micro'] = []
umbrella_scores['recall_micro'] = []
umbrella_scores['f1_weighted'] = []
umbrella_scores['auroc_weighted'] = []
umbrella_scores['precision_weighted'] = []
umbrella_scores['recall_weighted'] = []

# fetching the datasets
dataset = pd.read_csv(raw_path)
umbrella = pd.read_csv(computer_dataset_path)

dataset.set_index('CID', inplace=True)
umbrella.set_index('CID', inplace=True)

dataset['Descriptors'] = dataset['Descriptors'].apply(format_list)
dataset = dataset[['IsomericSMILES', 'Descriptors']]

X, y = x_y_split(dataset)

mordred = get_mordred(X)

X, y = branch_split(mordred, umbrella)

train_x, train_y, test_x, test_y = iterative_train_test_split(X, y, 0.2)
clf = RandomForestClassifier(random_state=0)
clf.fit(train_x, train_y)

y_hat = clf.predict(test_x)

f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

f1score_micro = MultilabelF1Score(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
auroc_micro = MultilabelAUROC(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
precision_micro = MultilabelPrecision(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
recall_micro = MultilabelRecall(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

f1score_weighted = MultilabelF1Score(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
auroc_weighted = MultilabelAUROC(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
precision_weighted = MultilabelPrecision(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
recall_weighted = MultilabelRecall(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

umbrella_scores['f1_macro'].append(f1score_macro)
umbrella_scores['auroc_macro'].append(auroc_macro)
umbrella_scores['precision_macro'].append(precision_macro)
umbrella_scores['recall_macro'].append(recall_macro)
umbrella_scores['f1_micro'].append(f1score_micro)
umbrella_scores['auroc_micro'].append(auroc_micro)
umbrella_scores['precision_micro'].append(precision_micro)
umbrella_scores['recall_micro'].append(recall_micro)
umbrella_scores['f1_weighted'].append(f1score_weighted)
umbrella_scores['auroc_weighted'].append(auroc_weighted)
umbrella_scores['precision_weighted'].append(precision_weighted)
umbrella_scores['recall_weighted'].append(recall_weighted)

rand_scores = {}
rand_scores['f1_macro'] = []
rand_scores['auroc_macro'] = []
rand_scores['precision_macro'] = []
rand_scores['recall_macro'] = []
rand_scores['f1_micro'] = []
rand_scores['auroc_micro'] = []
rand_scores['precision_micro'] = []
rand_scores['recall_micro'] = []
rand_scores['f1_weighted'] = []
rand_scores['auroc_weighted'] = []
rand_scores['precision_weighted'] = []
rand_scores['recall_weighted'] = []

labels_df = pd.read_excel(computer_path)
labels_to_remove = labels_df[labels_df['Umbrella Terms'].isna()]['Original Descriptors']
labels_df = labels_df.dropna()

trials = 2
for count in range(trials):
    replace = labels_df.copy()
    replace['Umbrella Terms'] = np.random.permutation(replace['Umbrella Terms'])
    replace = replace.values.tolist()
    # Changing to Umbrella terms and normalizing it once again
    rand = dataset.copy()
    rand['Descriptors'] = rand['Descriptors'].apply(lambda x: ';'.join([item for item in x.split(';') if item not in labels_to_remove.index]))
    rand = rand[rand['Descriptors'] != '']
    rand['Descriptors'] = rand['Descriptors'].apply(check_and_replace)
    rand['Descriptors'] = rand['Descriptors'].apply(make_unique)
    rand['Descriptors'] = rand['Descriptors'].dropna()
    rand = rand[rand['Descriptors'] != '']
    #rand['Descriptor Count'] = rand['Descriptors'].apply(lambda x: len(x.split(';')))
    rand['Descriptors'] = rand['Descriptors'].apply(lambda x: x.split(';'))
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(rand['Descriptors'])
    rand = rand.join(pd.DataFrame.sparse.from_spmatrix(mlb.transform(rand['Descriptors']), index=rand.index, columns=mlb.classes_))
    X, y = branch_split(mordred, rand)
    train_x, train_y, test_x, test_y = iterative_train_test_split(X, y, 0.2)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_x, np.asarray(train_y, dtype=np.float64))

    y_hat = clf.predict(test_x)

    f1score_macro = MultilabelF1Score(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
    auroc_macro = MultilabelAUROC(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    precision_macro = MultilabelPrecision(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    recall_macro = MultilabelRecall(num_labels=len(train_y.columns), average="macro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

    f1score_micro = MultilabelF1Score(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
    auroc_micro = MultilabelAUROC(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    precision_micro = MultilabelPrecision(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    recall_micro = MultilabelRecall(num_labels=len(train_y.columns), average="micro")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

    f1score_weighted = MultilabelF1Score(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))
    auroc_weighted = MultilabelAUROC(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    precision_weighted = MultilabelPrecision(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))
    recall_weighted = MultilabelRecall(num_labels=len(train_y.columns), average="weighted")(torch.tensor(y_hat, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.long))

    rand_scores['f1_macro'].append(float(f1score_macro))
    rand_scores['auroc_macro'].append(float(auroc_macro))
    rand_scores['precision_macro'].append(float(precision_macro))
    rand_scores['recall_macro'].append(float(recall_macro))
    rand_scores['f1_micro'].append(float(f1score_micro))
    rand_scores['auroc_micro'].append(float(auroc_micro))
    rand_scores['precision_micro'].append(float(precision_micro))
    rand_scores['recall_micro'].append(float(recall_micro))
    rand_scores['f1_weighted'].append(float(f1score_weighted))
    rand_scores['auroc_weighted'].append(float(auroc_weighted))
    rand_scores['precision_weighted'].append(float(precision_weighted))
    rand_scores['recall_weighted'].append(float(recall_weighted))

for key in umbrella_scores:
   z_statistic, p_value = stests.ztest(rand_scores[key], value=umbrella_scores[key])
   print(key)
   print("Umbrella score", umbrella_scores[key])
   print("Random scores mean", statistics.mean(rand_scores[key]))
   print("Random scores stdev", statistics.stdev(rand_scores[key]))
   print("Z-statistic:", z_statistic)
   print("P-value:", p_value)