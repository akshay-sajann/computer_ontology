# importing packages
import pandas as pd
from computer_ontology.featurizer import get_morgan
from computer_ontology.config import raw_path, computer_dataset_path
from computer_ontology.custom_funcs import *

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

