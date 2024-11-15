# config.py 

from pathlib import Path

data = Path('../data')
result = Path('../results')

# train 

alldesc_train_path = data / "train/alldesc_train.csv"
computer_train_path = data / "train/computer_train.csv"
expert_train_path = data / "train/expert_train.csv"
lemma_train_path = data / "train/lemma_train.csv"

# test 

alldesc_test_path = data / "test/alldesc_test.csv"
computer_test_path = data / "test/computer_test.csv"
expert_test_path = data / "test/expert_test.csv"
lemma_test_path = data / "test/lemma_test.csv"

# ontology 

computer_path = data / "ontology/computer/computer_ontology_Aug.xlsx" 
canon_path = data / "ontology/computer/labels_canonicalization.xlsx" 

expert_path = data / "ontology/expert/expert_derived_ontology.csv"
lemma_path =  data / "ontology/expert/labels-to-be-lemmatized.txt"

# randomization

comp_rand_xg_path = result / "comp_rand_xg.pkl"
comp_rand_rf_path = result / "comp_rand_rf.pkl"
comp_rand_logreg_path = result / "comp_rand_logreg.pkl"

exp_rand_xg_path = result / "exp_rand_xg.pkl"
exp_rand_rf_path = result / "exp_rand_rf.pkl"
exp_rand_logreg_path = result / "exp_rand_logreg.pkl"

# SHAP



# Features


# Hyperparameters


