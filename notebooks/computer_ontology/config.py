# config.py 
from pathlib import Path
from collections import OrderedDict

# Get the directory of the current script (config.py)
script_dir = Path(__file__).parent

# Resolve paths relative to the script directory
data = (script_dir / '../../data/2024')
data_2025 = (script_dir / '../../data/2025')
models =  (script_dir / '../../models')
result = (script_dir / '../../results')

# Entire dataset

alldesc_dataset_path = data_2025 / 'alldesc_dataset.csv'
expert_dataset_path = data_2025 / 'expert_dataset.csv'

# Train paths

alldesc_train_path = data / 'train/alldesc_train.csv'
lemma_train_path = data / 'train/lemma_train.csv'
computer_train_path = data / 'train/computer_train.csv'
expert_train_path = data / 'train/expert_train.csv'

alldesc_train_path_2025 = data_2025 / 'train/alldesc_train_2025.csv'
computer_train_path_2025 = data_2025 / 'train/computer_train_2025.csv'
expert_train_path_2025 = data_2025 / 'train/expert_train_2025.csv'

alldesc_train_path_wo_leff = data_2025 / 'train/alldesc_train_wo_leff.csv'
computer_train_path_wo_leff = data_2025 / 'train/computer_train_wo_leff.csv'
expert_train_path_wo_leff = data_2025 / 'train/expert_train_wo_leff.csv'

alldesc_train_path_wo_arc = data_2025 / 'train/alldesc_train_wo_arc.csv'
computer_train_path_wo_arc = data_2025 / 'train/computer_train_wo_arc.csv'
expert_train_path_wo_arc = data_2025 / 'train/expert_train_wo_arc.csv'

# Test paths

alldesc_test_path = data / 'test/alldesc_test.csv'
lemma_test_path = data / 'test/lemma_test.csv'
computer_test_path = data / 'test/computer_test.csv'
expert_test_path = data / 'test/expert_test.csv'

alldesc_test_path_2025 = data_2025 / 'test/alldesc_test_2025.csv'
computer_test_path_2025 = data_2025 / 'test/computer_test_2025.csv'
expert_test_path_2025 = data_2025 / 'test/expert_test_2025.csv'

alldesc_test_path_wo_leff = data_2025 / 'test/alldesc_test_wo_leff.csv'
computer_test_path_wo_leff = data_2025 / 'test/computer_test_wo_leff.csv'
expert_test_path_wo_leff = data_2025 / 'test/expert_test_wo_leff.csv'

alldesc_test_path_wo_arc = data_2025 / 'test/alldesc_test_wo_arc.csv'
computer_test_path_wo_arc = data_2025 / 'test/computer_test_wo_arc.csv'
expert_test_path_wo_arc = data_2025 / 'test/expert_test_wo_arc.csv'

# Taxonomy paths

computer_tax_path = data_2025 / 'ontology/computer_ontology_2025.xlsx'
expert_tax_path = data_2025 / 'ontology/expert_ontology_2025.xlsx'

canon_path = data_2025 / 'ontology/labels_canonicalization.xlsx'

computer_tax_path_wo_leff = data_2025 / 'ontology/computer_ontology_wo_leff.xlsx'

computer_tax_path_wo_arc = data_2025 / 'ontology/computer_ontology_wo_arc.xlsx'

# models

alldesc_xg_path = models / 'alldesc_xg_model.pkl'
alldesc_rf_path = models / 'alldesc_rf_model.pkl'
alldesc_logreg_path = models / 'alldesc_logreg_model.pkl'

comp_xg_path = models / 'comp_xg_model.pkl'
comp_rf_path = models / 'comp_rf_model.pkl'
comp_logreg_path = models / 'comp_logreg_model.pkl'

exp_xg_path = models / 'exp_xg_model.pkl'
exp_rf_path = models / 'exp_rf_model.pkl'
exp_logreg_path = models / 'exp_logreg_model.pkl'

comp_xg_path_2025 = models / 'comp_xg_model_2025.pkl'

exp_xg_path_2025 = models / 'exp_xg_model_2025.pkl'

comp_rf_path_2025 = models / 'comp_rf_model_2025.pkl'

exp_rf_path_2025 = models / 'exp_rf_model_2025.pkl'

comp_logreg_path_2025 = models / 'comp_logreg_model_2025.pkl'

exp_logreg_path_2025 = models / 'exp_logreg_model_2025.pkl'

# Randomization

comp_rand_xg_path = result/ 'comp_rand_xg.pkl'
comp_rand_rf_path = result/ 'comp_rand_rf.pkl'
comp_rand_logreg_path = result/ 'comp_rand_logreg.pkl'

comp_rand_xg_path_2025 = result/ 'comp_rand_xg_2025.pkl'
exp_rand_xg_path_2025 = result/ 'exp_rand_xg_2025.pkl'

comp_class_dist_path = result/ 'comp_class_dist.pkl'
exp_class_dist_path = result/ 'exp_class_dist.pkl'

# SHAP

comp_explainer_path = result / 'comp_shap_explainer.pkl'
comp_shap_path = result / 'comp_shap_values.pkl'

exp_explainer_path = result / 'exp_shap_explainer.pkl'
exp_shap_path = result / 'exp_shap_values.pkl'

comp_explainer_path_2025 = result / 'comp_shap_explainer_2025.pkl'
comp_shap_path_2025 = result / 'comp_shap_values_2025.pkl'

exp_explainer_path_2025 = result / 'exp_shap_explainer_2025.pkl'
exp_shap_path_2025 = result / 'exp_shap_values_2025.pkl'

# Features

selected_features = ['nS', 'BCUTc-1h', 'BCUTc-1l', 'BCUTdv-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTare-1l', 'BalabanJ', 'SpMAD_Dzp', 'RNCG', 'RPCG', 'C2SP1', 'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C4SP3', 'Xch-7dv', 'Xc-6d', 'Xpc-4d', 'Xpc-4dv', 'Xpc-6dv', 'Xp-1dv', 'NdsCH', 'NsssCH', 'NddC', 'NssNH', 'NaaN', 'SssCH2', 'SdsCH', 'SdssC', 'SaasN', 'SsSH', 'ETA_shape_y', 'AETA_beta_ns', 'AETA_beta_ns_d', 'AETA_eta_FL', 'ETA_dAlpha_A', 'ETA_dAlpha_B', 'ETA_epsilon_4', 'ETA_dEpsilon_B', 'AETA_dBeta', 'fMF', 'IC0', 'TIC0', 'SIC0', 'MIC0', 'PEOE_VSA3', 'PEOE_VSA7', 'PEOE_VSA11', 'PEOE_VSA12', 'SMR_VSA3', 'SMR_VSA9', 'SlogP_VSA3', 'SlogP_VSA8', 'SlogP_VSA10', 'SlogP_VSA11', 'EState_VSA5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'AMID_h', 'AMID_N', 'piPC1', 'piPC7', 'piPC8', 'TpiPC10', 'naHRing', 'n6ARing', 'n7ARing', 'nG12ARing', 'nAHRing', 'n6AHRing', 'n10FHRing', 'n10FaHRing', 'n7FARing', 'nRot', 'GGI5', 'GGI6', 'GGI9', 'JGI7', 'JGI8', 'MWC02', 'SRW10', 'AMW']


selected_features_2025 = ['BCUTc-1h', 'BCUTc-1l', 'BCUTZ-1h', 'BCUTv-1l', 'BCUTare-1l', 'RPCG', 'Xpc-4dv', 'Xp-1dv', 'Mare', 'ETA_shape_y', 'ETA_dAlpha_B', 'ETA_epsilon_4', 'SIC0', 'MIC1', 'SlogP_VSA5', 'EState_VSA5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'nRot', 'GGI7', 'SRW04', 'SRW10']

# Hyperparameter tuning

xg_params = OrderedDict([('random_state', 0), ('colsample_bynode', 1.0), ('learning_rate', 0.6), ('max_depth', 8), ('min_child_weight', 1), ('n_estimators', 5000), ('reg_lambda', 0.01), ('subsample', 1.0), ('tree_method', 'hist')])

logreg_params = OrderedDict([('random_state', 0), ('C', 10000.0), ('max_iter', 2500), ('penalty', 'l2'), ('solver', 'liblinear')])

rf_params = OrderedDict([('random_state', 0), ('bootstrap', False), ('max_depth', None), ('max_features', 'sqrt'), ('min_samples_leaf', 1), ('min_samples_split', 2), ('n_estimators', 2000)])

# - - - - 
xg_params_2025 = OrderedDict([('random_state', 0), ('colsample_bynode', 0.9), ('learning_rate', 0.4), ('max_depth', 3), ('min_child_weight', 1), ('n_estimators', 2000), ('reg_lambda', 5), ('subsample', 0.9), ('tree_method', 'hist')])

rf_params_2025 = OrderedDict([('random_state', 0), ('bootstrap', False), ('max_depth', 90), ('max_features', 'log2'), ('min_samples_leaf', 1), ('min_samples_split', 2), ('n_estimators', 1800)])

logreg_params_2025 = OrderedDict([('random_state', 0), ('C', 10000.0), ('max_iter', 2500), ('penalty', 'l2'), ('solver', 'lbfgs')])


