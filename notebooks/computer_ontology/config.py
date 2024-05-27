# config.py 

from pathlib import Path

computer_dir = Path('../data/computer')
expert_dir = Path('../data/expert')
result_path = Path('../results')

lemma_raw_path = expert_dir / 'dataset/raw/lemma_alldesc_dataset.csv'

canon_raw_path = computer_dir / 'dataset/raw/canon_alldesc_dataset.csv'

computer_dataset_path = computer_dir / 'dataset/computer_dataset_11.csv'

expert_dataset_path = expert_dir / 'dataset/expert_dataset.csv'

canon_path = computer_dir / 'canonicalization_and_umbrella-terms/labels_canonicalization.xlsx'

computer_path = computer_dir / 'canonicalization_and_umbrella-terms/computer_derived_ontology_11.xlsx'

lemma_path = expert_dir / 'lemmatization_and_umbrella-terms/labels-to-be-lemmatized.txt'

expert_path = expert_dir / 'lemmatization_and_umbrella-terms/expert_derived_ontology.csv'

comp_result_path = result_path / 'comp_scores.pkl'

exp_result_path = result_path / 'exp_scores.pkl'
