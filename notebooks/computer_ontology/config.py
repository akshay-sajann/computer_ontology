# config.py 

from pathlib import Path

computer_dir = Path('../data/computer')
expert_dir = Path('../data/expert')

raw_path = computer_dir / 'dataset/raw/alldesc_dataset.csv'

computer_dataset_path = computer_dir / 'dataset/computer_dataset.csv'

expert_dataset_path = expert_dir / 'dataset/expert_dataset.csv'

canon_path = computer_dir / 'canonicalization_and_umbrella-terms/labels-canonicalization.xlsx'

computer_path = computer_dir / 'canonicalization_and_umbrella-terms/computer_derived_ontology_11.xlsx'

lemma_path = expert_dir / 'lemmatization_and_umbrella-terms/computer_derived_ontology_11.xlsx'

expert_path = expert_dir / 'canonicalization_and_umbrella-terms/computer_derived_ontology_11.xlsx'
