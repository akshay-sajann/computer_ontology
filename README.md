# Hierarchies of Smell: Predicting Odor from Molecular Structure using Semantic Taxonomies and machine learning

Repository for the follorwing [arxiv paper](https://arxiv.org/abs/2508.09217)

Note: the file names are dated (for now), old names are: Merged Molecular Dataset (MMD) => alldesc dataset, Data derived Taxonomy (DT) => computer ontology, Expert derived Taxonomy (ET) => expert ontology 

## Quickstart

### Github installation 

'''
git clone https://github.com/akshay-sajann/computer_ontology
cd computer_ontology
'''

### activating environment

'''
conda env create -f environment.yml
conda activate ontology
'''

## Details

### Within data

Note: 2025 folder contains the latest versions 

### Within notebooks
dataset_creation.ipynb --> compiling all the data for Merged Molecular Dataset (MMD)
data_split.ipynb --> making the data splits for MMD and Expert derived Taxonomy (ET) imposed dataset
taxonomy_derivation.ipynb --> obtaining Data derived taxonomy (DT) imposed dataset from MMD train split

