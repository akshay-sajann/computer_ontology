# custom_funcs.py
import pyrfume
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from mordred import Calculator, descriptors
from skmultilearn.model_selection import IterativeStratification


def leffingwell_reverse_one_hot(row):
    """
    Takes a row of the Leffingwell dataset
    and reverses one-hot-encoding.

    :param row: A given row of the Leffingwell dataset.
    :type row: pandas Dataframe row
    :return: A list of classes/labels for each row.
    :rtype: List
    """
    labels = [col for col in leffingwell.columns if row[col] == 1]
    return ';'.join(labels)

def dravnieks_top_n_columns(row, n):
    """
    This is for the Dravnieks dataset
    since its format is unique. It
    selects the labels with the highest
    values and returns them in the standard
    format used for the other datasets.

    :param row: A given row of the Dravnieks dataset.
    :type row: pandas Dataframe row
    :param n: The number of labels to consider.
    :type n: int
    :return: top n labels.
    :rtype: string
    """
    sorted_columns = row.sort_values(ascending=False).index
    top_n = sorted_columns[:n]
    return ';'.join(top_n)

def get_unique(df):
    """
    This function takes in an odor dataset as a
    dataframe and returns a dataframe containing
    all the unique labels of the input dataframe.

    :requirements: labels should be called 'Descriptors'.

    :param df: A multilabel dataframe with labels separated by ';'
    :type df: pandas Dataframe
    :return: A dataframe containing the unique descriptors.
    :rtype: pandas Dataframe
    """
    all_descriptors = []

    for des in df['Descriptors']:
        all_descriptors.extend(des.split(';'))

    unique_descriptors = list(set(all_descriptors))
    unique_descriptors.sort()

    df = pd.DataFrame(unique_descriptors)
    return df

def get_dataset(name):
    """
    This function takes in a string which is the
    name of the dataset and returns the fetched
    dataset.

    :param name: Name of the dataset according to Pyrfume
    :type name: string
    :return: The dataset called
    :rtype: Dataframe
    """
    # Load molecular and stimulus data
    mols =  pyrfume.load_data(f'{name}/molecules.csv')["IsomericSMILES"]
    stim =  pyrfume.load_data(f'{name}/stimuli.csv')

    # Deal with exceptions for behavior data
    try:
      behav =  pyrfume.load_data(f'{name}/behavior.csv')
    except:
      try:
        behav =  pyrfume.load_data(f'{name}/behavior_1_sparse.csv')
      except:
        behav =  pyrfume.load_data(f'{name}/behavior_1.csv')

    if name == 'ifra_2019':
      behav['Descriptor 1'] = behav[['Descriptor 1', 'Descriptor 2', 'Descriptor 3']].astype(str).apply(';'.join, axis=1)

      behav = behav['Descriptor 1']

    labels = pd.merge(stim, behav, on='Stimulus')

    # Deal with exceptions during Merging
    try:
      df = pd.merge(mols, labels, on='CID')
    except:
      labels.rename(columns={'new_CID': 'CID'}, inplace=True)
      df = pd.merge(mols, labels, on='CID')

    return df

def check_and_replace(description):
    """
    Iterates through a given ";" separated strings
    and replaces them with the mapping assigned by
    any list labelled "mapping".

    :param description: Text separated by ';'
    :type name: string
    :return: Text replaced according to the mapping
    :rtype: string
    """
    descriptors = description.split(';')
    new_descriptors = []

    for descriptor in descriptors:
        for row in replace:
            if descriptor == row[0]:
                new_descriptors.append(row[1])

    return ';'.join(new_descriptors)

def make_unique(labels):
   """
   Takes a text separated by ";" and makes them
   unique.

   :param description: Text separated by ';'.
   :type name: string
   :return: words within text made unique.
   :rtype: string
    """
   return ';'.join(list(set(labels.split(';'))))

def count_words(label_str):
    """
    Takes a ";" separated strings and counts them

   :param description: Text separated by ';'.
   :type name: string
   :return: number of words separated by ";"
   :rtype: int
    """
    return len(label_str.split(';'))

def format_list(input_str):
    """
    Just changing the format of a column
    to make it workable
    """
    cleaned_str = input_str.strip("[]").replace("'", "")

    items = cleaned_str.split(',')
    items = [item.strip() for item in items]

    formatted_str = ';'.join(items)

    return formatted_str

def x_y_split(df):
  """
  Splies the oncoming dataset to X and
  y for classification.

  :param df: A molecular dataset for odor prediction
  :type df: pandas Dataframe
  :return: A list of classes/labels for each row.
  :rtype: pandas dataframes
  """
  try:
    x = df[['IsomericSMILES', 'CID']].copy()
  except:
     x = df['IsomericSMILES'].copy()
  try:
    y = df.drop(['IsomericSMILES', 'Descriptors', 'CID', 'Descriptor Count'], axis=1).copy()
    return x,y
  except:
    try:
      y = df.drop(['IsomericSMILES', 'Descriptors', 'CID'], axis=1).copy()
      return x,y
    except:
      y = df.drop(['IsomericSMILES', 'Descriptors'], axis=1).copy()
      return x,y
  
def iterative_train_test_split(X, y, test_size):
  """
  Function doing a train-test split
  using the second order iterative
  stratification method.

  :param df: X and y dataframes for a multilabel machine learning task
  :type df: pandas Dataframes
  :return: train-test split dataframes
  :rtype: pandas dataframes
  """
  stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
  train_indexes, test_indexes = next(stratifier.split(X, y))

  X_train, y_train = X.iloc[train_indexes], y.iloc[train_indexes]
  X_test, y_test = X.iloc[test_indexes], y.iloc[test_indexes]

  return X_train, y_train, X_test, y_test

def branch_split(template, df):
  ignore, y = x_y_split(df)
  common_indices = template.index.intersection(y.index)
  X = template.loc[common_indices].copy()
  X = template.sort_index(axis=0)
  y = y.sort_index(axis=0)
  return X, y
