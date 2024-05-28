# Featurizer.py

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from mordred import Calculator, descriptors
from sklearn.preprocessing import MinMaxScaler

def get_morgan(X):
  """
  This function takes in a dataframe and returns
  a featurized dataframe with morgan fingerprints.

  :param df: A molecular dataset for odor prediction with SMILES strings
  :type df: pandas Dataframe
  :return: A featurized dataframe.
  :rtype: pandas dataframes
  """
  '''
  df['molecule'] = df['IsomericSMILES'].apply(lambda x: Chem.MolFromSmiles(x))
  df['MorganFP'] = df['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=2048,useFeatures=True,useChirality=True))

  df_list = []

  for i in range(df.shape[0]):
    array = np.array(df['MorganFP'][i])
    df_i = pd.DataFrame(array)
    df_i = df_i.T
    df_list.append(df_i)
  morganfp = pd.concat(df_list, ignore_index=True)

  return morganfp
  '''
  X['molecule'] = X['IsomericSMILES'].apply(lambda x: Chem.MolFromSmiles(x))
  X['MorganFP'] = X['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=2048,useFeatures=True))
  X['descrip'] = X['molecule'].apply(lambda x: list(Descriptors.CalcMolDescriptors(x).values()))

  df = {}
  df_list = []
  cols = ['MorganFP', 'descrip']
  for j in range(len(cols)):
    for i in range(X.shape[0]):
      array = np.array(X[cols[j]][i])
      df_i = pd.DataFrame(array)
      df_i = df_i.T
      df_list.append(df_i)
    df[cols[j]] = pd.concat(df_list, ignore_index=True)
    df_list = []

  x_scaler = MinMaxScaler()
  descrip = x_scaler.fit_transform(df['descrip'])
  morganfp = pd.DataFrame(np.column_stack((df['MorganFP'], np.asarray(descrip))))
  return morganfp

def get_mordred(data):
  """
  This function takes in a dataframe and returns
  a Mordred descriptors.

  :param data: A molecular dataset for odor prediction with SMILES strings
  :type data: pandas Dataframe
  :param y: The labels y of the data variable
  :type y: pandas Dataframe
  :return df: A featurized dataframe.
  :rtype df: pandas dataframes
  :return y: Labels
  :rtype y: pandas dataframes
  """
  filtered_descriptors = [descriptor for descriptor in descriptors.all if descriptor is not descriptors.Autocorrelation]
  calc = Calculator(filtered_descriptors, ignore_3D=False)
  mols = [Chem.MolFromSmiles(smi) for smi in data]

  # pandas df
  df = calc.pandas(mols)
  df.index = data.index
  for column in df.columns:
      df[column] = pd.to_numeric(df[column], errors='coerce')
  missing_values = df.isna().sum()
  df = df.loc[:, missing_values <= 0]
  return df
