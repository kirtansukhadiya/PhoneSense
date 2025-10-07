import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('data/raw/Mobile_adicted.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

categorical_column_index = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_column_index)], remainder= 'passthrough') 
X = np.array(ct.fit_transform(X))

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:19])
X[:, 1:19] = imputer.transform(X[:, 1:19])