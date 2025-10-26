import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('data/raw/Mobile_adicted.csv')
X = dataset.iloc[:, 2:-2].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)

categorical_column_index = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_column_index)], remainder= 'passthrough') 
X = np.array(ct.fit_transform(X))

feature_names = ct.get_feature_names_out()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:18])
X[:, 1:18] = imputer.transform(X[:, 1:18])

#saving into new csv file
imputed_df = pd.DataFrame(X, columns=feature_names)
imputed_df.columns = imputed_df.columns.str.replace(r'^\w+__', '', regex=True)
imputed_df.columns = [
    old.split('_', 1)[1] if '_' in old else old
    for old in imputed_df.columns
]
imputed_df = imputed_df.rename(columns={
    'x1': 'Age',
    'x17': 'Phone Use for Playing Games'
})
imputed_df.to_csv('data/processed/mobile_addiction_cleaned.csv', index=False)