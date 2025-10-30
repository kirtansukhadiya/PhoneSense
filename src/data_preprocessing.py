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
dataset_target = pd.DataFrame(y, columns=['addiction_level'])
dataset_target.to_csv('data/processed/mobile_addiction_target.csv', index=False)
X9 = le.fit_transform(X[:, 9])

categorical_column_index = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_column_index)], remainder= 'passthrough') 
X = np.array(ct.fit_transform(X))

feature_names = ct.get_feature_names_out()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:18])
X[:, 1:18] = imputer.transform(X[:, 1:18])



#saving into new csv file
imputed_df = pd.DataFrame(X, columns=feature_names)
imputed_df['X9'] = X9
imputed_df.columns = imputed_df.columns.str.replace(r'^.*?__', '', regex=True)
imputed_df.to_csv('data/processed/mobile_addiction_cleaned.csv', index=False)
column_needed = ['x0_Female', 'x2_No', 'x3_No', 'x4_No', 'x5_No', 'x6_No', 'x7_No', 'x8_No', 'X9', 'x10_No', 'x11_No', 'x12_No', 'x13_No', 'x14_No', 'x15_No', 'x16_No', 'x18_No','x1', 'x17']
Final_imputed_df = imputed_df[column_needed].rename(columns={
    'x0_Female': 'Gender',
    'x2_No': 'Use Phone for Class Notes',
    'x3_No' : 'Buy/Access Books from Phone',
    'x4_No' : 'Phone\'s Battery Lasts a Day',
    'x5_No' : 'Run for Charger When Battery Dies',
    'x6_No' : 'Worry About Losing Cell Phone',
    'x7_No' : 'Take Phone to Bathroom',
    'x8_No' : 'Use Phone in Social Gatherings',
    'X9' : 'Check Phone Without Notification',
    'x10_No' : 'Check Phone Before Sleep/After Waking Up',
    'x11_No' : 'Keep Phone Next to While Sleeping',
    'x12_No' : 'Check Emails/Calls/Texts During Class',
    'x13_No' : 'Rely on Phone in Awkward Situations',
    'x14_No' : 'On Phone While Watching TV/Eating',
    'x15_No' : 'Panic Attack if Phone Left Elsewhere',
    'x16_No' : 'Use Phone on Date',
    'x18_No' : 'Live a Day Without Phone',
    'x1': 'Age',
    'x17': 'Phone Use for Playing Games'
})
Final_imputed_df.to_csv('data/processed/mobile_addiction_cleaned_final.csv', index=False)