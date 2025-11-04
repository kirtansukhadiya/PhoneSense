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

categorical_column_index = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), categorical_column_index)], remainder= 'passthrough') 
X = np.array(ct.fit_transform(X))

feature_names = ct.get_feature_names_out()

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:18])
X[:, 1:18] = imputer.transform(X[:, 1:18])


#saving into new csv file
imputed_df = pd.DataFrame(X, columns=feature_names)
#imputed_df['X9'] = X9
imputed_df.columns = imputed_df.columns.str.replace(r'^.*?__', '', regex=True)
imputed_df.to_csv('data/processed/mobile_addiction_cleaned.csv', index=False)
column_needed = ['x0_Male', 'x2_Yes', 'x3_Yes', 'x4_Yes', 'x5_Yes', 'x6_Yes', 'x7_Yes', 'x8_Yes', 'x9_Often', 'x9_Rarely', 'x9_Sometimes', 'x10_Yes', 'x11_Yes', 'x12_Yes', 'x13_Yes', 'x14_Yes', 'x15_Yes', 'x16_Yes', 'x18_Yes', 'x1', 'x17']

Final_imputed_df = imputed_df[column_needed].rename(columns={
    'x0_Male': 'Gender',
    'x2_Yes': 'Use Phone for Class Notes',
    'x3_Yes': 'Buy/Access Books from Phone',
    'x4_Yes': 'Phone\'s Battery Lasts a Day',
    'x5_Yes': 'Run for Charger When Battery Dies',
    'x6_Yes': 'Worry About Losing Cell Phone',
    'x7_Yes': 'Take Phone to Bathroom',
    'x8_Yes': 'Use Phone in Social Gatherings',
    'x9_Often': 'Check Phone Without Notification (Often)',
    'x9_Rarely': 'Check Phone Without Notification (Rarely)',
    'x9_Sometimes': 'Check Phone Without Notification (Sometimes)',
    'x10_Yes': 'Check Phone Before Sleep/After Waking Up',
    'x11_Yes': 'Keep Phone Next to While Sleeping',
    'x12_Yes': 'Check Emails/Calls/Texts During Class',
    'x13_Yes': 'Rely on Phone in Awkward Situations',
    'x14_Yes': 'On Phone While Watching TV/Eating',
    'x15_Yes': 'Panic Attack if Phone Left Elsewhere',
    'x16_Yes': 'Use Phone on Date',
    'x18_Yes': 'Live a Day Without Phone',
    'x1': 'Age',
    'x17': 'Phone Use for Playing Games'
})

Final_imputed_df.to_csv('data/processed/mobile_addiction_cleaned_final.csv', index=False)
