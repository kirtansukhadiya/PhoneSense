import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
dataset = pd.read_csv('data/raw/phone_addiction_dataset.csv')
dataset = dataset.drop(columns=['Name','Location','School_Grade','Parental_Control'])

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].astype(int)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='mean'), numeric_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

pd.DataFrame(X_train_scaled).to_csv('data/processed/mobile_addiction_X_Train.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('data/processed/mobile_addiction_X_Test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/processed/mobile_addiction_y_Train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/mobile_addiction_y_Test.csv', index=False)