# %% [markdown]
# importing dataset

# %%
import pandas as pd
X_train = pd.read_csv('data/processed/mobile_addiction_X_Train.csv')
X_test = pd.read_csv('data/processed/mobile_addiction_X_Test.csv')
y_train = pd.read_csv('data/processed/mobile_addiction_y_Train.csv')
y_test= pd.read_csv('data/processed/mobile_addiction_y_Test.csv')
y_train= y_train.values.ravel()
y_test= y_test.values.ravel()

# %% [markdown]
# Doing Oversampling and Training modal on logistic regression

# %%
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=1)),
    ('logreg', LogisticRegression(random_state=0, max_iter= 10000))
])
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# %% [markdown]
# Evaluating accuracy

# %%
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv_strategy = StratifiedKFold(n_splits=5)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", cm)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=cv_strategy)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))  


import joblib
joblib.dump(classifier, 'models/logistic_regression.pkl')