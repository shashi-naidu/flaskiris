# -*- coding: utf-8 -*-
"""pima_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k8aBKFCGzMS3gx0Dy2uknMotV5GWrj03
"""

pip install joblib

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

df=pd.read_csv('/content/drive/MyDrive/diabetes.csv')
df

X=df.drop('class',axis=1)
y=df['class']

# standardize X data
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled

# K-Fold Cross-validation
model=DecisionTreeClassifier(random_state=42)
kf=KFold(n_splits=5,shuffle=True, random_state=42)
scores=cross_val_score(model, X_scaled, y, cv=kf)
print("Cross-validation scores:",scores)
print("Mean accuracy:", scores.mean())

# Train final model
model.fit(X_scaled, y)
# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

