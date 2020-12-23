"""Clustering using Boston dataset."""
# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
iris = datasets.load_iris()
X = iris.data

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create k-mean object
clt = KMeans(n_clusters=3, random_state=0, n_jobs=-1)

# Train model
model = clt.fit(X_std)

# View predict class
model.labels_

# Create new observation
new_observation = [0.8, 0.8, 0.8, 0.8]

# Predict observation's cluster
model.predict(new_observation)

# View cluster centers
model.cluster_centers_


#Encoding categorical features

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

df = pd.read_csv('http://bit.ly/kaggletrain')
df = df.loc[df.Embarked.notna(), ['Survived', 'Pclass', 'Sex', 'Embarked']]
X = df.drop('Survived', axis='columns')
y = df.Survived

column_trans = make_column_transformer(
    (OneHotEncoder(), ['Sex', 'Embarked']),
    remainder='passthrough')
logreg = LogisticRegression(solver='lbfgs')

pipe = make_pipeline(column_trans, logreg)

cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
X_new = X.sample(5, random_state=99)
pipe.fit(X, y)
pipe.predict(X_new)
