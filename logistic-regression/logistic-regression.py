"""Logistic regression Iris dataset."""
# Load libraries
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data with only two classes
iris = datasets.load_iris()
X = iris.data[:100, :]
y = iris.target[:100]

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create logistic regression object
clf = LogisticRegression(random_state=0)

# Train model
model = clf.fit(X_std, y)

# Create new observation
new_observation = [[.5, .5, .5, .5]]

# Predict class
model.predict(new_observation)

# View predicted probabilities
model.predict_proba(new_observation)
