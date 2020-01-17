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
