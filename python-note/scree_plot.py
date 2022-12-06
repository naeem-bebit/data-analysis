from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/USArrests.csv"
data = pd.read_csv(url)
df = data.iloc[:, 1:5]
scaler = StandardScaler()
scaled_df = df.copy()
scaled_df = pd.DataFrame(scaler.fit_transform(
    scaled_df), columns=scaled_df.columns)

pca = PCA(n_components=4)
pca_fit = pca.fit(scaled_df)
