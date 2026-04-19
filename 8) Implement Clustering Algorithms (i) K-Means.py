import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("ai_ml_lab_data.csv")

data = df[['amount','frequency']]

model = KMeans(n_clusters=2, n_init=10)
model.fit(data)

print(model.labels_)
