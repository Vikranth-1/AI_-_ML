import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.read_csv("ai_ml_lab_data.csv")

data = df[['amount','frequency']]

model = DBSCAN(eps=10, min_samples=2)
print(model.fit_predict(data))
