import pandas as pd
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("ai_ml_lab_data.csv")

data = df[['hair','feathers','eggs','milk']]

model = AgglomerativeClustering(n_clusters=2)
print(model.fit_predict(data))
