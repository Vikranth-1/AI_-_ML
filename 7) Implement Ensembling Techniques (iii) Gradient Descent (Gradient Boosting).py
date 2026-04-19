import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['rd_spend','admin','marketing']]
y = df['salary']

model = GradientBoostingRegressor()
model.fit(X,y)

print(model.score(X,y))
