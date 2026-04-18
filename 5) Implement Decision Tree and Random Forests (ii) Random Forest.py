import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['rd_spend','admin','marketing']]
y = df['salary']

model = RandomForestRegressor()
model.fit(X,y)

print(model.predict([[70000,12000,8000]]))
