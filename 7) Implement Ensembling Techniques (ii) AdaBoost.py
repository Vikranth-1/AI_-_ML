import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['amount','frequency']]
y = df['label']

model = AdaBoostClassifier(n_estimators=50)
model.fit(X,y)

print(model.score(X,y))
