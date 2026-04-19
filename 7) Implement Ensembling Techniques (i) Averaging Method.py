import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['amount','frequency']]
y = df['label']

model = BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
model.fit(X,y)

print(model.score(X,y))
