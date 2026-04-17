import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['experience','salary']]
y = df['label']

model = DecisionTreeClassifier()
model.fit(X,y)

print(model.predict([[3,50000]]))
