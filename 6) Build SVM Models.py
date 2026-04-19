import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['weight','size']]
y = df['fruit']

le = LabelEncoder()
y = le.fit_transform(y)

model = SVC(kernel='rbf')
model.fit(X,y)

print(model.predict([[130,13]]))
