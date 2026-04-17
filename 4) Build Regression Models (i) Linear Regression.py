import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("ai_ml_lab_data.csv")

X = df[['experience']]
y = df['salary']

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

plt.scatter(X,y)
plt.plot(X,y_pred)
plt.show()
