import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("ai_ml_lab_data.csv")

X = df["message"]
y = df["label"]

cv = CountVectorizer()
X_vec = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

print(model.predict(cv.transform(["free money"])))
