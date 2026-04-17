import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

df = pd.read_csv("ai_ml_lab_data.csv")

data = df[['age','sex','cp','exang','restecg','chol','heartdisease']]

model = BayesianModel([
    ('age','heartdisease'),
    ('sex','heartdisease'),
    ('cp','heartdisease'),
    ('exang','heartdisease'),
    ('heartdisease','restecg')
])

model.fit(data, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)
print(infer.query(variables=['heartdisease'], evidence={'restecg':1}))
