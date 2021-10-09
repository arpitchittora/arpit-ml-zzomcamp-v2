import pickle

dv_file = 'dv.bin'
model_file = 'model1.bin'

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]
churn = y_pred >= 0.5

result = {
    'churn_probability': float(y_pred),
    'churn': bool(churn)
}
print(result)
