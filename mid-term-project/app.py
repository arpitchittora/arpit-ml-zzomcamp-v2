import pickle

from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb
import numpy as np

model_file = 'model_1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    features = dv.get_feature_names()
    movie = request.get_json()

    X = dv.transform([movie])
    dval = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dval)

    expected_rating = round(np.expm1(float(y_pred)), 3)

    result = {
        'expected_rating': expected_rating
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
