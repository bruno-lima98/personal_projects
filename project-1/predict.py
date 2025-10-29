import pandas as pd
import os
import pickle
from preprocessing import preprocess_data
from model_wrapper import StartupFailureModel
from flask import Flask, request, jsonify

model_path = os.path.join(os.path.dirname(__file__), "model_C=0.1.bin")

with open(model_path, "rb") as f_in:
    model_wrapper = pickle.load(f_in)

app = Flask('startup_failure')
@app.route('/predict', methods=['POST'])

def predict():
    startup = request.get_json()

    df_startup = pd.DataFrame([startup])

    df_processed, cat_cols, num_cols = preprocess_data(df_startup)
    pred_class = model_wrapper.predict(df_processed)
    pred_proba = model_wrapper.predict_proba(df_processed)

    result = {
        'failure_probability' : float(pred_proba),
        'failure' : bool(pred_class)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)