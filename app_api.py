from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    result = "FRAUDE" if pred == 1 else "NORMALE"
    return jsonify({"prediction": int(pred), "status": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
