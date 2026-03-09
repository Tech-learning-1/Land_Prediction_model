from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)
path = Path("artifacts/land_prediction.pkl")

if not path.exists():
    import train
    train.main()
model=joblib.load(path)


@app.route("/health",  methods=["Get"])
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["post"])
def predict():
      data = request.get_json()
      if not data or not  "features" in data:
           return jsonify({"error": "please send the features in curl request"})
      features = data["features"]
      try:
          prediction = model.predict([features])
          return jsonify({"predictions": int(prediction[0])})
      except Exception as e:
          return jsonify({"error": str(e)})
      
if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5001)