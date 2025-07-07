from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Absolute path to model file
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "xgb_adr_model.pkl")
print("Loading model from:", model_path)

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        med_count = int(request.form["med_count"])
        condition = 1 if request.form["condition"] == "yes" else 0
        visits = int(request.form["visits"])
        allergies = 1 if request.form["allergies"] == "yes" else 0
        vaccines = 1 if request.form["vaccines"] == "yes" else 0

        # Sample features (replace as needed)
        features = np.array([[med_count, med_count, condition, visits, 1, age, gender, allergies, vaccines]])
        prob = model.predict_proba(features)[0][1]
        prediction = f"⚠️ ADR Risk: {prob:.2f}" if prob >= 0.3 else f"✅ No ADR Risk: {prob:.2f}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
