import pycaret
import pandas as pd
from pycaret.classification import load_model, predict_model

import numpy as np
from flask import Flask, request, render_template

# create flask app
app = Flask(__name__)

# Load the pickle model
model = load_model("best-model")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = pd.DataFrame([np.array(float_features)],columns=['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight',
       'Viscera_weight', 'Shell_weight', 'Rings'])
    prediction = predict_model(model,features)

    return render_template("index.html", prediction_text="The abalone sex is {}".format(prediction['prediction_label'][0]))


if __name__ == "__main__":
    app.run(debug=True)
