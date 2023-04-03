from XRay_vision import trained_model
from flask import Flask
from flask import request
import numpy as np
import cv2

app = Flask(__name__)


@app.route("/predict", methods=["post"])
def predict():
    file = request.files["XRay"]
    file.save(f'./temp/{file.filename}')
    image = cv2.imread(f'./temp/{file.filename}')
    image = cv2.resize(image, dsize=(200, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_test = []
    x_test.append(image)
    X_test = np.array(x_test)
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    prediction = trained_model.predict(X_test)

    return {"Bacteria": str(prediction[0][0]), "Normal": str(prediction[0][1]), "Virus": str(prediction[0][2])}


app.run(port=3000)
