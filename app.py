from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
# from tensorflow.keras.applications import MobileNetV2 # more advanced model if we get there

app = Flask(__name__)

MODEL_PATH = "model/model.h5"  # Teachable Machine file (may need to convert the model first)
# CLASS_NAMES = ["Recyclable", "Compostable", "Garbage"]

model = tf.keras.models.load_model(MODEL_PATH)
with open("model/labels.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]


def predict_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    print(f"Prediction: {CLASS_NAMES[class_idx]} ({confidence*100:.2f}% confidence)")
    return CLASS_NAMES[class_idx], confidence




@app.route("/", methods=["GET", "POST"]) # anytime we visit / it will run 
def index():
    prediction = None
    confidence = None

    if request.method == "POST": # if html's form is submitted it sends this post request
        file = request.files["file"] # get file from users post req. Form field's input name=file for this
        if file: 
            os.makedirs("static/uploads", exist_ok=True)
            filepath = os.path.join("static/uploads", file.filename)
            file.save(filepath)  # save uploaded file in folder
            prediction, confidence = predict_image(filepath)
        

    return render_template("index.html", prediction=prediction, confidence=confidence)



@app.route("/info")
def showInfo(): # and maybe also show a history of past scans they've done -- mongodb per user?  
    print("Info about what materials are recyclable, compostable, etc")


if __name__ == "__main__":
    app.run(debug=True)

    # for testing initially, delete later
    image_path = "test_images/test_bottle.jpg"  # change this to your image
    predict_image(image_path) # to test
