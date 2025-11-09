from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
import warnings

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "Keras_Model_2")
MODEL_PATH = os.path.join(MODEL_DIR, "model_unquant.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: model file not found: {MODEL_PATH}")
    sys.exit(1)
if not os.path.exists(LABELS_PATH):
    print(f"ERROR: labels file not found: {LABELS_PATH}")
    sys.exit(1)

with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224)).convert("RGB")
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    class_idx = np.argmax(output_data[0])
    confidence = output_data[0][class_idx]

    return CLASS_NAMES[class_idx], float(confidence)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    uploaded_file = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            os.makedirs(os.path.join(SCRIPT_DIR, "static", "uploads"), exist_ok=True)
            filepath = os.path.join(SCRIPT_DIR, "static", "uploads", file.filename)
            file.save(filepath)
            prediction, confidence = predict_image(filepath)
            uploaded_file = os.path.relpath(filepath, SCRIPT_DIR)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_file=uploaded_file
    )


@app.route("/info")
def showInfo():
    return "Info about what materials are recyclable, compostable, etc"


if __name__ == "__main__":
    # Bind to all interfaces so you can access from other devices if needed
    app.run(host="127.0.0.1", port=5000, debug=True)
