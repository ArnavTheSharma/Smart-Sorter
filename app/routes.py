from flask import Blueprint, render_template, request, url_for
from flask_login import login_required, current_user
from app.models import Upload
from app import db
import tensorflow as tf
import numpy as np
from PIL import Image
import os

main_bp = Blueprint('main', __name__)

MODEL_PATH = "./model/Keras_Model_2/model_unquant.tflite"
with open("model/Keras_Model_2/labels.txt", "r") as f:
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
    idx = np.argmax(output_data[0])
    return CLASS_NAMES[idx], float(output_data[0][idx])

@main_bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    prediction = confidence = uploaded_file = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload":
            file = request.files.get("file")
            if file and file.filename:
                os.makedirs("app/static/uploads", exist_ok=True)
                upload_dir = os.path.join(os.getcwd(), "app/static/uploads")
                os.makedirs(upload_dir, exist_ok=True)

                filepath = os.path.join(upload_dir, file.filename)
                file.save(filepath)
                prediction, confidence = predict_image(filepath)

                # Flask static URLs are relative to /static/
                uploaded_file = url_for('static', filename=f'uploads/{file.filename}')

                relative_path = f'uploads/{file.filename}'  
                upload = Upload(user_id=current_user.id, image_path=relative_path,
                                prediction=prediction, confidence=confidence)
                db.session.add(upload)
                db.session.commit()

        elif action == "clear":
            prediction = confidence = uploaded_file = None

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           uploaded_file=uploaded_file)

@main_bp.route("/info")
def showInfo():
    return render_template("info.html")

@main_bp.route("/past")
@login_required
def past_uploads():
    uploads = Upload.query.filter_by(user_id=current_user.id).all()
    return render_template("past.html", uploads=uploads)
