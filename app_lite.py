from flask import Flask, render_template, request
from flask_login import LoginManager, current_user, login_required
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import warnings
from datetime import datetime

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'


from models import db, User, Scan
db.init_app(app)


login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


from auth import auth as auth_blueprint
app.register_blueprint(auth_blueprint)

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

    class_idx = np.argmax(output_data[0])
    confidence = output_data[0][class_idx]

    return CLASS_NAMES[class_idx], confidence


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    prediction = None
    confidence = None
    uploaded_file = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            os.makedirs("static/uploads", exist_ok=True)
            filepath = os.path.join("static/uploads", filename)
            file.save(filepath)
            prediction, confidence = predict_image(filepath)
            uploaded_file = os.path.join("uploads", filename)  

            
            scan = Scan(
                filename=filename,
                prediction=prediction,
                confidence=confidence,
                user_id=current_user.id
            )
            db.session.add(scan)
            db.session.commit()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_file=uploaded_file
    )


@app.route("/info")
def info():
    return render_template("info.html")


@app.route("/past")
@login_required
def past():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template("past.html", scans=scans)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
