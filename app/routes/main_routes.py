from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app.models import Scan, db
import os
import numpy as np
from PIL import Image
from datetime import datetime

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prediction(image_path):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    
    # Load TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path="model/Keras_Model_2/model_unquant.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0]
    
    # Load labels
    with open("model/Keras_Model_2/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Get the predicted class and confidence
    predicted_class = labels[np.argmax(prediction)]
    confidence = float(max(prediction))
    
    return predicted_class, confidence

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('main.index'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('main.index'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            prediction, confidence = get_prediction(filepath)
            
            # Save scan to database
            new_scan = Scan(
                filename=filename,
                prediction=prediction,
                confidence=confidence,
                user_id=current_user.id
            )
            db.session.add(new_scan)
            db.session.commit()
            
            return {'prediction': prediction, 'confidence': confidence}
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('main.index'))
            
    flash('Invalid file type. Please upload a PNG or JPG file.')
    return redirect(url_for('main.index'))

@main.route('/past')
@login_required
def past_scans():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('past.html', scans=scans)

@main.route('/info')
def info():
    return render_template('info.html')