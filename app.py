from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(os.path.join('models', 'shoe-sandal-boot.h5'))

# Define allowed file extensions and upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Please Upload the Image First')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='Please Upload the Image First')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        resize = tf.image.resize(img, (256, 256))
        resized_img = np.expand_dims(resize / 255, 0)

        prediction = model.predict(resized_img)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:
            category = "boots"
        elif predicted_class == 1:
            category = "sandals"
        elif predicted_class == 2:
            category = "shoes"
        else:
            category = "unknown"

        return render_template('index.html', filename=filename, category=category)

    return render_template('index.html', error='Invalid File Type')

if __name__ == '__main__':
    app.run(debug=True)