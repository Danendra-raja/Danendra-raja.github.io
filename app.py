from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image
import io

app = Flask(__name__)

# Load model SVM
model = joblib.load("model_svm_trash.pkl")

# Label urutan sama seperti waktu training
labels = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((150, 150))
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features.reshape(1, -1)

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    file = request.files['image']
    image_bytes = file.read()
    try:
        features = preprocess_image(image_bytes)
        prediction = model.predict(features)[0]
        result = prediction
    except Exception as e:
        print("EROR:", e)
        result = "Prediction failed"
    return jsonify({'result': result})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    file = request.files['image']
    image_bytes = file.read()
    try:
        features = preprocess_image(image_bytes)
        prediction = model.predict(features)[0]
        result = prediction
    except Exception as e:
        print("EROR:", e)
        result = "Prediction failed"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
