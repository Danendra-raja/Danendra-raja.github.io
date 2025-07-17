import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import joblib

# Konfigurasi
data_dir = 'dataset'  # Folder dataset dengan subfolder nama kelas
classes = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']
img_size = (150, 150)

features = []
labels = []

print("Ekstraksi fitur HOG dari dataset...")

# Load dan ekstrak fitur HOG dari gambar
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    return hog_features

for label in classes:
    class_dir = os.path.join(data_dir, label)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            features.append(extract_hog_features(img))
            labels.append(label)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Training SVM
print("Training model SVM...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluasi
print("Evaluasi model:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Simpan model
model_path = 'model_svm_trash.pkl'
joblib.dump(model, model_path)
print(f"Model disimpan di: {model_path}")