import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "emotion_speech")  # Use existing data for testing

# Load model and scaler
model = load_model(os.path.join(MODEL_DIR, "emotion_model.h5"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_classes = np.load(os.path.join(MODEL_DIR, "label_classes.npy"), allow_pickle=True)

# Function to extract features
def extract_features(file_path, max_length=173):
    y, sr = librosa.load(file_path, duration=3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    return mfccs.T

# Predict function
def predict_emotion(file_path):
    mfccs = extract_features(file_path)
    mfccs = scaler.transform(mfccs.reshape(1, -1)).reshape(1, 173, 13, 1)
    prediction = model.predict(mfccs)
    predicted_class = np.argmax(prediction, axis=1)
    return label_classes[predicted_class][0]

# Test with a sample file
sample_file = os.path.join(SAMPLE_DIR, "Actor_01", "03-01-03-01-02-01-01.wav")  # Example: happy
if os.path.exists(sample_file):
    emotion = predict_emotion(sample_file)
    print(f"✅ Predicted emotion for {sample_file}: {emotion}")
else:
    print("⚠️ Sample file not found. Please ensure the dataset is correctly placed.")

print("✅ Prediction script ready. Use predict_emotion() with any .wav file path.")