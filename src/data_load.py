import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "emotion_speech")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion mapping based on RAVDESS filename convention
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Function to extract MFCCs
def extract_features(file_path, max_length=173):
    y, sr = librosa.load(file_path, duration=3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    return mfccs.T  # Transpose for time-series shape

# Load and process data
X, y = [], []
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                mfccs = extract_features(file_path)
                emotion_code = file.split('-')[2]  # Extract emotion code (e.g., '03' for happy)
                emotion = emotion_map[emotion_code]
                X.append(mfccs)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save processed data
np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), y_encoded)
np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), label_encoder.classes_)

print(f"✅ Processed {len(X)} audio files.")
print(f"✅ Features shape: {X.shape}")
print(f"✅ Unique emotions: {label_encoder.classes_}")