import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")

# Load processed data
X = np.load(os.path.join(DATA_DIR, "features.npy"))
y_encoded = np.load(os.path.join(DATA_DIR, "labels.npy"))
label_classes = np.load(os.path.join(DATA_DIR, "label_classes.npy"), allow_pickle=True)

# Reshape for CNN (add channel dimension)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[1] * X.shape[2])).reshape(X.shape)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

# Convert labels to categorical
y_categorical = to_categorical(y_encoded, num_classes=len(label_classes))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_classes), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# Save model
model.save(os.path.join(OUTPUT_DIR, "emotion_model.h5"))
print(f"✅ Model saved to {OUTPUT_DIR}/emotion_model.h5")