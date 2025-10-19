import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_results")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
X = np.load(os.path.join(DATA_DIR, "features.npy"))
y_encoded = np.load(os.path.join(DATA_DIR, "labels.npy"))
label_classes = np.load(os.path.join(DATA_DIR, "label_classes.npy"), allow_pickle=True)

# Reduce dimensions for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.reshape(X.shape[0], -1))

# Plot t-SNE visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap='viridis')
plt.colorbar(scatter, label='Emotion')
plt.clim(-0.5, 7.5)  # Adjust for 8 classes (0 to 7)
plt.title('t-SNE Visualization of Emotion Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_emotion_plot.png"))
plt.close()

# Plot emotion distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=label_classes[y_encoded])
plt.title('Distribution of Emotions in Dataset')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, "emotion_distribution.png"))
plt.close()

print(f"✅ EDA plots saved to {OUTPUT_DIR}")