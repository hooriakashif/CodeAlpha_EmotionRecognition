# CodeAlpha_EmotionRecognition

## Project Overview
This repository contains my implementation of **Task 2: Emotion Recognition from Speech** as part of the CodeAlpha ML Internship 2025. The project leverages a Convolutional Neural Network (CNN) to classify emotions from audio data using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The goal is to identify emotions such as happy, sad, angry, and neutral from speech samples, showcasing practical machine learning applications in audio processing.

## Features
- **Data Preprocessing**: Audio files are converted into spectrograms for CNN input.
- **Model Training**: A CNN architecture is trained using TensorFlow to achieve high accuracy in emotion classification.
- **Prediction Script**: Includes a script to predict emotions from new audio samples.
- **EDA**: Exploratory Data Analysis with visualizations to understand emotion distribution and audio features.

## Technologies Used
- **Python**: Core programming language.
- **TensorFlow**: For building and training the CNN.
- **Librosa**: For audio feature extraction and spectrogram generation.
- **Matplotlib/Seaborn**: For data visualization.

## Installation
1. Clone the repository: `git clone https://github.com/hooriakashif/CodeAlpha_EmotionRecognition.git`.
2. Install dependencies: `pip install tensorflow librosa matplotlib seaborn numpy`.
3. Download the RAVDESS dataset manually (not included due to size) and place it in the `data/` folder (create if absent).

## Usage
- Run `python src/data_preprocess.py` to generate spectrograms.
- Train the model with `python src/model_train.py`.
- Predict emotions with `python src/predict.py` using a sample audio file.

## Results
The model achieves competitive accuracy on the RAVDESS dataset, with performance varying by emotion class. Detailed metrics are logged during training.

## Notes
The RAVDESS dataset is excluded from this repository due to its large size. Please download it separately and adjust paths in `src/data_preprocess.py` accordingly.

## Acknowledgments
Thanks to CodeAlpha for this enriching internship opportunity! #MachineLearning #DeepLearning #CodeAlpha #Internship #AI
