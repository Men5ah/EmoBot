import streamlit as st
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle
import soundfile as sf

# Load the pre-trained model and encoder
model = load_model('model.h5')
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Function to extract MFCC features from audio
def extract_features(filename):
    y, sr = librosa.load(filename, duration=10, offset=0.5)
    y = librosa.util.normalize(y)  # Normalize the audio signal
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    emotion = encoder.categories_[0][emotion_index]
    return emotion

# Streamlit App
st.title('Speech-based Emotion Recognition App')

# Specify the directory containing audio files
audio_directory = st.text_input('Enter the path to the directory containing audio files:', '.')

# List all audio files in the specified directory
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav') or f.endswith('.mp3')]

# Display the list of available audio files
selected_file = st.selectbox('Select an audio file:', audio_files)

# Display the selected audio file
if selected_file:
    audio_path = os.path.join(audio_directory, selected_file)
    st.audio(audio_path, format='audio/wav')

    if st.button('Predict Emotion'):
        try:
            # Apply the same feature extraction as during training
            features = extract_features(audio_path)

            # Apply the same one-hot encoding as during training
            label = predict_emotion(audio_path)
            encoded_label = encoder.transform([[label]]).toarray()

            st.success(f'The predicted emotion for {selected_file} is: {label}')
            # st.text(f'One-hot encoded label: {encoded_label}')
        except Exception as e:
            st.error(f'Error: {e}')