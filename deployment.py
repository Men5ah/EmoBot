import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle

# Load the pre-trained model and encoder
model = load_model('model.pkl')
encoder = pickle.load(open('encoder.pkl', 'rb'))


# Streamlit App
st.title('Speech-based Emotion Recognition App')

# Upload audio file through Streamlit
audio_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3'])

# if audio_file is not None:
#     st.audio(audio_file, format='audio/wav')

#     # Make prediction when the user clicks the button
# if st.button('Predict Emotion'):
#     try:
#         emotion = predict_emotion(audio_file)
#         st.success(f'The predicted emotion is: {emotion}')
#     except Exception as e:
#         st.error(f'Error: {e}')

# Function to extract MFCC features from audio
def extract_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1, 1)

# Function to predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    emotion = encoder.categories_[0][emotion_index]
    return emotion
