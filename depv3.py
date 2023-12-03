# import streamlit as st
# import os
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.models import load_model
# import pickle
# import librosa
# import numpy as np

# model = load_model('model.h5')
# encoder = pickle.load(open('encoder.pkl', 'rb'))

# # Streamlit App
# st.title('Create DataFrame from Audio Files')

# # Specify the directory containing audio files
# audio_directory = st.text_input('Enter the path to the directory containing audio files:', '.')

# # List all audio files in the specified directory
# audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav') or f.endswith('.mp3')]

# # Display the list of available audio files
# if audio_files:
#     st.write('Available Audio Files:')
#     st.write(audio_files)

#     # Button to create DataFrame
#     if st.button('Create DataFrame'):
#         try:
#             path = []
#             labels = []
#             # Create a DataFrame with paths and labels
#             df = pd.DataFrame()
#             for filename in audio_files:
#                 path.append(os.path.join(audio_directory, filename))
#                 label = filename.split('_')[-1]
#                 label = label.split('.')[0]
#                 labels.append(label.lower())

#             df['speech'] = path
#             df['label'] = labels

#             def extract_features(filename):
#                 y, sr = librosa.load(filename, duration = 3, offset=0.5)
#                 mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
#                 return mfcc

#             # Apply feature extraction to all rows in the DataFrame
#             df['features'] = df['speech'].apply(lambda x: extract_features(x))

#             # Convert the 'features' column to a numpy array
#             X = np.array(df['features'].tolist())
#             X = np.expand_dims(X, -1)

#             # Use the encoder to transform labels
#             y = encoder.transform(df[['label']]).toarray()

#             # Make predictions using the model
#             predictions = model.predict(X)

#             # Display the predictions
#             st.write('Predicted emotions for each audio file:')
#             st.write(predictions)

#             # Display the DataFrame
#             st.write('DataFrame from Audio Files:')
#             st.dataframe(df)
#         except Exception as e:
#             st.error(f'Error: {e}')
# else:
#     st.warning('No audio files found in the specified directory.')

import streamlit as st
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import pickle
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model.h5')
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Streamlit App
st.title('Speech-based Emotion Recognition App')

# Upload audio file through Streamlit
audio_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3'])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    # Display spectrogram when the user clicks the button
    if st.button('Show Spectrogram'):
        try:
            # Save the uploaded audio to a file
            file_path = "uploaded_audio.wav"
            with open(file_path, "wb") as f:
                f.write(audio_file.read())

            # Load the audio file
            y, sr = librosa.load(file_path)

            # Display the spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y))), sr=sr, x_axis='time', y_axis='log')
            plt.title('Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            st.pyplot()

            # Optionally, you can also display the emotion prediction
            # This assumes you have defined the functions 'extract_features' and 'predict_emotion' as in the previous examples
            features = extract_features(file_path)
            prediction = model.predict(np.expand_dims(features, axis=0))
            emotion_index = np.argmax(prediction)
            predicted_emotion = encoder.categories_[0][emotion_index]
            st.success(f'The predicted emotion is: {predicted_emotion}')
        except Exception as e:
            st.error(f'Error: {e}')
