# Project Title: EmoBot
# Introduction
This is an Introduction to Ai project that was developed by Robert Sika and John Adenu-Mensah. The essence of this project is to create an unsupervised AI model that is able to predict emotions using a Recurrent Neural Network model Long Short Term Model (LSTM)
##
# Contents
* Features
* How To Use
* Demonstration
##

# Features
  The emobot uses the Tess dataset to train and test it's accuracy and validation. It used MFCC to extract features from the audio files and those features were used in the training and testing process using an LSTM and some Dense models which used various activation functions like relu and softmax and the rmsprop optimizer.
  The application has a user-friendly interface which was made using Streamlit, a lightweight python library that can be used to host web applications, The aplication allows a user to upload a file and then predict the emotion that is found in the uploaded file.

# How To Use
  To use our web application you can follow these simple steps:
  ## Step 1:
    Access our application at emo-bot.streamlit.app and upload a 1 to 3 second long file in the form OAF_test_[emotion] or YAF_test_[emotion]
