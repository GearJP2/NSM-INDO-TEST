import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import time
import os

# Authenticate and create the PyDrive client
@st.cache_resource
def create_drive_client():
    # Define the scopes
    scope = ['https://www.googleapis.com/auth/drive']
    
    # Authenticate using service account credentials
    creds = ServiceAccountCredentials.from_json_keyfile_name('path/to/heart-d9410-9a288317e3c7.json', scope)
    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)
    return drive

drive = create_drive_client()

# Function to download files from Google Drive
def download_from_drive(file_id, download_path):
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(download_path)

# Set the Google Drive file IDs
MODEL_FILE_ID = '1A2VnaPoLY3i_LakU1Y_9hB2bWuncK37X'
LABELS_FILE_ID = '1zIMcBrAi4uiL4zOVU7K2tvbw8Opcf5cW'

# Load the pre-trained model and encoder from Google Drive
@st.cache_resource
def load_model_and_labels():
    download_from_drive(MODEL_FILE_ID, 'my_model.h5')
    download_from_drive(LABELS_FILE_ID, 'labels.csv')
    model = tf.keras.models.load_model('my_model.h5', compile=False)
    labels = pd.read_csv('labels.csv')
    encoder = LabelEncoder()
    encoder.fit(labels['label'])
    
    # Optionally compile the model if needed
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model, encoder

model, encoder = load_model_and_labels()

# Function to preprocess the audio file
def preprocess_audio(file):
    audio, sample_rate = librosa.load(file, sr=None)
    
    # Check for non-finite values and replace them
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)  # Replace NaNs and infs with zero

    # Normalize the audio
    audio = audio / np.max(np.abs(audio))  # Ensure normalization is done

    # Generate the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    spectrogram = librosa.power_to_db(spectrogram)

    # Define a fixed length for the spectrogram
    fixed_length = 1000  # Adjust this value as necessary

    # Pad or truncate the spectrogram to the fixed length
    if spectrogram.shape[1] > fixed_length:
        spectrogram = spectrogram[:, :fixed_length]
    else:
        padding = fixed_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), 'constant')

    # Reshape the spectrogram to fit the model
    spectrogram = spectrogram.reshape((1, 128, 1000, 1))

    return spectrogram

# Display audio waveform if recording or file upload is finished
if audio_data is not None:
    # Simulate a progress bar for the recording
    progress_text = st.empty()
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        progress_bar.progress(percent_complete + 1)
    progress_text.text("ดำเนินการอัดเสียงเสร็จสิ้น กดปุ่มข้างล่างเพื่อรับผลการทำนายโรค")

    # Upload button
    if st.button('Diagnose'):
        with st.spinner('Uploading audio and getting prediction...'):
            # Preprocess the audio file
            spectrogram = preprocess_audio(audio_data)

            # Make prediction
            y_pred = model.predict(spectrogram)
            y_pred_class = np.argmax(y_pred, axis=1)
            result = encoder.inverse_transform(y_pred_class)
            st.write(f"Prediction: {result[0]}")
