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
    
    # The actual filename of your service account JSON file
    service_account_file = 'onstreamlit-test/streamlit-audio-recorder-main/heart-d9410-9a288317e3c7.json'  # Update this path as necessary
    if not os.path.exists(service_account_file):
        st.error(f"Service account file '{service_account_file}' not found. Please upload the file.")
        st.stop()
    
    # Authenticate using service account credentials
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_file, scope)
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
    model = tf.keras.models.load_model('my_model.h5')
    labels = pd.read_csv('labels.csv')
    encoder = LabelEncoder()
    encoder.fit(labels['label'])
    return model, encoder

model, encoder = load_model_and_labels()

# Function to preprocess the audio file
def preprocess_audio(raw_audio):
    # Convert raw audio data to a numpy array
    audio = np.frombuffer(raw_audio, dtype=np.float32)

    # Validate the audio data
    if not np.isfinite(audio).all():
        raise ValueError("Audio buffer contains non-finite values")

    # Assuming the sample rate is 44.1kHz for the audio data
    sample_rate = 44100

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

# Main Streamlit app logic
audio_data = st_audiorec()  # Use the st_audiorec function to record audio

# Display audio waveform if recording or file upload is finished
if audio_data is not None:
    # Simulate a progress bar for the recording
    progress_text = st.empty()
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        progress_bar.progress(percent_complete + 1)
    progress_text.text("Recording complete. Click the button below to diagnose.")

    # Upload button
    if st.button('Diagnose'):
        with st.spinner('Uploading audio and getting prediction...'):
            try:
                # Preprocess the audio file
                spectrogram = preprocess_audio(audio_data)

                # Make prediction
                y_pred = model.predict(spectrogram)
                y_pred_class = np.argmax(y_pred, axis=1)
                result = encoder.inverse_transform(y_pred_class)
                st.success(f'Diagnosis result: {result[0]}')
            except ValueError as e:
                st.error(f"Error in audio processing: {e}")
