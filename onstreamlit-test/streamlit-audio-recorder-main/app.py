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

st.set_page_config(page_title="Heart Sound Recorder", page_icon="🎙️")

# Custom CSS to style the UI
st.markdown('''
    <style>
        .css-1egvi7u {margin-top: -3rem;}
        .stAudio {height: 45px;}
        .css-v37k9u a, .css-nlntq9 a {color: #ff4c4b;}
        .header {background-color: #b71c1c; color: white; padding: 10px;}
        .title {font-size: 30px; margin-bottom: 10px;}
        .waveform {background-color: #f0f0f0; padding: 20px; border-radius: 5px;}
        .progress-bar {margin-top: 20px;}
    </style>
''', unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><div class="title">Heart Sound Recorder</div></div>', unsafe_allow_html=True)

# Recording status text
recording_status = st.empty()

# Function to initialize Google Drive client
@st.cache_resource
def create_drive_client():
    try:
        scope = ['https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
        gauth = GoogleAuth()
        gauth.credentials = creds
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        st.error(f"Error creating Google Drive client: {e}")

drive = create_drive_client()

# Function to download files from Google Drive
def download_from_drive(file_id, download_path):
    try:
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(download_path)
    except Exception as e:
        st.error(f"Error downloading file from Google Drive: {e}")

# Set the Google Drive file IDs
MODEL_FILE_ID = '1A2VnaPoLY3i_LakU1Y_9hB2bWuncK37X'
LABELS_FILE_ID = '1zIMcBrAi4uiL4zOVU7K2tvbw8Opcf5cW'

# Function to load model and labels
@st.cache_resource
def load_model_and_labels():
    try:
        download_from_drive(MODEL_FILE_ID, 'my_model.h5')
        download_from_drive(LABELS_FILE_ID, 'labels.csv')
        model = tf.keras.models.load_model('my_model.h5')
        labels = pd.read_csv('labels.csv')
        encoder = LabelEncoder()
        encoder.fit(labels['label'])
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model and labels: {e}")

model, encoder = load_model_and_labels()

# Function to preprocess the audio file
def preprocess_audio(file):
    try:
        audio, sample_rate = librosa.load(file, sr=None)
        if not np.isfinite(audio).all():
            raise ValueError("Audio buffer contains non-finite values")
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        else:
            raise ValueError("Audio data contains all zeros or invalid values")
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        spectrogram = librosa.power_to_db(spectrogram)
        fixed_length = 1000
        if spectrogram.shape[1] > fixed_length:
            spectrogram = spectrogram[:, :fixed_length]
        else:
            padding = fixed_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), 'constant')
        spectrogram = spectrogram.reshape((1, 128, 1000, 1))
        return spectrogram
    except Exception as e:
        st.error(f"Error in audio processing: {e}")

# Display audio waveform if recording or file upload is finished
audio_data = None  # Ensure audio_data is properly initialized
if audio_data is not None:
    try:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            progress_bar.progress(percent_complete + 1)
        progress_text.text("ดำเนินการอัดเสียงเสร็จสิ้น กดปุ่มข้างล่างเพื่อรับผลการทำนายโรค")
        if st.button('Diagnose'):
            with st.spinner('Uploading audio and getting prediction...'):
                spectrogram = preprocess_audio(audio_data)
                y_pred = model.predict(spectrogram)
                y_pred_class = np.argmax(y_pred, axis=1)
                result = encoder.inverse_transform(y_pred_class)
                st.write(f"Prediction: {result[0]}")
    except Exception as e:
        st.error(f"Error during diagnosis: {e}")
