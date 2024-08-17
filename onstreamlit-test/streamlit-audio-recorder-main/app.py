import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import time
import os

# Set the page configuration
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

# Initialize recording state
if 'recording' not in st.session_state:
    st.session_state['recording'] = False

# Update recording status text based on the recording state
if st.session_state['recording']:
    recording_status.markdown('<div class="waveform">กำลังบันทึกเสียง</div>', unsafe_allow_html=True)
else:
    recording_status.markdown('<div class="waveform">กดเพื่อบันทึกเสียง</div>', unsafe_allow_html=True)

# Audio recorder
wav_audio_data = st_audiorec()

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['wav'])

# Use recorded audio or uploaded file for prediction
audio_data = None

if wav_audio_data is not None:
    audio_data = wav_audio_data
    st.audio(wav_audio_data, format='audio/wav')
elif uploaded_file is not None:
    audio_data = uploaded_file
    st.audio(uploaded_file, format='audio/wav')

# Authenticate and create the PyDrive client
def authenticate_gdrive():
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    gauth.credentials = credentials
    drive = GoogleDrive(gauth)
    return drive

# Function to download files from Google Drive
def download_from_gdrive(drive, file_id, download_path):
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(download_path)

# Authenticate and create the drive client
drive = authenticate_gdrive()

# IDs of the files in Google Drive
MODEL_FILE_ID = 'your_model_file_id'
LABELS_FILE_ID = 'your_labels_file_id'

# Paths to store downloaded files
MODEL_FILE_PATH = 'my_model.h5'
LABELS_FILE_PATH = 'labels.csv'

# Download model and labels from Google Drive
@st.cache_resource
def load_model_and_labels():
    try:
        download_from_gdrive(drive, MODEL_FILE_ID, MODEL_FILE_PATH)
        download_from_gdrive(drive, LABELS_FILE_ID, LABELS_FILE_PATH)
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        labels = pd.read_csv(LABELS_FILE_PATH)
        encoder = LabelEncoder()
        encoder.fit(labels['label'])
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model and labels: {e}")
        return None, None

model, encoder = load_model_and_labels()

# Function to preprocess the audio file
def preprocess_audio(file):
    audio, sample_rate = librosa.load(file, sr=None)

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
        if model is not None and encoder is not None:
            with st.spinner('Uploading audio and getting prediction...'):
                try:
                    # Preprocess the audio file
                    spectrogram = preprocess_audio(audio_data)

                    # Make prediction
                    y_pred = model.predict(spectrogram)
                    y_pred_class = np.argmax(y_pred, axis=1)
                    result = encoder.inverse_transform(y_pred_class)

                    st.write(f"Prediction: {result[0]}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.error("Model or encoder is not loaded properly.")
