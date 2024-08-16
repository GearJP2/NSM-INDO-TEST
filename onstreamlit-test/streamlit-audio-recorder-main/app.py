# Import necessary libraries
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
import io

# Function to create Google Drive client
def create_drive_client():
    scope = ['https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('onstreamlit-test/streamlit-audio-recorder-main/heart-d9410-9a288317e3c7.json', scope)
    gauth = GoogleAuth()
    gauth.credentials = creds
    drive = GoogleDrive(gauth)
    return drive

# Create Google Drive client
drive = create_drive_client()

# Function to download files from Google Drive
def download_from_drive(file_id, download_path):
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(download_path)

# Set the Google Drive file IDs
MODEL_FILE_ID = '1A2VnaPoLY3i_LakU1Y_9hB2bWuncK37X'
LABELS_FILE_ID = '1zIMcBrAi4uiL4zOVU7K2tvbw8Opcf5cW'

# Load the pre-trained model and encoder from Google Drive
def load_model_and_labels():
    download_from_drive(MODEL_FILE_ID, 'my_model.h5')
    download_from_drive(LABELS_FILE_ID, 'labels.csv')
    model = tf.keras.models.load_model('my_model.h5', compile=False)
    labels = pd.read_csv('labels.csv')
    encoder = LabelEncoder()
    encoder.fit(labels['label'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, encoder

# Load model and encoder
model, encoder = load_model_and_labels()

# Function to extract heart sounds using Fourier transform
def extract_heart_sound(audio):
    fourier_transform = np.fft.fft(audio)
    heart_sound = np.abs(fourier_transform)
    return heart_sound

# Function to preprocess the audio file
def preprocess_audio(file):
    audio, sample_rate = librosa.load(file, sr=None)
    heart_sound = extract_heart_sound(audio)
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
    recording_status.markdown('<div class="waveform">Recording...</div>', unsafe_allow_html=True)
else:
    recording_status.markdown('<div class="waveform">Click to start recording</div>', unsafe_allow_html=True)

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

# Display audio waveform if recording or file upload is finished
if audio_data is not None:
    # Simulate a progress bar for the recording
    progress_text = st.empty()
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        progress_bar.progress(percent_complete + 1)
    progress_text.text("Recording complete. Click the button below to get the diagnosis.")

    # Upload button
    if st.button('Diagnose'):
        with st.spinner('Uploading audio and getting prediction...'):
            audio_file = io.BytesIO(audio_data)
            spectrogram = preprocess_audio(audio_file)

            # Make prediction
            y_pred = model.predict(spectrogram)
            y_pred_class = np.argmax(y_pred, axis=1)
            result = encoder.inverse_transform(y_pred_class)

            st.write(f"Prediction: {result[0]}")
