import io
import time
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
from st_audiorec import st_audiorec
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import audioread

# Google Drive setup
SERVICE_ACCOUNT_FILE = 'onstreamlit-test/streamlit-audio-recorder-main/heart-d9410-9a288317e3c7.json'
def extract_heart_sound(audio):
# Function to preprocess the audio file
def preprocess_audio(file, file_format):
    file_bytes = io.BytesIO(file.read())

    # Load the audio file using librosa (which uses audioread)
    y, sr = librosa.load(file_bytes, sr=None)

    # Normalize the audio
    audio = y / np.max(np.abs(y))


    if file_format == 'mp3':
        audio = AudioSegment.from_mp3(file_bytes)
    elif file_format == 'm4a':
        audio = AudioSegment.from_file(file_bytes, format='m4a')
    elif file_format == 'wav':
        audio = AudioSegment.from_wav(file_bytes)
    else:
        raise ValueError("Unsupported file format")

    sample_rate = audio.frame_rate  # Get the sample rate before conversion
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
    audio = audio / np.max(np.abs(audio))  # Normalize the audio

    # Extract heart sound using Fourier transform
    heart_sound = extract_heart_sound(audio)

    # Generate the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
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
# Streamlit interface for recording and uploading audio files
st.set_page_config(page_title="Heart Sound Recorder", page_icon="🎙️")
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
st.markdown('<div class="header"><div class="title">Heart Sound Recorder</div></div>', unsafe_allow_html=True)
recording_status = st.empty()
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if st.session_state['recording']:
    recording_status.markdown('<div class="waveform">Recording...</div>', unsafe_allow_html=True)
else:
    recording_status.markdown('<div class="waveform">Click to record</div>', unsafe_allow_html=True)
wav_audio_data = st_audiorec()
uploaded_file = st.file_uploader("Choose a file", type=['wav', 'mp3', 'm4a'])
audio_data = None
file_format = None
if wav_audio_data is not None:
    audio_data = io.BytesIO(wav_audio_data)
    file_format = 'wav'
    st.audio(wav_audio_data, format='audio/wav')
elif uploaded_file is not None:
    audio_data = uploaded_file
    file_format = uploaded_file.type.split('/')[1]
    st.audio(uploaded_file, format=f'audio/{file_format}')
if audio_data is not None:
    progress_text = st.empty()
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        progress_bar.progress(percent_complete + 1)
    progress_text.text("Recording complete. Click the button below to get the prediction.")
    if st.button('Diagnose'):
        with st.spinner('Uploading audio and getting prediction...'):
            spectrogram = preprocess_audio(audio_data, file_format)
            y_pred = model.predict(spectrogram)
            y_pred_class = np.argmax(y_pred, axis=1)
            result = encoder.inverse_transform(y_pred_class)
            st.write(f"Prediction: {result[0]}")
