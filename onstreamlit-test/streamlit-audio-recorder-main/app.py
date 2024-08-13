import streamlit as st
from st_audiorec import st_audiorec
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import boto3
import time

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

# Load the pre-trained model and encoder from S3
AWS_ACCESS_KEY_ID = 'AKIAQE43KAXRWF4UZUQL'
AWS_SECRET_ACCESS_KEY = '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'
S3_BUCKET_NAME = 'model-mcexcel'
MODEL_FILE_KEY = 'my_model.h5'
LABELS_FILE_KEY = 'labels.csv'

# Function to download files from S3
def download_from_s3(bucket_name, key, download_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3.download_file(bucket_name, key, download_path)

# Download model and labels from S3
@st.cache_resource
def load_model_and_labels():
    download_from_s3(S3_BUCKET_NAME, MODEL_FILE_KEY, 'my_model.h5')
    download_from_s3(S3_BUCKET_NAME, LABELS_FILE_KEY, 'labels.csv')
    model = tf.keras.models.load_model('my_model.h5')
    labels = pd.read_csv('labels.csv')
    encoder = LabelEncoder()
    encoder.fit(labels['label'])
    return model, encoder

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
        with st.spinner('Uploading audio and getting prediction...'):
            # Preprocess the audio file
            spectrogram = preprocess_audio(audio_data)

            # Make prediction
            y_pred = model.predict(spectrogram)
            y_pred_class = np.argmax(y_pred, axis=1)
            result = encoder.inverse_transform(y_pred_class)

            st.write(f"Prediction: {result[0]}")
