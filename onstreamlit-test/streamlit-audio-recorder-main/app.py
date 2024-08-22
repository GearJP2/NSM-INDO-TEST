import io
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
import tempfile
import os
import matplotlib.pyplot as plt
import h5py

st.set_page_config(page_title="Heart Sound Recorder", page_icon="🎙️")

# Google Drive setup
SERVICE_ACCOUNT_FILE = 'onstreamlit-test/streamlit-audio-recorder-main/heart-d9410-9a288317e3c7.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def download_file_from_google_drive(file_id, destination):
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(destination, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

GOOGLE_DRIVE_MODEL_FILE_ID = '1A2VnaPoLY3i_LakU1Y_9hB2bWuncK37X'
MODEL_FILE_PATH = 'my_model.h5'
LABELS_FILE_PATH = 'onstreamlit-test/streamlit-audio-recorder-main/labels.csv'

# Attempt to download files
download_file_from_google_drive(GOOGLE_DRIVE_MODEL_FILE_ID, MODEL_FILE_PATH)

# Function to load the model with error handling
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_FILE_PATH, custom_objects=None, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        if st.button('Retry Loading Model'):
            st.experimental_rerun()

# Load the pre-trained model
model = load_model()

# Initialize the encoder
encoder = LabelEncoder()
labels = pd.read_csv(LABELS_FILE_PATH)
encoder.fit(labels['label'])

def extract_heart_sound(audio):
    fourier_transform = np.fft.fft(audio)
    heart_sound = np.abs(fourier_transform)
    return heart_sound

def preprocess_audio(file, file_format):
    temp_file_path = None
    try:
        if file_format == 'wav':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                temp_file_path = temp_file.name
        elif file_format in ['m4a', 'x-m4a']:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                temp_file_path = temp_file.name
            audio = AudioSegment.from_file(temp_file_path, format=file_format)
            temp_wav_path = temp_file_path.replace(f".{file_format}", ".wav")
            audio.export(temp_wav_path, format='wav')
            temp_file_path = temp_wav_path
        else:
            raise ValueError("Unsupported file format")

        # Load the audio file using librosa
        y, sr = librosa.load(temp_file_path, sr=None)
        audio = y / np.max(np.abs(y))
        heart_sound = extract_heart_sound(audio)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
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
        st.error(f"Error processing audio: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Streamlit interface for recording and uploading audio files
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
uploaded_file = st.file_uploader("Choose a file (WAV, M4A, X-M4A)", type=['wav', 'm4a', 'x-m4a'])

audio_data = None
file_format = None

if wav_audio_data is not None:
    audio_data = io.BytesIO(wav_audio_data)
    file_format = 'wav'
    st.audio(wav_audio_data, format='audio/wav')
elif uploaded_file is not None:
    audio_data = uploaded_file.read()
    file_format = uploaded_file.type.split('/')[1]
    st.audio(uploaded_file, format=f'audio/{file_format}')

if audio_data is not None:
    progress_text = st.empty()
    progress_text.text("Recording complete. Click the button below to get the prediction.")
    if st.button('Diagnose'):
        with st.spinner('Uploading audio and getting prediction...'):
            spectrogram = preprocess_audio(io.BytesIO(audio_data), file_format)
            if spectrogram is not None:
                y_pred = model.predict(spectrogram)
                class_probabilities = y_pred[0]
                sorted_indices = np.argsort(-class_probabilities)
                predicted_label = encoder.inverse_transform([sorted_indices[0]])[0]
                confidence_score = class_probabilities[sorted_indices[0]]
                if predicted_label == 'artifact' and confidence_score >= 0.70:
                    st.write("Artifact detected with high confidence. Please try recording again due to too many noises.")
                    st.write(f"Prediction: artifact")
                    st.write(f"Confidence: {confidence_score:.2f}")
                else:
                    if predicted_label == 'artifact':
                        predicted_label = encoder.inverse_transform([sorted_indices[1]])[0]
                        confidence_score = class_probabilities[sorted_indices[1]]
                    st.write(f"Prediction: {predicted_label}")
                    st.write(f"Confidence: {confidence_score:.2f}")
                    fig, ax = plt.subplots()
                    ax.bar(encoder.classes_, class_probabilities, color='blue')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Probability')
                    ax.set_title('Class Probabilities')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    with st.expander("Show Class Accuracies"):
                        for i, label in enumerate(encoder.classes_):
                            st.write(f"Accuracy for class '{label}': {class_probabilities[i]:.2f}")
            else:
                st.error("Failed to process the audio.")
