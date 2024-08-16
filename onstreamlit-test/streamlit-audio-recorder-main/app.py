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
    creds = ServiceAccountCredentials.from_json_keyfile_name('onstreamlit-test/streamlit-audio-recorder-main/heart-d9410-9a288317e3c7.json', scope)
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
    try:
        # Convert raw audio data to a numpy array
        audio = np.frombuffer(raw_audio, dtype=np.float32)
        
        # Normalize audio data
        if np.max(np.abs(audio)) != 0:
            audio = audio / np.max(np.abs(audio))
        
        # Validate the audio data
        if not np.isfinite(audio).all():
            raise ValueError("Audio buffer contains non-finite values")
        
        # Assuming the sample rate is 44.1kHz for the audio data
        sample_rate = 44100

        # Generate the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        spectrogram
