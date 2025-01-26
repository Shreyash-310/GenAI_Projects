import os
import streamlit as st
import time
import tempfile
import whisper
import google.generativeai as genai
from app_MoM import *


# Streamlit App code
st.title("Audio Transcription and Meeting Minutes")

transcript_generator = GenerateTranscript()
minutes_generator = GenerateMinutes()

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "m4a", "wav"])

# Initialize the session state for transcript and minutes
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'minutes' not in st.session_state:
    st.session_state.minutes = None

if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file, format=uploaded_file.type)
    # print(f"$$$ uploaded_file => {uploaded_file.name}")

    if st.button("Generate Transcript"):
        with st.spinner("Generating Transcript ...."):
            st.session_state.transcript = transcript_generator.get_transcribe(uploaded_file.name)
            time.sleep(3)
        st.success("Transcript Generated")

    if st.session_state.transcript:
        st.text_area("Transcript", value=st.session_state.transcript, height=300)

        if st.button("Generate Minutes of Meeting"):
            with st.spinner("Generating Transcript ...."):
                st.session_state.minutes = minutes_generator.summarize(st.session_state.transcript)
                time.sleep(3)
            st.success("MoM Generated")        
        
        if st.session_state.minutes:
            st.write(st.session_state.minutes)
            
            if st.button("Save MoM"):
                minutes_generator.save_MoM(st.session_state.minutes, uploaded_file.name[:-4]+'.txt')
                st.write(f"Saved the MoM file for {uploaded_file.name[:-4]+'.txt'}")