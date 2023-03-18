import streamlit as st
from audio_recorder_streamlit import audio_recorder


picture = st.camera_input("Take a picture")
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

if picture:
    st.image(picture)