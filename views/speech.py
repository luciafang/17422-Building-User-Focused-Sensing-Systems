import streamlit as st
import wave
import pyaudio
import pickle
from emotion_recognition import EmotionRecognizer
import pandas as pd
from audio_recorder_streamlit import audio_recorder


def load_view():
    st.subheader('How was your day?')
    WAVE_OUTPUT_FILENAME = "output.wav"
    with open('emotion_model.pkl', 'rb') as fr:
        rec = pickle.load(fr)
    colL, colR = st.columns(2)
    session_expander = colL.expander('Record your voice', expanded=True)
    emotion_expander = colR.expander('Predicted emotion', expanded=True)
    with session_expander:
        audio_bytes = audio_recorder(sample_rate=96000)
    if audio_bytes:
        session_expander.audio(audio_bytes, format="audio/wav")
        with wave.open(WAVE_OUTPUT_FILENAME, "wb") as audiofile:
            audiofile.setsampwidth(2)
            audiofile.setnchannels(1)
            audiofile.setframerate(96000)
            audiofile.writeframes(audio_bytes)

        emotions_by_speech = rec.predict_proba(WAVE_OUTPUT_FILENAME)
        emotion_df = pd.DataFrame(
            emotions_by_speech.items(), columns=["emotion", "score"]
        )
        emotion_df = emotion_df.sort_values(
            by=["score"], ascending=False
        ).reset_index(drop=True)
        current_emotion = []
        emotion_score = []
        for index, instance in emotion_df.iterrows():
            if index < 3:
                current_emotion.append(instance["emotion"])
                emotion_score.append(instance["score"])
        emotion_expander.table(emotion_df)
        st.success(f'Based on your tone, you are most likely feeling: {current_emotion[0]}')
