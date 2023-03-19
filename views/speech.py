import streamlit as st
import wave
import pyaudio
import pickle
from emotion_recognition import EmotionRecognizer
import pandas as pd


def load_view():
    st.subheader('How is your day?')
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output.wav"
    with open('emotion_model.pkl', 'rb') as fr:
        rec = pickle.load(fr)

    colL, colR = st.columns(2)
    session_expander = colL.expander('Session details', expanded=True)
    session_length = session_expander.number_input('Session length:',
                                     min_value=3, max_value=90, value=5)
    session_begin = session_expander.button('Start Session')
    emotion_expander = colR.expander('Predicted emotion', expanded=True)

    if session_begin:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        with st.spinner("Recording..."):
            frames = []
            for i in range(0, int(RATE / CHUNK * session_length)):
                data = stream.read(CHUNK)
                frames.append(data)
            # st.success("Done recording")
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

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
        colL.audio(WAVE_OUTPUT_FILENAME, format="audio/wav")
        st.success(f'Based on your tone, you are most likely feeling: {current_emotion[0]}')
