import streamlit as st
import wave #save soundbytes to .wav file 
import pickle #load the emotional classifier based on speech 
import pandas as pd
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from revChatGPT.V1 import Chatbot


def load_view():
    st.subheader('How was your day?')
    # initialize the recognizer
    # r = sr.Recognizer()
    WAVE_OUTPUT_FILENAME = "output.wav"
    with open('emotion_model.pkl', 'rb') as fr:
        rec = pickle.load(fr)
    colL, colR = st.columns(2)
    session_expander = colL.expander('Record your voice', expanded=True)
    emotion_expander = colR.expander('Predicted emotion', expanded=True)
    chatbot = Chatbot(config={
        "email": "ahsu2@andrew.cmu.edu",
        "password": "+zapubno1"
    })
    if session_expander.button('Talk to me!'):
        r = sr.Recognizer()
        m = sr.Microphone()
        # colL, colR = session_expander.columns(2)
        # set threhold level
        with m as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2
        session_expander.write("Set minimum energy threshold to {}".format(r.energy_threshold))
        # obtain audio from the microphone

        with sr.Microphone() as source:
            with colL:
                with st.spinner("Say something!"):
                    audio = r.listen(source)

        with open(WAVE_OUTPUT_FILENAME, "wb") as file:
            file.write(audio.get_wav_data())

        prompt = f"{r.recognize_google(audio)}"
        response = ""
        session_expander.write(f'User: {prompt}')

        for data in chatbot.ask(
                prompt
        ):
            response = data["message"]

        emotion_expander.write(f'ChatGPT: {response}')

    # with session_expander:
    #     audio_bytes = audio_recorder()
    # if audio_bytes:
    #
        # with wave.open(WAVE_OUTPUT_FILENAME, "wb") as audiofile:
        #     audiofile.setsampwidth(2)
        #     audiofile.setnchannels(1)
        #     audiofile.setframerate(96000)
        #     audiofile.writeframes(audio_bytes)


        session_expander.audio(WAVE_OUTPUT_FILENAME, format="audio/wav")
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
        # emotion_expander.table(emotion_df)
        # st.info(current_emotion)
        # st.warning(emotion_score)
        # st.error('ERROR')
        # st.success(f'Based on your tone, you are most likely feeling: {current_emotion[0]}')
