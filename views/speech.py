import streamlit as st
from revChatGPT.V1 import Chatbot
from audio_recorder_streamlit import audio_recorder
import wave
import speech_recognition as sr
import numpy as np
import soundfile as sf
import io


def load_view():
    # st.subheader('How was your day?')
    default_prompt = "Of the following emotions, which am I most likely feeling right now: " \
                     "angry, fear, neutral, sad, disgust, happy, surprise. Pick only one."
    emotions_list = ['angry', 'fear', 'neutral', 'sad', 'disgust', 'happy', 'surprise']
    color_list = ['red', '', '', 'blue', '', 'green', '']
    WAVE_OUTPUT_FILENAME = './output.wav'
    chatbot = Chatbot(config={
        "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJhaHN1MkBhbmRyZXcuY211LmVkdSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLVF1QmtPRDFxVDRQc0dRRklmQjZtZjB5RSJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjQxZGU4YTYzOTU5M2Q4NGU0NjQ5YzRlIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY4MDkyNTgyNiwiZXhwIjoxNjgyMTM1NDI2LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9mZmxpbmVfYWNjZXNzIn0.gTPI0wtQH5faXU4SiqWChWMp-ftmbsyXFTU74sb96viY7VdWHn3Aj5QRdVVbUiSse-fBHFq5fWh4xSMTgZbnqzju1eZQJQRYJiWjssmutPMrVYxk5vWq1IpbD5DrwBXVRnZSASt-6yAxv8U2EKD8LPB8-970KFnrsBJ6hcKDfVnbmsh-Wg-SHDM72FuZc1533d-TL5R68JBnJdO_A_UxEzhbn32UzRH3I77ZO5r0xBqMa29dFh1S6flzRA0ngYi5zPygxn_gJu-8rlg2s19Ttn4ztt3KNPadzNMoXwenSGewHKwnxqpmnrMCVSYmy0nsHCffPdjHHbxF3rrvaKbz1g"
    })

    _, colL,  colR = st.columns([1, 2, 2])
    record_expander = colR.expander("", expanded=True)
    # colL, colM, colR = st.columns([5, 3, 5])
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colR.write('')
    colL.image('./images/speechpage_logo.png')

    emotion_expander = st.expander('Predicted emotion', expanded=True)
    with colR:
        #tutorial 
        audio_bytes = audio_recorder(text=""
                                     ,
                                     # recording_color="#e8b62c",
                                     # neutral_color="#6aa36f",
                                     # icon_name="user",
                                     icon_size="8x", )
        #save audio file 
        if audio_bytes:
            with wave.open(WAVE_OUTPUT_FILENAME, "wb") as audiofile:
                audiofile.setsampwidth(2)
                audiofile.setnchannels(2)
                audiofile.setframerate(44100)
                audiofile.writeframes(audio_bytes)

    #Speech to Text ML Model 
    # https://www.geeksforgeeks.org/python-convert-speech-to-text-and-text-to-speech/
    r = sr.Recognizer()
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        # listen for the data (load audio to memory)
        r.adjust_for_ambient_noise(source)
        audio_data = r.listen(source)
        # recognize (convert from speech to text)
        user_text = r.recognize_google(audio_data)
    prompt = f"{user_text}. {default_prompt}"
    # revchatGPT
    response = ""
    # emotion_expander.write(f'User: {user_text}'
    for data in chatbot.ask(
            prompt
    ):
        response = data["message"]
    emotion_expander.write(f'ChatGPT: {response}')
    for e, emotion in enumerate(emotions_list):
        if emotion in response:
            current_emotion = emotion
            coloring = color_list[e]
    try:
        # st.write(coloring)
        emotion_expander.markdown(f'Based on your statement, you are most likely feeling: :{coloring}[{current_emotion}]')
    except:
        emotion_expander.warning(f"No emotion detected.")
    # st.write(current_emotion)
    np.save('./speech_emotion.npy', [current_emotion])

    # WAVE_OUTPUT_FILENAME = './output.wav'
    # with session_expander:
    #     audio_bytes = audio_recorder(text="   ",
    #                                  # recording_color="#e8b62c",
    #                                  # neutral_color="#6aa36f",
    #                                  # icon_name="user",
    #                                  icon_size="6x",)
    # if audio_bytes:
    #     with wave.open(WAVE_OUTPUT_FILENAME, "wb") as audiofile:
    #         audiofile.setsampwidth(2)
    #         audiofile.setnchannels(2)
    #         audiofile.setframerate(44100)
    #         audiofile.writeframes(audio_bytes)
    # r = sr.Recognizer()
    # with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
    #     # listen for the data (load audio to memory)
    #     r.adjust_for_ambient_noise(source)
    #     audio_data = r.listen(source)
    #     # recognize (convert from speech to text)
    #     user_text = r.recognize_google(audio_data)
    # prompt = f"{user_text}. {default_prompt}"
    # response = ""
    # session_expander.write(f'User: {user_text}')
    # for data in chatbot.ask(
    #         prompt
    # ):
    #     response = data["message"]
    # emotion_expander.write(f'ChatGPT: {response}')
    # for emotion in emotions_list:
    #     if emotion in response:
    #         current_emotion = emotion
    # try:
    #     emotion_expander.success(f'Based on your statement, you are most likely feeling: {current_emotion}')
    # except:
    #     emotion_expander.warning(f"No emotion detected.")



    # if session_expander.button('Talk to me!'):
    #     r = sr.Recognizer()
    #     m = sr.Microphone()
    #     # colL, colR = session_expander.columns(2)
    #     # set threhold level
    #     with m as source:
    #         r.adjust_for_ambient_noise(source)
    #         r.pause_threshold = 2
    #     session_expander.write("Set minimum energy threshold to {}".format(r.energy_threshold))
    #     # obtain audio from the microphone
    #
    #     with sr.Microphone() as source:
    #         with colL:
    #             with st.spinner("Say something!"):
    #                 audio = r.listen(source)
    #
    #     with open(WAVE_OUTPUT_FILENAME, "wb") as file:
    #         file.write(audio.get_wav_data())


    # # prompt = f"{r.recognize_google(audio)}"
    # response = ""
    # session_expander.write(f'User: {prompt}')
    #
    # for data in chatbot.ask(
    #         prompt
    # ):
    #     response = data["message"]
    #
    # emotion_expander.write(f'ChatGPT: {response}')

    # with session_expander:
    #     audio_bytes = audio_recorder()
    # if audio_bytes:
    #
        # with wave.open(WAVE_OUTPUT_FILENAME, "wb") as audiofile:
        #     audiofile.setsampwidth(2)
        #     audiofile.setnchannels(1)
        #     audiofile.setframerate(96000)
        #     audiofile.writeframes(audio_bytes)


    # session_expander.audio(WAVE_OUTPUT_FILENAME, format="audio/wav")
    # emotions_by_speech = rec.predict_proba(WAVE_OUTPUT_FILENAME)
    # emotion_df = pd.DataFrame(
    #     emotions_by_speech.items(), columns=["emotion", "score"]
    # )
    # emotion_df = emotion_df.sort_values(
    #     by=["score"], ascending=False
    # ).reset_index(drop=True)
    # current_emotion = []
    # emotion_score = []
    # for index, instance in emotion_df.iterrows():
    #     if index < 3:
    #         current_emotion.append(instance["emotion"])
    #         emotion_score.append(instance["score"])
        # emotion_expander.table(emotion_df)
        # st.info(current_emotion)
        # st.warning(emotion_score)
        # st.error('ERROR')
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"Emotion2Music is developed by Lucia Fang</h1> "
                    , unsafe_allow_html=True)
