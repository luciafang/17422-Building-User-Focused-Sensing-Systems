import streamlit as st
from bokeh.models import CustomJS
from bokeh.models.widgets import Button
from revChatGPT.V1 import Chatbot
from streamlit_bokeh_events import streamlit_bokeh_events


def load_view():
    st.subheader('How was your day?')
    default_prompt = "Of the following emotions, which am I most likely feeling right now: " \
                     "angry, fear, neutral, sad, disgust, happy, surprise. Pick only one."
    emotions_list = ['angry', 'fear', 'neutral', 'sad', 'disgust', 'happy', 'surprise']

    # with open('emotion_model.pkl', 'rb') as fr:
    #     rec = pickle.load(fr)
    colL, colR = st.columns(2)
    session_expander = colL.expander('Record your voice', expanded=True)
    emotion_expander = colR.expander('Predicted emotion', expanded=True)

    chatbot = Chatbot(config={
        "email": "ahsu2@andrew.cmu.edu",
        "password": "+zapubno1"
    })

    with session_expander:
        stt_button = Button(label="Talk to me!", width=150, button_type="primary")

        stt_button.js_on_event("button_click", CustomJS(code="""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.onresult = function (e) {
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) {
                        value += e.results[i][0].transcript;
                    }
                }
                if ( value != "") {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }
            }
            recognition.start();
            """))

        result = streamlit_bokeh_events(
            stt_button,
            events="GET_TEXT",
            key="listen",
            refresh_on_update=False,
            override_height=45,
            debounce_time=0)

        try:
            if result:
                if "GET_TEXT" in result:
                    prompt = f"{result.get('GET_TEXT')}. {default_prompt}"
                    response = ""
                    session_expander.write(f'User: {prompt}')
                    for data in chatbot.ask(
                            prompt
                    ):
                        response = data["message"]
                    emotion_expander.write(f'ChatGPT: {response}')
                    for emotion in emotions_list:
                        if emotion in response:
                            current_emotion = emotion
                    emotion_expander.success(f'Based on your statement, you are most likely feeling: {current_emotion}')
        except:
            pass


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
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format('Emotion2Music is developed by Lucia Fang'),
                    unsafe_allow_html=True)
