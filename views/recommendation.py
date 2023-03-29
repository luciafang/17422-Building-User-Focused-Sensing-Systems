import streamlit as st
import webbrowser
import numpy as np


def load_view():
    st.subheader('Music Recommendation')
    btn = st.button("Recommend me songs")
    face_emotion = np.load('./facial_emotion.npy')
    speech_emotion = np.load('./speech_emotion.npy')
    final_emotion = None
    if speech_emotion in face_emotion:
        st.write(speech_emotion)
        # final_emotion = speech_emotion
    if btn:
        if not (final_emotion):
            st.warning("Please let me capture your emotion first")
            st.session_state["run"] = "true"
        else:
            webbrowser.open(f"https://open.spotify.com/search/{final_emotion}+song")
            # np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format('Emotion2Music is developed by Lucia Fang'),
                    unsafe_allow_html=True)