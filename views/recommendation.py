import streamlit as st
# import webbrowser
import numpy as np

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)


def load_view():
    st.subheader('Music Recommendation')
    btn = st.button("Recommend me songs")
    face_emotion = np.load('./facial_emotion.npy')
    speech_emotion = np.load('./speech_emotion.npy')
    final_emotion = None
    # st.write(face_emotion, speech_emotion)
    if speech_emotion in face_emotion:
        # st.write(speech_emotion)
        final_emotion = speech_emotion[0]
    if btn:
        if not (final_emotion):
            st.warning("Please let me capture your emotion first")
            st.session_state["run"] = "true"
        else:
            url = f"https://open.spotify.com/search/{final_emotion} chinese playlist"
            nav_to(url)
            # webbrowser.open_new_tab(f"https://open.spotify.com/search/{final_emotion} chinese playlist")
            # np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format('Emotion2Music is developed by Lucia Fang'),
                    unsafe_allow_html=True)