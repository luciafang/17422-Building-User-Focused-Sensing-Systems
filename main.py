import streamlit as st
import utl
from views import home, speech, face, recommendation
from audio_recorder_streamlit import audio_recorder

st.set_page_config(layout="wide", page_title='Emotion2Music')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()

# print(audio_bytes)

def navigation():
    route = utl.get_current_route()
    if route == "home":
        home.load_view()
    elif route == "speech analysis":
        speech.load_view()
    elif route == "expression analysis":
        face.load_view()
    elif route == "music recommendation":
        recommendation.load_view()
    elif route == None:
        home.load_view()


navigation()



