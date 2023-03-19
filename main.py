import streamlit as st
from views import home, speech, face, recommendation
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", page_title='Emotion2Music', page_icon='😊')

selected = option_menu(None, ["home", "speech analysis", "expression analysis", 'music recommendation'],
                       icons=['house', 'soundwave', "emoji-laughing", 'file-earmark-music'],
                       menu_icon="cast", default_index=0, orientation="horizontal",
                       styles={
                           "container": {"padding": "0!important", "background-color": "#fafafa"},
                           "icon": {"color": "black", "font-size": "20px"},
                           "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px",
                                        "--hover-color": "#eee"},
                           "nav-link-selected": {"font-size": "16px", "font-weight": "normal",},
                       }
                       )


def navigation():
    if selected == "home":
        home.load_view()
    elif selected == "speech analysis":
        speech.load_view()
    elif selected == "expression analysis":
        face.load_view()
    elif selected == "music recommendation":
        recommendation.load_view()
    elif selected == None:
        home.load_view()


navigation()
