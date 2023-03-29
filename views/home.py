import streamlit as st
import numpy as np


def load_view():
    st.header('Welcome to Emotion2Music')
    st.subheader('A music therapy virtual assistant')
    desc_box = st.expander('Description', expanded=True)
    desc_box.write("""
    Welcome to our interactive emotional analysis assistant!\n
    Here, we take a holistic approach to understanding your emotions. \n
    Our assistant will start by asking you about your day. 
    From your description of your day, it will predict and analyze your emotion using ChatGPT. \n
    But that's not all - our assistant takes it one step further. 
    We also require a picture of how you're truly feeling about your day, 
    analyzing your top three emotions from your facial expression.
    With all this data, our assistant is then able to recommend a personalized music playlist just for you. 
    So sit back, relax, and let our assistant help you understand and manage your emotions like never before.
    # """)
    # desc_box.image("https://static.streamlit.io/examples/dice.jpg")
    emotion = np.load('./speech_emotion.npy')
    st.write(emotion)
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format('Emotion2Music is developed by Lucia Fang'),
                    unsafe_allow_html=True)