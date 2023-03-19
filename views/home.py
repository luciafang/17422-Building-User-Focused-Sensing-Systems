import streamlit as st

def load_view():
    st.header('Welcome to Emotion2Music')
    st.subheader('A music therapy virtual assistant')
    desc_box = st.expander('Description', expanded=True)
    desc_box.write("""
    This assistant will analyze your emotions using a holistic approach. \n
    First, the assistant will ask about your day, of which it can be in any language. 
    Top 3 predicted emotions are analyzed from how you speak. \n
    Second, upon your answers, the assistant will require an picture of how you really feel about your day. 
    Top 3 predicted emotions are analyzed from your facial expression. \n
    Finally, the assistant will then recommend a music playlist for you.
    """)
    # desc_box.image("https://static.streamlit.io/examples/dice.jpg")
