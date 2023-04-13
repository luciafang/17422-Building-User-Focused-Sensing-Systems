import streamlit as st


def load_view():
    st.markdown(f" <h1 style='text-align: left; color: #000000; font-size:30px; "
                f"font-family:Avenir; font-weight:normal;'>Welcome to EmoSense </h1> "
                , unsafe_allow_html=True)
    # st.header('Welcome to Emotion2Music')
    # st.subheader('A music therapy virtual assistant')
    desc_box = st.expander('Description', expanded=True)
    desc_box.write("""
    Introducing "EmoSense", the cutting-edge no code website for emotion analysis.\n 
    With our platform, you can easily record videos of individuals talking about their day,
    extract images for facial expression prediction, and extract audio for semantic expression prediction. 
    By combining both data sources, EmotionSense offers a holistic emotional classification 
    that provides deeper insights into an individual's emotional state. \n
    Our platform offers detailed and rigorous analysis, 
    enabling you to identify emotional trends and patterns that may be affecting the individual's well-being. 
    With our reactive visualization design, you can explore and interpret the data in real-time, 
    gaining insights that would be impossible to identify otherwise.\n
    And for those seeking professional help, 
    EmotionSense offers the ability to download previous recordings for therapists to listen to, 
    providing an even deeper level of insight and understanding. 
    So why wait? Sign up for EmotionSense today and take the first step towards a deeper understanding of human emotions.
    Here, we take a holistic approach to understanding your emotions. 
    """)
    # desc_box.image("https://static.streamlit.io/examples/dice.jpg")
    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown(f" <h1 style='text-align: left; color: gray; font-size:16px; "
                    f"font-family:Avenir; font-weight:normal'>"
                    f"Emotion2Music is developed by Lucia Fang</h1> "
                    , unsafe_allow_html=True)