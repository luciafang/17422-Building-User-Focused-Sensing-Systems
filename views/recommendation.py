import streamlit as st


def load_view():
    st.subheader('Music Recommendation')


    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.write('')
        st.markdown('<span style="color:grey">{}</span>'.format('Emotion2Music is developed by Lucia Fang'),
                    unsafe_allow_html=True)