import glob
import io
import time
from pathlib import Path

from revChatGPT.V1 import Chatbot
import cv2  # rgb. bgr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import speech_recognition as sr
import streamlit as st
from moviepy.editor import *

from deepface import DeepFace
from deepface.commons import functions


def analyze_emotions(video_file, f_container, vid_counter):
    colors_list = ['firebrick', 'peru', 'gold', 'olivedrab', 'royalblue', 'indigo', 'hotpink']
    chatbot = Chatbot(config={
        "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJhaHN1MkBhbmRyZXcuY211LmVkdSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLVF1QmtPRDFxVDRQc0dRRklmQjZtZjB5RSJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjQxZGU4YTYzOTU5M2Q4NGU0NjQ5YzRlIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY4MDkyNTY4OCwiZXhwIjoxNjgyMTM1Mjg4LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9mZmxpbmVfYWNjZXNzIn0.xZpIfzn2H_-dEVFidxuKL6-oiXV53ycskvVqFF7GsLIBuVaP1VjYbN_QfUVt0LbN1kGi0qmb2DHLCD6mBfv_xtd5ojQs3kLLJNye81Cq0NF5bHiyYXrCsP2R2Rw49g9RHPeJtfptE-0BY30KEjNJ2HglWpcOupbiPCcCtQsNqTcc5jJlMhWhBLW4Xjm3EomoWKUdST8zpvruz8nuoSc3UHR3fiH1UcyTNomjwTTr0KIwZi6gAsJGMd83Vkn-E-5Ba0hs_TEZRZ075ML4ccZUJUDD2mmfh-WfvQccN7USeevFjQkqGSsjngxrKpL4w8_TMyH53A7HmQvL0kfM-ibl8g"
    })
    emotions_list_default = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'fear', 'surprise']
    lang_selected = f_container.radio('select language',
                                      ('English', '中文',
                                       'Español', 'Français'),
                                      horizontal=True, key=f'language_{vid_counter}')
    if lang_selected == 'English':
        lang = 'en-US'
        emotions_list = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'fear', 'surprise']
        default_prompt_1 = "Of the following emotions: "
        default_prompt = "which am I most likely feeling right now. " \
                         "Pick only one."
    elif lang_selected == '中文':
        lang = 'zh-CN'
        emotions_list = ['愤怒', '厌恶', '快乐', '中性', '悲伤', '恐惧', '惊讶']
        default_prompt_1 = "下列情绪之一："
        default_prompt = "我现在最有可能感觉到的是什么。只选一个。"
    elif lang_selected == 'Español':
        lang = 'es-MX'
        emotions_list = ['enojado', 'asco', 'feliz', 'neutral', 'triste', 'miedo', 'sorpresa']
        default_prompt_1 = "De las siguientes emociones:"
        default_prompt = "que es lo más probable que estoy sintiendo en este momento. Elige solo uno."
    elif lang_selected == 'Français':
        lang = 'fr-FR'
        emotions_list = ['en colère', 'dégoût', 'heureux', 'neutre', 'triste', 'peur', 'surprise']
        default_prompt_1 = "Parmi les émotions suivantes :"
        default_prompt = "ce que je ressens très probablement en ce moment. Choisissez-en un seul."
    if f_container.button('analyze this video', key=f'analyzebutton_{vid_counter}'):
        output_file = str.join('',
                               (video_file.rpartition('.mp4')[0], '.wav'))
        sound = AudioFileClip(video_file)
        sound.write_audiofile(output_file, codec='pcm_s16le')
        # f_container.audio(output_file)
        r = sr.Recognizer()
        with sr.AudioFile(output_file) as source:
            # listen for the data (load audio to memory)
            # r.adjust_for_ambient_noise(source)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            user_text = r.recognize_google(audio_data, language=lang)


        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        framerate = cap.get(cv2.CAP_PROP_FPS)
        emotion_dict = {key: [] for key in emotions_list_default}
        my_bar = f_container.progress(0)
        frame_counter = 0
        while True:
            _, img = cap.read()
            if img is None:
                break
            # analyze every half a second
            if frame_counter % int(framerate/2) == 0:
                model_name = "VGG-Face"
                detector_backend = "opencv"
                distance_metric = "cosine"
                enable_face_analysis = True

                # global variables
                pivot_img_size = 112  # face recognition result image
                enable_emotion = True
                # ------------------------
                # find custom values for this input set
                target_size = functions.find_target_size(model_name=model_name)
                # ------------------------
                # build models once to store them in the memory
                # otherwise, they will be built after cam started and this will cause delays
                DeepFace.build_model(model_name=model_name)
                if enable_face_analysis:
                    DeepFace.build_model(model_name="Emotion")
                # -----------------------
                # visualization
                freeze = False
                face_detected = False
                face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
                freezed_frame = 0
                tic = time.time()
                raw_img = img.copy()
                if freeze == False:
                    try:
                        # just extract the regions to highlight in webcam
                        face_objs = DeepFace.extract_faces(
                            img_path=img,
                            target_size=target_size,
                            detector_backend=detector_backend,
                            enforce_detection=False,
                        )
                        faces = []
                        for face_obj in face_objs:
                            facial_area = face_obj["facial_area"]
                            faces.append(
                                (
                                    facial_area["x"],
                                    facial_area["y"],
                                    facial_area["w"],
                                    facial_area["h"],
                                )
                            )
                    except:  # to avoid exception if no face detected
                        faces = []

                    if len(faces) == 0:
                        face_included_frames = 0
                else:
                    faces = []
                detected_faces = []
                face_index = 0
                for x, y, w, h in faces:
                    if w > 100:  # discard small detected faces

                        face_detected = True
                        if face_index == 0:
                            face_included_frames = (
                                    face_included_frames + 1
                            )  # increase frame for a single face

                        detected_face = img[int(y): int(y + h), int(x): int(x + w)]  # crop detected face

                        # -------------------------------------

                        detected_faces.append((x, y, w, h))
                        face_index = face_index + 1

                        # -------------------------------------
                base_img = raw_img.copy()
                detected_faces_final = detected_faces.copy()
                for detected_face in detected_faces_final:
                    x = detected_face[0]
                    y = detected_face[1]
                    w = detected_face[2]
                    h = detected_face[3]

                    # -------------------------------
                    # extract detected face
                    custom_face = base_img[y: y + h, x: x + w]
                    # -------------------------------
                    # facial attribute analysis

                    if enable_face_analysis == True:

                        demographies = DeepFace.analyze(
                            img_path=custom_face,
                            detector_backend=detector_backend,
                            enforce_detection=False,
                            silent=True,
                        )

                        if len(demographies) > 0:
                            # directly access 1st face cos img is extracted already
                            demography = demographies[0]

                            if enable_emotion:
                                emotion = demography["emotion"]
                                emotion_df = pd.DataFrame(
                                    emotion.items(), columns=["emotion", "score"]
                                )
                                emotion_df = emotion_df.sort_values(
                                    by=["score"], ascending=False
                                ).reset_index(drop=True)
                                current_emotion = []
                                emotion_score = []
                                for index, instance in emotion_df.iterrows():
                                    if index < 3:  # limited to Top3
                                        current_emotion.append(instance["emotion"])
                                        # emotion_label = f"{current_emotion} "
                                        emotion_score.append(instance["score"] / 100)
                    emotion_dict[current_emotion[0]].append(frame_counter)
            frame_counter += 1
            # st.write(frame_counter/length)
            my_bar.progress((frame_counter) / length)
        final_emo_counts = {'emotion': emotions_list,
                            'colors': colors_list,
                            'counts': np.hstack([len(emotion_dict[emo]) for emo in emotions_list_default])}

        final_emo_df = pd.DataFrame(data=final_emo_counts)
        fig = go.Figure(data=[go.Pie(labels=final_emo_df["emotion"],
                                     values=final_emo_df["counts"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='percent',
                          textfont_size=16,
                          marker=dict(colors=final_emo_df["colors"],
                                      line=dict(color='#000000', width=1)))
        f_container.plotly_chart(fig, use_container_width=True)
        # select top 3
        final_emo_df = final_emo_df.sort_values(
            by=["counts"], ascending=False
        ).reset_index(drop=True)
        current_emotion = []
        for index, instance in final_emo_df.iterrows():
            if index < 3:  # limited to Top3
                current_emotion.append(instance["emotion"])

        prompt = f"{user_text}. {default_prompt_1} {', '.join(current_emotion)}, {default_prompt}"
        # f_container.write(prompt)

        # revchatGPT
        with st.spinner("Inquiring large language model's response..."):
            response = ""
            # emotion_expander.write(f'User: {user_text}'
            for data in chatbot.ask(
                    prompt
            ):
                response = data["message"]
        f_container.write(f'ChatGPT: {response}')


def load_view():
    HERE = Path(__file__).resolve().parent.parent
    st.markdown(f" <h1 style='text-align: left; color: #FF6A95; font-size:30px; "
                f"font-family:Arial; font-weight:normal;'>Analyze Emotions, Upload Video Now! </h1> "
                , unsafe_allow_html=True)
    st.write('')
    RECORD_DIR = os.path.join(HERE, f"./records/{st.session_state.user}")
    os.makedirs(RECORD_DIR, exist_ok=True)
    st.write("---")
    example_video = st.file_uploader('',
                                     accept_multiple_files=False,
                                     label_visibility='collapsed',
                                     key='video')
    # copy video to local, only when csv and mp4/avi are uploaded
    if example_video is not None:
        if os.path.exists(example_video.name):
            out_file = os.path.join(RECORD_DIR,
                                    f"./{str.join('', (example_video.name.rpartition('.')[0], '.mp4'))}")
            temporary_location = f'{out_file}'
        else:
            g = io.BytesIO(example_video.read())  # BytesIO Object
            out_file = os.path.join(RECORD_DIR,
                                    f"./{str.join('', (example_video.name.rpartition('.')[0], '.mp4'))}")
            temporary_location = f'{out_file}'
            with open(temporary_location, 'wb') as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            out.close()

    vid_counter = 0
    col1, col2, col3 = st.columns(3)
    for video_filename in glob.glob(f"{RECORD_DIR}/*.mp4"):
        if vid_counter % 3 == 0:
            with col1:
                col1_expander = st.expander('', expanded=True)
                col1_expander.video(video_filename)
                analyze_emotions(video_filename, col1_expander, vid_counter)
                if col1_expander.button('Delete video',
                                        key=f'delete_{vid_counter}'):
                    os.remove(video_filename)
                    os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
                                            '.wav')))
                    st.experimental_rerun()
        elif vid_counter % 3 == 1:
            with col2:
                col2_expander = st.expander('', expanded=True)
                col2_expander.video(video_filename)
                analyze_emotions(video_filename, col2_expander, vid_counter)
                if col2_expander.button('Delete video',
                                        key=f'delete_{vid_counter}'):
                    os.remove(video_filename)
                    os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
                                            '.wav')))
                    st.experimental_rerun()
        elif vid_counter % 3 == 2:
            with col3:
                col3_expander = st.expander('', expanded=True)
                col3_expander.video(video_filename)
                analyze_emotions(video_filename, col3_expander, vid_counter)
                if col3_expander.button('Delete video',
                                        key=f'delete_{vid_counter}'):
                    os.remove(video_filename)
                    os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
                                            '.wav')))
                    st.experimental_rerun()
        vid_counter += 1

    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: #FF6A95; font-size:16px; "
                    f"font-family:Arial; font-weight:normal'>"
                    f"EmoSense is developed by Lucia Fang</h1> "
                    , unsafe_allow_html=True)

if __name__ == "__main__":
    load_view()
