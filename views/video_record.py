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
from datetime import datetime

from deepface import DeepFace
from deepface.commons import functions


def analyze_emotions(video_file, f_container):
    colors_list = ['firebrick', 'peru', 'gold', 'olivedrab', 'royalblue', 'indigo', 'hotpink']
    chatbot = Chatbot(config={"access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJhaHN1MkBhbmRyZXcuY211LmVkdSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InVzZXJfaWQiOiJ1c2VyLVF1QmtPRDFxVDRQc0dRRklmQjZtZjB5RSJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiYXV0aDB8NjQxZGU4YTYzOTU5M2Q4NGU0NjQ5YzRlIiwiYXVkIjpbImh0dHBzOi8vYXBpLm9wZW5haS5jb20vdjEiLCJodHRwczovL29wZW5haS5vcGVuYWkuYXV0aDBhcHAuY29tL3VzZXJpbmZvIl0sImlhdCI6MTY5MDM5Nzk4NiwiZXhwIjoxNjkxNjA3NTg2LCJhenAiOiJUZEpJY2JlMTZXb1RIdE45NW55eXdoNUU0eU9vNkl0RyIsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyJ9.KBfBVRS8KoVfYKbRs7EaMy72_m8SUIBfWESaTy19x-ADI0uu3Bcj8qWJPVKRrLk5h-kgrACjFIYHrr72apzObKb9z1C_Bcjr08pUVZ-xSsl6mZOXpDDxbExWomx5tRy0ycE8ySOczEORbNn75kzEH9jR1ETlDBNyBcdJ81AdaFjL4nfJTyk7u8unlNyJoDV5qjExFv-5EqqEn2ThB5hPF8WdiNimStUWZYXm6xDhE38rqV0rrggHGKbeVm3cKcLwriVRzPdqBOKBm0w4KPgyWG-Er0rdw0eNsh7oF8CyC764VTVVhIrtOMOWfRz3Lzing7VIaIQPxCRPk-n7U8Kh5w"})
    emotions_list_default = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'fear', 'surprise']
    lang_selected = f_container.radio('select language',
                                      ('English', '中文',
                                       'Español', 'Français'),
                                      horizontal=True, key=f'language')
    output_csv = str.join('',
                          (video_file.rpartition('.mp4')[0], '.csv'))
    output_txt = str.join('',
                          (video_file.rpartition('.mp4')[0], '.txt'))
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

    try:
        final_emo_df = pd.read_csv(output_csv)
        fig = go.Figure(data=[go.Pie(labels=final_emo_df["emotion"],
                                     values=final_emo_df["counts"], hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='percent',
                          textfont_size=16,
                          marker=dict(colors=final_emo_df["colors"],
                                      line=dict(color='#000000', width=1)))
        f_container.plotly_chart(fig, use_container_width=True)
        file = open(output_txt, "rb")
        lines = list()
        for line in file.readlines():
            lines.append(line.rstrip().decode("utf-8"))
        file.close()
        for line in lines:
            f_container.write(line)
    except:
        # with f_container:
        #     with st.form(key=f'form_{vid_counter}', clear_on_submit=True):
        # date_col, time_col = f_container.columns(2)
        # d = date_col.date_input("input date", key=f'date_input_{vid_counter}')
        # t = time_col.time_input("input time", key=f'time_input_{vid_counter}')
        current_dateTime = datetime.now()
        meta_dict = {'datetime': current_dateTime,
                     'video': video_file}

                # submitted = st.form_submit_button("submit")
                # if submitted:
                #     os.rename(video_file, os.path.join(RECORD_DIR, str.join('', (str(d), '.mp4'))))
        # st.button('')
        # files = fnmatch.filter((f for f in os.listdir(RECORD_DIR)), 'd*.mp4')
        # f_container.write(files)
        if f_container.button('analyze this video', key=f'analyzebutton'):
            output_npy = str.join('',
                                   (video_file.rpartition('.mp4')[0], '.npy'))
            np.save(output_npy, meta_dict)
            # files = fnmatch.filter((f for f in os.listdir(RECORD_DIR)), 'd*.mp4')
            # os.rename(video_file, os.path.join(RECORD_DIR, str.join('', (str(d), '.mp4'))))

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
            st.write(user_text)

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
                if frame_counter % int(framerate / 2) == 0:
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
                my_bar.progress((frame_counter) / length)
            final_emo_counts = {'emotion': emotions_list,
                                'colors': colors_list,
                                'counts': np.hstack([len(emotion_dict[emo]) for emo in emotions_list_default])}

            final_emo_df = pd.DataFrame(data=final_emo_counts)
            final_emo_df.to_csv(output_csv)
            fig = go.Figure(data=[go.Pie(labels=final_emo_df["emotion"],
                                         values=final_emo_df["counts"], hole=.4)])
            fig.update_traces(hoverinfo='label+percent',
                              textinfo='percent',
                              textfont_size=16,
                              marker=dict(colors=final_emo_df["colors"],
                                          line=dict(color='#000000', width=1)))
            f_container.plotly_chart(fig, use_container_width=True)
            # select top 3
            # st.write(final_emo_df)
            final_emo_df = final_emo_df.sort_values(
                by=["counts"], ascending=False
            ).reset_index(drop=True)
            current_emotion = []
            for index, instance in final_emo_df.iterrows():
                if index < 3:  # limited to Top3
                    current_emotion.append(instance["emotion"])
            # st.write(current_emotion)
        # st.write(default_prompt)
        prompt = f"{user_text}. {default_prompt_1} {', '.join(current_emotion)}, {default_prompt}"
        # st.write(prompt)
        # revchatGPT
        with f_container:
            with st.spinner("Inquiring large language model's response..."):
                response = ""
                # emotion_expander.write(f'User: {user_text}'
                for data in chatbot.ask(
                        prompt
                ):
                    response = data["message"]

        f_container.write(f'ChatGPT: {response}')
        with open(output_txt, 'w') as f:
            f.write(f'{st.session_state.user}: {user_text}')
            f.write('\n')
            f.write(f'ChatGPT: {response}')


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
    try:
        video_filename = temporary_location
        vid_expander = st.expander('', expanded=True)
        vid_placeholder, analysis_placeholder = vid_expander.columns(2)
        vid_placeholder.video(video_filename)
        analyze_emotions(video_filename, analysis_placeholder)
        # if analysis_placeholder.button('Delete video',
        #                                key=f'delete'):
        #     try:
        #         os.remove(video_filename)
        #         os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
        #                                 '.wav')))
        #         os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
        #                                 '.csv')))
        #         os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
        #                                 '.txt')))
        #         os.remove(str.join('', (video_filename.rpartition('.mp4')[0],
        #                                 '.npy')))
        #         st.experimental_rerun()
        #     except:
        #         pass
    except:
        pass


    bottom_cont = st.container()
    with bottom_cont:
        st.markdown("""---""")
        st.markdown(f" <h1 style='text-align: left; color: #FF6A95; font-size:16px; "
                    f"font-family:Arial; font-weight:normal'>"
                    f"EmoSense is developed by Lucia Fang</h1> "
                    , unsafe_allow_html=True)


if __name__ == "__main__":
    load_view()
