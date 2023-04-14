import uuid
from pathlib import Path
import datetime
import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import streamlit as st
import os
import time
import pandas as pd
import cv2  # rgb. bgr
import numpy as np
from deepface import DeepFace
from deepface.commons import functions
import matplotlib.pyplot as plt


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def load_view():
    emotions_list = ['angry', 'fear', 'neutral', 'sad', 'disgust', 'happy', 'surprise']
    RECORD_DIR = Path(str.join('', ("./records", f"/{st.session_state.user}")))
    RECORD_DIR.mkdir(exist_ok=True)
    currentDateAndTime = datetime.datetime.now()
    prefix = currentDateAndTime.strftime("%Y-%m-%d_%H-%M")
    out_file = RECORD_DIR / f"{prefix}.mp4"

    # def in_recorder_factory() -> MediaRecorder:
    #     return MediaRecorder(
    #         str(in_file), format="mp4"
    #     )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331

    def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(str(out_file), format="mp4")

    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        video_frame_callback=video_frame_callback,
        # in_recorder_factory=in_recorder_factory,
        out_recorder_factory=out_recorder_factory,
    )
    # st.experimental_rerun()
    # emotion_expander = st.empty()
    chosen_recording = st.selectbox('select a previous recording', os.listdir(RECORD_DIR), index=0)
    chosen_video_file = os.path.join(RECORD_DIR, chosen_recording)
    if chosen_recording.endswith('.mp4'):
        st.video(chosen_video_file)
        if st.button('delete'):
            os.remove(chosen_video_file)
            st.experimental_rerun()

        with open(chosen_video_file, "rb") as f:
            st.download_button(
                "Download the recorded video", f, chosen_recording
            )

        if st.button('analyze this video'):
            cap = cv2.VideoCapture(chosen_video_file)  # webcam
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # st.write(np.arange(int(length)))
            emotion_dict = {key: [] for key in emotions_list}
            # st.write(emotion_dict)
            my_bar = st.progress(0)
            frame_counter = 0
            while True:
                _, img = cap.read()
                if img is None:
                    # emotion_dfs.append(0)
                    break
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
                    # st.write(current_emotion[0])
                    # emotion_expander.table(emotion_df)  # place a table on the right
                    # st.success(
                    #     f'Based on your expression, you are most likely feeling: {current_emotion[0]}')  # use the highest emotin score
                    # np.save('./facial_emotion.npy', current_emotion)

                    emotion_dict[current_emotion[0]].append(frame_counter)
                frame_counter += 1
                my_bar.progress((frame_counter) / length)
            final_emo_counts = {'emotion': emotions_list,
                                'counts': np.hstack([len(emotion_dict[emo]) for emo in emotions_list])}
            # final_emo_counts = {key: len(emotion_dict[key]) for key in emotions_list}
            final_emo_df = pd.DataFrame(data=final_emo_counts)
            fig, ax = plt.subplots(1, 1)
            ax.bar(final_emo_df['emotion'], final_emo_counts['counts'])
            st.pyplot(fig)

        # with in_file.open("rb") as f:
        #     st.download_button(
        #         "Download the recorded video without video filter", f, "input.mp4"
        #     )

    # if out_file.exists():
    #     with out_file.open("rb") as f:
    #         st.download_button(
    #             "Download the recorded video with video filter", f, "output.mp4"
    #         )


if __name__ == "__main__":
    load_view()
