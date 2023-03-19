import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions
from emotion_recognition import EmotionRecognizer
import pickle
import wave
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
# RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
with open('emotion_model.pkl', 'rb') as fr:
    rec = pickle.load(fr)
# rec.predict('./output.wav')
# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks

session_length = st.number_input('Session Length:',
                                 min_value=5, max_value=120, value=10)
colL, _, colR = st.columns([1, 3, 1])
image_placeholder = st.empty()
text_placeholder = st.empty()
session_begin = colL.checkbox('Start Session')
st.session_state['face_emo'] = None
st.session_state['speech_emo'] = None
if session_begin:
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    with st.spinner("* recording..."):

        frames = []

        for i in range(0, int(RATE / CHUNK * session_length)):
            data = stream.read(CHUNK)
            frames.append(data)

    st.success("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    st.session_state['speech_emo'] = rec.predict(WAVE_OUTPUT_FILENAME)


    picture = st.camera_input("Snap to finalize session.")


    # audio_bytes = audio_recorder()
    # with open('emotion_model.pkl', 'rb') as fr:
    #     rec = pickle.load(fr)
    # if audio_bytes:
    #     st.audio(audio_bytes, format="audio/wav")
    #     with wave.open("./myaudiofile.wav", "wb") as audiofile:
    #         audiofile.setsampwidth(2)
    #         audiofile.setnchannels(1)
    #         audiofile.setframerate(96000)
    #         audiofile.writeframes(audio_bytes)
    #     speech_emo = rec.predict('./myaudiofile.wav')
    #     st.write(speech_emo)
    st.write(st.session_state['speech_emo'])
    if picture:
        img_bytes = picture.getvalue()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
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
        print(f"facial recognition model {model_name} is just built")

        if enable_face_analysis:
            DeepFace.build_model(model_name="Emotion")
            print("Emotion model is just built")

        # -----------------------
        # visualization
        freeze = False
        face_detected = False
        face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
        freezed_frame = 0
        tic = time.time()
        emotion_df = None
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
            if w > 130:  # discard small detected faces

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
                        st.session_state['face_emo'] = emotion_df.iloc[0, 0]


                        for index, instance in emotion_df.iterrows():
                            current_emotion = instance["emotion"]
                            emotion_label = f"{current_emotion} "
                            emotion_score = instance["score"] / 100
    st.write(st.session_state["speech_emo"])
    text_placeholder.header(f'Face emotion: {st.session_state["face_emo"]}, '
                            f'while speech emotion: {st.session_state["speech_emo"]})')
        # image_placeholder.image(picture)