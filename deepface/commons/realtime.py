import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions

import streamlit as st

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks


def analysis(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    # global variables
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

    enable_emotion = True
    enable_age_gender = False
    # ------------------------
    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    print(f"facial recognition model {model_name} is just built")

    if enable_face_analysis:
        DeepFace.build_model(model_name="Age")
        print("Age model is just built")
        DeepFace.build_model(model_name="Gender")
        print("Gender model is just built")
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

    image_placeholder = st.empty()
    text_placeholder = st.empty()

    cap = cv2.VideoCapture(source)  # webcam
    while True:
        _, img = cap.read()

        if img is None:
            break

        # cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
        # cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]

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

                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]  # crop detected face

                # -------------------------------------

                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

                # -------------------------------------

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:

            toc = time.time()
            if (toc - tic) < time_threshold:

                if freezed_frame == 0:
                    freeze_img = base_img.copy()
                    # here, np.uint8 handles showing white area issue
                    # freeze_img = np.zeros(resolution, np.uint8)

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        # -------------------------------
                        # extract detected face
                        custom_face = base_img[y : y + h, x : x + w]
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

                                    text_placeholder.write(emotion_df.iloc[0, 0])

                                    for index, instance in emotion_df.iterrows():
                                        current_emotion = instance["emotion"]
                                        emotion_label = f"{current_emotion} "
                                        emotion_score = instance["score"] / 100

                                        bar_x = 35  # this is the size if an emotion is 100%
                                        bar_x = int(bar_x * emotion_score)

                                if enable_age_gender:
                                    apparent_age = demography["age"]
                                    dominant_gender = demography["dominant_gender"]
                                    gender = "M" if dominant_gender == "Man" else "W"
                                    # print(f"{apparent_age} years old {dominant_emotion}")
                                    analysis_report = str(int(apparent_age)) + " " + gender

                                    # -------------------------------

                        tic = time.time()  # in this way, freezed image can show 5 seconds

                        # -------------------------------

                time_left = int(time_threshold - (toc - tic) + 1)
                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0
            image_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            image_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

