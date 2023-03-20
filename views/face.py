import streamlit as st
import os 
import time
import pandas as pd
import cv2 #rgb. bgr 
import numpy as np
from deepface import DeepFace
from deepface.commons import functions

#tweak code from DeepFace.commons.realtime.analyze  
def load_view():
    st.subheader('Take a picture of how you really feel.')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    colL, colR = st.columns(2)
    image_placeholder = colR.empty()
    live_expander = colL.expander('Take a picture', expanded=True)
    emotion_expander = colR.expander('Predicted emotion', expanded=True)
    picture = live_expander.camera_input(" ") ##added a live camera 

    #from DeepFace.commons.realtime.analyze  
    if picture:
        img_bytes = picture.getvalue()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        model_name = "VGG-Face"
        detector_backend = "mediapipe" #tweak into mediapipe 
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
        # emotion_df = None
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
                            if index < 3: #limited to Top3 
                                current_emotion.append(instance["emotion"])
                                # emotion_label = f"{current_emotion} "
                                emotion_score.append(instance["score"] / 100)
            emotion_expander.table(emotion_df) #place a table on the right
            st.success(f'Based on your expression, you are most likely feeling: {current_emotion[0]}') #use the highest emotin score