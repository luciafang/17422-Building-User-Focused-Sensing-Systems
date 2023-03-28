import streamlit as st
import wave #save soundbytes to .wav file 
import pickle #load the emotional classifier based on speech 
import pandas as pd
# from audio_recorder_streamlit import audio_recorder
# import speech_recognition as sr


from revChatGPT.V1 import Chatbot
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from pydub import AudioSegment

import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List
import os

import av
import numpy as np
import pydub
import streamlit as st

from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main_speech():
    # st.header("Real Time Speech-to-Text")
#     st.markdown(
#         """
# This demo app is using [DeepSpeech](https://github.com/mozilla/DeepSpeech),
# an open speech-to-text engine.
# A pre-trained model released with
# [v0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3),
# trained on American English is being served.
# """
#     )

    # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3

    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models-zh-CN.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models-zh-CN.scorer"
    # if not HERE / "models/deepspeech-0.9.3-models-zh-CN.pbmm":
    # if not HERE / "models/deepspeech-0.9.3-models.pbmm":
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    sound_only_page = "Sound only (sendonly)"
    # with_video_page = "With video (sendrecv)"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page])

    if app_mode == sound_only_page:
        app_sst(
            str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )
    # elif app_mode == with_video_page:
    #     app_sst_with_video(
    #         str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
    #     )


def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break




def load_view():
    st.subheader('How was your day?')
    # initialize the recognizer
    # r = sr.Recognizer()
    WAVE_OUTPUT_FILENAME = "output.wav"
    with open('emotion_model.pkl', 'rb') as fr:
        rec = pickle.load(fr)
    colL, colR = st.columns(2)
    session_expander = colL.expander('Record your voice', expanded=True)
    emotion_expander = colR.expander('Predicted emotion', expanded=True)
    chatbot = Chatbot(config={
        "email": "ahsu2@andrew.cmu.edu",
        "password": "+zapubno1"
    })



    # if session_expander.button('Talk to me!'):
    #     r = sr.Recognizer()
    #     m = sr.Microphone()
    #     # colL, colR = session_expander.columns(2)
    #     # set threhold level
    #     with m as source:
    #         r.adjust_for_ambient_noise(source)
    #         r.pause_threshold = 2
    #     session_expander.write("Set minimum energy threshold to {}".format(r.energy_threshold))
    #     # obtain audio from the microphone
    #
    #     with sr.Microphone() as source:
    #         with colL:
    #             with st.spinner("Say something!"):
    #                 audio = r.listen(source)
    #
    #     with open(WAVE_OUTPUT_FILENAME, "wb") as file:
    #         file.write(audio.get_wav_data())



    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)


    main_speech()

    # # prompt = f"{r.recognize_google(audio)}"
    # response = ""
    # session_expander.write(f'User: {prompt}')
    #
    # for data in chatbot.ask(
    #         prompt
    # ):
    #     response = data["message"]
    #
    # emotion_expander.write(f'ChatGPT: {response}')

    # with session_expander:
    #     audio_bytes = audio_recorder()
    # if audio_bytes:
    #
        # with wave.open(WAVE_OUTPUT_FILENAME, "wb") as audiofile:
        #     audiofile.setsampwidth(2)
        #     audiofile.setnchannels(1)
        #     audiofile.setframerate(96000)
        #     audiofile.writeframes(audio_bytes)


    # session_expander.audio(WAVE_OUTPUT_FILENAME, format="audio/wav")
    # emotions_by_speech = rec.predict_proba(WAVE_OUTPUT_FILENAME)
    # emotion_df = pd.DataFrame(
    #     emotions_by_speech.items(), columns=["emotion", "score"]
    # )
    # emotion_df = emotion_df.sort_values(
    #     by=["score"], ascending=False
    # ).reset_index(drop=True)
    # current_emotion = []
    # emotion_score = []
    # for index, instance in emotion_df.iterrows():
    #     if index < 3:
    #         current_emotion.append(instance["emotion"])
    #         emotion_score.append(instance["score"])
        # emotion_expander.table(emotion_df)
        # st.info(current_emotion)
        # st.warning(emotion_score)
        # st.error('ERROR')
        # st.success(f'Based on your tone, you are most likely feeling: {current_emotion[0]}')
