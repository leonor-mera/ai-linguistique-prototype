
import streamlit as st
import whisper
import tempfile
import os
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import queue
import time
import soundfile as sf

st.set_page_config(page_title="AI Linguistique - Prononciation directe", layout="centered")
st.title("🎙️ AI Linguistique - Entraînement à la prononciation en direct")

st.markdown("Parlez directement depuis votre navigateur et l'IA analysera votre prononciation.")

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

expected_text = st.text_input("📌 Entrez la phrase à pratiquer :", "Bonjour, je m'appelle Leonor.")

if expected_text:
    tts = gTTS(expected_text, lang='fr')
    tts.save("audio_reference.mp3")
    st.audio("audio_reference.mp3", format="audio/mp3")
    st.markdown("*Écoutez la prononciation correcte ci-dessus.*")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_queue = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_queue.put(audio)
        return frame

ctx = webrtc_streamer(key="prononciation",
                      mode=WebRtcMode.SENDONLY,
                      audio_receiver_size=256,
                      media_stream_constraints={"audio": True, "video": False},
                      async_processing=True)

if not ctx.state.playing:
    st.warning("🎤 Le module n'est pas encore actif. Autorisez l'accès au micro si demandé.")
else:
    st.success("🎙️ Micro actif ! Vous pouvez commencer à parler.")
