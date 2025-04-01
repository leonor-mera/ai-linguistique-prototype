
import streamlit as st
import whisper
import tempfile
import os
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
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
        self.recorded_frames = []
        self.recording = False
        self.audio_queue = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_queue.put(audio)
        return frame

ctx = webrtc_streamer(key="prononciation",
                      mode="sendonly",
                      audio_receiver_size=256,
                      media_stream_constraints={"audio": True, "video": False},
                      async_processing=True)

if ctx.audio_receiver:
    audio_data = []
    for i in range(50):
        try:
            audio_chunk = ctx.audio_receiver.get_audio_frame()
            audio_data.append(audio_chunk.to_ndarray())
            time.sleep(0.2)
        except queue.Empty:
            break

    if audio_data:
        audio_array = np.concatenate(audio_data, axis=1)[0]
        temp_audio_path = "user_voice.wav"
        sf.write(temp_audio_path, audio_array, 48000)

        st.audio(temp_audio_path, format="audio/wav")

        with st.spinner("🧠 Analyse de votre prononciation..."):
            result = model.transcribe(temp_audio_path, language="fr")
            user_text = result["text"]

        st.markdown("### 📄 Transcription de votre voix :")
        st.write(user_text)

        if expected_text.strip().lower() in user_text.strip().lower():
            st.success("✅ Bonne prononciation ! La phrase est correcte.")
        else:
            st.warning("🔍 Il semble y avoir des différences avec la phrase attendue.")
            st.markdown("**Phrase attendue :** " + expected_text)
            st.markdown("**Votre phrase :** " + user_text)

        os.remove(temp_audio_path)
