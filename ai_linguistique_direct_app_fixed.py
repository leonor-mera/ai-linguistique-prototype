
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
import whisper
import queue
import numpy as np
import soundfile as sf
import time
from difflib import SequenceMatcher
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="AI Linguistique - Direct", layout="centered")
st.title("🎙️ AI Linguistique - Évaluation vocale directe")

phrase_modele = "Bonjour, je m'appelle Leonor et j’apprends le français québécois."
st.markdown(f"🗣️ **Phrase à prononcer** : "{phrase_modele}"")
st.markdown("🎧 Parlez directement et recevez votre transcription + score de prononciation.")

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

audio_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        if audio_queue.qsize() < 100:
            audio_queue.put(audio)
        return frame

def evaluer_similarite(phrase_attendue, phrase_detectee):
    ratio = SequenceMatcher(None, phrase_attendue.lower(), phrase_detectee.lower()).ratio()
    return round(ratio * 100)

ctx = webrtc_streamer(
    key="direct-ia",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

if ctx.state.playing:
    st.success("🎙️ Micro actif ! Parlez maintenant...")

    if st.button("📣 Évaluer ma prononciation"):
        st.info("🔄 Traitement vocal en cours...")
        audio_data = []

        # Collecter ~8 secondes d'audio
        start_time = time.time()
        while time.time() - start_time < 8:
            try:
                audio = audio_queue.get(timeout=1)
                audio_data.append(audio)
            except queue.Empty:
                break

        if audio_data:
            audio_array = np.concatenate(audio_data, axis=1)[0]
            wav_path = "temp_direct.wav"
            sf.write(wav_path, audio_array, 48000)

            with st.spinner("🔍 Transcription..."):
                result = model.transcribe(wav_path, language="fr")
                phrase_utilisateur = result["text"]

            st.markdown("### ✍️ Transcription détectée :")
            st.write(phrase_utilisateur)

            score = evaluer_similarite(phrase_modele, phrase_utilisateur)
            st.markdown("### 🧪 Score de prononciation :")
            st.success(f"🎯 {score}% de précision")

            os.remove(wav_path)
        else:
            st.warning("Aucun audio détecté. Veuillez parler clairement et réessayer.")
else:
    st.warning("🎤 Le micro n’est pas encore activé. Cliquez sur 'Autoriser' si demandé.")
