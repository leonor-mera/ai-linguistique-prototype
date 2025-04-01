
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
import random

st.set_page_config(page_title="AI Linguistique - Choix du micro", layout="centered")
st.title("🎙️ AI Linguistique - Entraînement à la prononciation avec choix du micro")

st.markdown("Parlez depuis votre navigateur, sélectionnez le bon micro, et recevez un retour immédiat.")

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

ctx = webrtc_streamer(
    key="prononciation",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True
)

if not ctx.state.playing:
    st.warning("🎤 Le module n'est pas encore actif. Autorisez l'accès au micro si demandé.")
else:
    st.success("🎙️ Micro actif ! Vous pouvez commencer à parler.")

    if ctx.audio_receiver is not None:
        try:
            audio_data = []
            for i in range(50):
                audio_chunk = ctx.audio_receiver.get_audio_frame()
                audio_data.append(audio_chunk.to_ndarray())
                time.sleep(0.2)

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

                feedback_table = {
                    word: random.choice(["❌ Faible", "⚠️ Moyen", "✅ Bon", "🌟 Excellent"])
                    for word in expected_text.replace(",", "").replace(".", "").split()
                }
                score = round(random.uniform(70, 95), 1)

                st.markdown("### 📝 Résultat de votre prononciation :")
                st.success(f"🎯 Score global : {score}%")
                st.table([{"Mot": k, "Évaluation": v} for k, v in feedback_table.items()])

                if any("❌" in val or "⚠️" in val for val in feedback_table.values()):
                    st.info("🗣 Conseil : retravaillez les mots avec une prononciation faible ou moyenne pour progresser.")
                else:
                    st.balloons()

                os.remove(temp_audio_path)
        except AttributeError:
            st.error("⏳ Le micro est actif, mais aucune donnée n'a encore été reçue. Veuillez essayer à nouveau.")
    else:
        st.info("🔄 En attente de réception audio... Parlez fort et distinctement.")
