
import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
import whisper
import tempfile
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import queue
import time
import soundfile as sf
import openai

st.set_page_config(page_title="AI Linguistique - Agent vocal", layout="centered")
st.title("🗣️ AI Linguistique - Agent vocal en français québécois")

st.markdown("Parlez à l’agent IA, il vous répondra comme un tuteur québécois 👨‍🏫🇨🇦")

# Clé API de l'utilisateur (à remplacer ou configurer comme variable d’environnement)
openai.api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else "YOUR_API_KEY"

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_queue.put(audio)
        return frame

ctx = webrtc_streamer(
    key="linguistique",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True
)

if not ctx or not ctx.state.playing:
    st.warning("🎤 Activez votre micro pour commencer à parler.")
elif ctx.audio_receiver is not None:
    try:
        st.success("🎙️ Micro actif ! Parlez maintenant...")
        audio_data = []
        for _ in range(50):
            audio_chunk = ctx.audio_receiver.get_audio_frame()
            audio_data.append(audio_chunk.to_ndarray())
            time.sleep(0.2)

        if audio_data:
            audio_array = np.concatenate(audio_data, axis=1)[0]
            temp_audio_path = "input.wav"
            sf.write(temp_audio_path, audio_array, 48000)

            st.audio(temp_audio_path, format="audio/wav")

            with st.spinner("🔎 Transcription..."):
                result = model.transcribe(temp_audio_path, language="fr")
                user_text = result["text"]

            st.markdown("### ✍️ Ce que vous avez dit :")
            st.write(user_text)

            # Appel à ChatGPT pour une réponse style tuteur québécois
            prompt = f"Tu es un tuteur bienveillant québécois. Réponds à cette phrase d’un apprenant de manière encourageante, avec un ton québécois naturel et simple : « {user_text} »"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            reply_text = response.choices[0].message["content"]

            st.markdown("### 🤖 Réponse de l’agent IA :")
            st.write(reply_text)

            # Synthèse vocale avec gTTS pour simuler la voix québécoise
            tts = gTTS(reply_text, lang="fr")
            reply_audio_path = "reply.mp3"
            tts.save(reply_audio_path)
            st.audio(reply_audio_path, format="audio/mp3")

            os.remove(temp_audio_path)
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
else:
    st.info("⏳ En attente de la voix... Parlez fort et distinctement.")
