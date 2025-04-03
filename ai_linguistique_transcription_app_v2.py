
import streamlit as st
import tempfile
import whisper
import os
from pydub import AudioSegment

st.set_page_config(page_title="AI Linguistique - Transcription", layout="centered")
st.title("🧠 Analyse de prononciation - AI Linguistique")

st.markdown("📤 Importez ici le fichier audio `.webm` généré par l'enregistreur vocal. "
            "Nous vous recommandons un extrait de **5 secondes ou plus** pour une meilleure précision.")

audio_file = st.file_uploader("Téléversez votre fichier `.webm` ici :", type=["webm"])

@st.cache_resource
def load_model():
    return whisper.load_model("small")  # Utiliser un modèle plus robuste que "base"

model = load_model()

if audio_file is not None:
    st.audio(audio_file, format="audio/webm")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        temp_webm.write(audio_file.read())
        webm_path = temp_webm.name

    try:
        # Convertir le .webm en .wav avec pydub
        wav_path = webm_path.replace(".webm", ".wav")
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")

        with st.spinner("🧠 Transcription de votre voix en cours..."):
            result = model.transcribe(wav_path, fp16=False, language="fr")
            st.markdown("### ✍️ Texte détecté :")
            st.write(result["text"])

        os.remove(webm_path)
        os.remove(wav_path)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'audio : {e}")
