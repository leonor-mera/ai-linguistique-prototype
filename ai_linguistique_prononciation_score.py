
import streamlit as st
import tempfile
import whisper
import os
from pydub import AudioSegment
from difflib import SequenceMatcher

st.set_page_config(page_title="AI Linguistique - Prononciation", layout="centered")
st.title("🧠 Analyse de prononciation - AI Linguistique")

# Phrase de référence
phrase_modele = "Bonjour, je m'appelle Leonor et j’apprends le français québécois."

st.markdown(f"""
📤 Importez ici le fichier audio `.webm` généré par l'enregistreur vocal.  
📏 La phrase attendue est :  
> **🗣️ {phrase_modele}**

Nous vous recommandons un extrait de **5 secondes ou plus** pour une meilleure précision.
""")

audio_file = st.file_uploader("Téléversez votre fichier `.webm` ici :", type=["webm"])

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

def evaluer_similarite(phrase_attendue, phrase_detectee):
    ratio = SequenceMatcher(None, phrase_attendue.lower(), phrase_detectee.lower()).ratio()
    return round(ratio * 100)

if audio_file is not None:
    st.audio(audio_file, format="audio/webm")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        temp_webm.write(audio_file.read())
        webm_path = temp_webm.name

    try:
        # Conversion en WAV
        wav_path = webm_path.replace(".webm", ".wav")
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")

        with st.spinner("🧠 Transcription de votre voix en cours..."):
            result = model.transcribe(wav_path, fp16=False, language="fr")
            phrase_utilisateur = result["text"]

            st.markdown("### ✍️ Texte détecté :")
            st.write(phrase_utilisateur)

            score = evaluer_similarite(phrase_modele, phrase_utilisateur)
            st.markdown("### 🧪 Évaluation de la prononciation :")
            st.success(f"🎯 Score de prononciation : **{score}%**")

        os.remove(webm_path)
        os.remove(wav_path)

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'audio : {e}")
