
import streamlit as st
import tempfile
import whisper
import os
from gtts import gTTS

st.set_page_config(page_title="AI Linguistique - Prototype", layout="centered")
st.title("🎙️ AI Linguistique - Prototype de prononciation en français")

st.markdown("Enregistrez votre voix, écoutez la phrase modèle, et l'IA vous aidera à améliorer votre prononciation.")

# Chargement du modèle Whisper
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Entrée utilisateur : phrase à prononcer
expected_text = st.text_input("📌 Entrez la phrase à pratiquer :", "Bonjour, je m'appelle Leonor.")

# Génération audio de la phrase
if expected_text:
    tts = gTTS(expected_text, lang='fr')
    audio_path = "audio_reference.mp3"
    tts.save(audio_path)
    st.audio(audio_path, format="audio/mp3")
    st.markdown("*Écoutez la prononciation correcte ci-dessus.*")

# Enregistrement audio
audio_file = st.file_uploader("🎧 Téléversez votre enregistrement vocal (format .wav ou .mp3)", type=["wav", "mp3"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    # Transcription avec Whisper
    with st.spinner("🧠 Analyse de votre prononciation..."):
        result = model.transcribe(tmp_path, language="fr")
        user_text = result["text"]

    # Affichage des résultats
    st.markdown("### 📄 Transcription de votre voix :")
    st.write(user_text)

    # Évaluation simple
    if expected_text.strip().lower() in user_text.strip().lower():
        st.success("✅ Bonne prononciation ! La phrase est correcte.")
    else:
        st.warning("🔍 Il semble y avoir des différences avec la phrase attendue.")
        st.markdown("**Phrase attendue :** " + expected_text)
        st.markdown("**Votre phrase :** " + user_text)

    os.remove(tmp_path)
