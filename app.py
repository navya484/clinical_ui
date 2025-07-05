import streamlit as st
from test4 import transcribe_audio
from main import extract_medical_entities
from main import generate_soap_note

st.set_page_config(page_title="Clinical AI", layout="centered")

st.title("🩺 AI-Powered Clinical Documentation")

# Step 1: Input Section
st.header("Step 1: Upload or Enter Patient Conversation")

audio_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3","m4a"])
manual_text = st.text_area("Or paste conversation transcript manually:")

transcribed_text = ""

if audio_file and st.button("Transcribe Audio"):
    with st.spinner("Transcribing audio using Whisper..."):
        transcribed_text = transcribe_audio(audio_file)
        st.success("Transcription Complete!")
        st.text_area("Transcribed Text", value=transcribed_text, height=200)

elif manual_text and st.button("Use Manual Text"):
    transcribed_text = manual_text

# Step 2: Process Text
if transcribed_text:
    st.header("Step 2: Extract Medical Entities and Generate Summary")

    with st.spinner("Running Named Entity Recognition (NER)..."):
        entities = extract_medical_entities(transcribed_text)

    with st.expander("🔍 View Extracted Entities"):
        for ent in entities:
            st.markdown(f"- **{ent['entity_group']}** → `{ent['word']}` (Score: {ent['score']:.2f})")

    with st.spinner("Generating SOAP-format Clinical Note..."):
        soap_note = generate_soap_note(transcribed_text, entities)

    st.subheader("📝 Structured Clinical Note (SOAP Format)")
    st.code(soap_note, language="markdown")
else:
    st.info("Please upload audio or input text to begin.")
