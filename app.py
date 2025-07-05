import streamlit as st
from main import whisper_model, ner, model, contract, web3, collection, datetime, Account
from main import transcript as static_transcript  # only used if no audio uploaded
import tempfile
import os

# Page Config
st.set_page_config(page_title="🩺 AI Clinical Documentation", layout="centered")
st.title("🩺 AI-Powered Clinical Documentation")
st.markdown("Upload a doctor-patient audio conversation to auto-generate clinical SOAP notes using AI.")

# Audio Upload
audio_file = st.file_uploader("🎙️ Upload a patient consultation audio file (.wav/.m4a)", type=["wav", "m4a"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    st.info("🔊 Transcribing audio using Whisper...")
    result = whisper_model.transcribe(temp_path)
    transcript = result["text"]
    os.remove(temp_path)

    st.subheader("📝 Transcript")
    st.text_area("Transcribed Text", transcript, height=200)

    # NER
    st.info("🧬 Extracting medical entities...")
    entities = ner(transcript)
    entity_words = list(set([e['word'] for e in entities]))

    st.subheader("🔍 Named Medical Entities")
    for e in entities:
        st.markdown(f"- **{e['entity_group']}**: {e['word']} (score: {e['score']:.2f})")

    # Summary
    st.info("🧠 Generating summary using Gemini...")
    prompt_summary = (
        "Summarize the following medical entities extracted from a doctor-patient conversation. "
        "Include patient's condition, diagnosis, and any treatment clues:\n\n"
        f"Entities: {', '.join(entity_words)}"
    )
    response_summary = model.generate_content(prompt_summary)
    summary = response_summary.text.strip()
    st.subheader("🧠 Summary")
    st.write(summary)

    # SOAP Note
    st.info("📋 Generating SOAP note...")
    prompt_soap = f"""
    You are a clinical assistant.

    Your task is to read the following unstructured transcript of a doctor-patient consultation and extract relevant medical information to generate a SOAP clinical note.

    Transcript:
    {transcript}

    Return the SOAP note in the following format:

    S: (Subjective)
    O: (Objective)
    A: (Assessment)
    P: (Plan)
    """
    response_soap = model.generate_content(prompt_soap)
    soap_text = response_soap.text.strip()

    # Parse SOAP
    soap_data = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    for line in soap_text.splitlines():
        if line.startswith("S:"):
            soap_data["subjective"] = line[2:].strip()
        elif line.startswith("O:"):
            soap_data["objective"] = line[2:].strip()
        elif line.startswith("A:"):
            soap_data["assessment"] = line[2:].strip()
        elif line.startswith("P:"):
            soap_data["plan"] = line[2:].strip()

    st.subheader("📋 SOAP Note")
    st.markdown(f"**S (Subjective):** {soap_data['subjective']}")
    st.markdown(f"**O (Objective):** {soap_data['objective']}")
    st.markdown(f"**A (Assessment):** {soap_data['assessment']}")
    st.markdown(f"**P (Plan):** {soap_data['plan']}")

    # Store to Blockchain and MongoDB
    if st.button("💾 Save Record"):
        try:
            patient_id = "PATIENT_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            tx_hash = contract.functions.saveRecord(
                patient_id,
                summary,
                soap_data["subjective"],
                soap_data["objective"],
                soap_data["assessment"],
                soap_data["plan"]
            ).transact()
            web3.eth.wait_for_transaction_receipt(tx_hash)
            st.success("✅ Record saved to Blockchain!")

            collection.insert_one({
                "patientId": patient_id,
                "summary": summary,
                "soap": soap_data,
                "transcript": transcript,
                "timestamp": datetime.datetime.now().isoformat()
            })
            st.success("✅ Record saved to MongoDB!")

            st.markdown("---")
            st.subheader("📦 Retrieved Record")
            retrieved = contract.functions.getRecord(patient_id).call()
            st.markdown(f"**From Blockchain:** `{patient_id}`")
            st.text(f"Summary: {retrieved[0]}")
            st.text(f"S: {retrieved[1]}")
            st.text(f"O: {retrieved[2]}")
            st.text(f"A: {retrieved[3]}")
            st.text(f"P: {retrieved[4]}")
            st.text(f"Timestamp: {datetime.datetime.fromtimestamp(retrieved[5])}")
        except Exception as e:
            st.error(f"❌ Error storing/retrieving: {e}")
else:
    st.info("👆 Upload a valid audio file to begin.")
