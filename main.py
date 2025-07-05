import sys
import whisper
from transformers import pipeline
import google.generativeai as genai

# 🎙 Step 1: Transcribe audio with Whisper
sys.stdout.reconfigure(encoding='utf-8')  # for emojis and unicode

print("🔊 Transcribing audio...")
whisper_model = whisper.load_model("base")  # or "tiny" for faster but less accurate
result = whisper_model.transcribe("test4.m4a")  # replace with your audio file
transcript = result["text"]

print("\n📝 Transcript:\n")
print(transcript)

# 🧬 Step 2: Load Biomedical NER pipeline
ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

# 🧠 Step 3: Perform NER on transcript
entities = ner(transcript)
entity_words = list(set([e['word'] for e in entities]))  # remove duplicates

print("\n🔍 Extracted Entities:")
for entity in entities:
    print(f"{entity['entity_group']} → {entity['word']} (Score: {entity['score']:.2f})")

# 💡 Step 4: Configure Gemini API
genai.configure(api_key="AIzaSyA286T4FaYw2MMpfUxxp6eQWQRhr8RblMg")  # Replace with your actual Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")

# ✍ Step 5: Prompt 1 — Summary of Entities
prompt_summary = (
    "Summarize the following medical entities extracted from a doctor-patient conversation. "
    "Include patient's condition, diagnosis, and any treatment clues:\n\n"
    f"Entities: {', '.join(entity_words)}"
)

response_summary = model.generate_content(prompt_summary)

print("\n🧠 Summary:")
print(response_summary.text)

# 📋 Step 6: Prompt 2 — SOAP Note
prompt_soap = f"""
You are a clinical assistant.

Your task is to read the following unstructured transcript of a doctor-patient consultation and extract relevant medical information to generate a SOAP clinical note.

The text may not clearly separate who is speaking. Use your medical knowledge and context to interpret the patient symptoms, observations, diagnosis, and treatment.

Transcript:
{transcript}

Return the SOAP note in the following format:

S: (Subjective - patient complaints and symptoms)
O: (Objective - vitals, observations, clinical findings)
A: (Assessment - diagnosis or reasoning)
P: (Plan - prescriptions, investigations, or treatment plan)
"""

response_soap = model.generate_content(prompt_soap)

print("\n📋 SOAP Note:")
print(response_soap.text)