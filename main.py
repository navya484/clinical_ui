import sys
import whisper
from transformers import pipeline
import google.generativeai as genai
from web3 import Web3
from pymongo import MongoClient
import datetime
from pprint import pprint

# 🎙️ Step 1: Transcribe audio with Whisper
sys.stdout.reconfigure(encoding='utf-8')

print("🔊 Transcribing audio...")
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe("test4.m4a")
transcript = result["text"]

print("\n📝 Transcript:\n")
print(transcript)

# 🧬 Step 2: NER
ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
entities = ner(transcript)
entity_words = list(set([e['word'] for e in entities]))

print("\n🔍 Extracted Entities:")
for entity in entities:
    print(f"{entity['entity_group']} → {entity['word']} (Score: {entity['score']:.2f})")

# 💡 Step 3: Gemini
genai.configure(api_key="AIzaSyA286T4FaYw2MMpfUxxp6eQWQRhr8RblMg")
model = genai.GenerativeModel("gemini-1.5-flash")

# Summary
prompt_summary = (
    "Summarize the following medical entities extracted from a doctor-patient conversation. "
    "Include patient's condition, diagnosis, and any treatment clues:\n\n"
    f"Entities: {', '.join(entity_words)}"
)
response_summary = model.generate_content(prompt_summary)
summary = response_summary.text.strip()

print("\n🧠 Summary:")
print(summary)

# SOAP note
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
soap_note_text = response_soap.text.strip()

print("\n📋 SOAP Note:")
print(soap_note_text)

# Parse SOAP
soap_data = {
    "subjective": "",
    "objective": "",
    "assessment": "",
    "plan": ""
}
for line in soap_note_text.splitlines():
    if line.startswith("S:"):
        soap_data["subjective"] = line[2:].strip()
    elif line.startswith("O:"):
        soap_data["objective"] = line[2:].strip()
    elif line.startswith("A:"):
        soap_data["assessment"] = line[2:].strip()
    elif line.startswith("P:"):
        soap_data["plan"] = line[2:].strip()

# ----------------- Blockchain Setup -----------------
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
assert web3.is_connected(), "❌ Could not connect to Ganache"

# Use first account from Ganache (no private key needed)
address = web3.eth.accounts[0]

contract_address = Web3.to_checksum_address("0x21a2Bbdad8d810a6E9690Bc45D3A88b747CF477e")
contract_abi = [
    {
        "inputs": [
            {"internalType": "string", "name": "_patientId", "type": "string"},
            {"internalType": "string", "name": "_summary", "type": "string"},
            {"internalType": "string", "name": "_subjective", "type": "string"},
            {"internalType": "string", "name": "_objective", "type": "string"},
            {"internalType": "string", "name": "_assessment", "type": "string"},
            {"internalType": "string", "name": "_plan", "type": "string"}
        ],
        "name": "saveRecord",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "_patientId", "type": "string"}
        ],
        "name": "getRecord",
        "outputs": [
            {"internalType": "string", "name": "summary", "type": "string"},
            {"internalType": "string", "name": "subjective", "type": "string"},
            {"internalType": "string", "name": "objective", "type": "string"},
            {"internalType": "string", "name": "assessment", "type": "string"},
            {"internalType": "string", "name": "plan", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# ----------------- MongoDB Setup -----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_records"]
collection = db["soap_notes"]

# ----------------- Store Data -----------------
patient_id = "PATIENT_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

try:
    tx_hash = contract.functions.saveRecord(
        patient_id,
        summary,
        soap_data["subjective"],
        soap_data["objective"],
        soap_data["assessment"],
        soap_data["plan"]
    ).transact({"from": address})  # No private key needed here

    web3.eth.wait_for_transaction_receipt(tx_hash)
    print("✅ SOAP record stored on blockchain.")
except Exception as e:
    print("❌ Blockchain store failed:", e)

# Save to MongoDB
mongo_record = {
    "patientId": patient_id,
    "summary": summary,
    "soap": soap_data,
    "transcript": transcript,
    "timestamp": datetime.datetime.now().isoformat()
}
collection.insert_one(mongo_record)
print("✅ Record saved to MongoDB.")
print("🆔 Patient ID:", patient_id)

# ----------------- Retrieve and Display -----------------
print("\n🔁 Retrieving record from blockchain...")
try:
    retrieved = contract.functions.getRecord(patient_id).call()
    print("\n🧾 Blockchain Data:")
    print("Summary:", retrieved[0])
    print("S:", retrieved[1])
    print("O:", retrieved[2])
    print("A:", retrieved[3])
    print("P:", retrieved[4])
    print("Timestamp:", datetime.datetime.fromtimestamp(retrieved[5]))
except Exception as e:
    print("❌ Retrieval failed:", e)

print("\n📦 MongoDB Data:")
pprint(collection.find_one({"patientId": patient_id}))
