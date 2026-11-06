# list_models.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
skip = ("guard", "safeguard", "whisper", "tts", "orpheus", "embed")

for m in Groq(api_key=os.getenv("GROQ_API_KEY")).models.list().data:
    if not any(s in m.id.lower() for s in skip):
        print(m.id)