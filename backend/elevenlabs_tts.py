# elevenlabs_tts.py
import base64
import io
from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

eleven_client = ElevenLabs(api_key=api_key)

def generate_tts_audio(text: str) -> bytes:
    audio_chunks = []

    for chunk in eleven_client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    ):
        if isinstance(chunk, bytes):
            audio_chunks.append(chunk)

    audio_bytes = b"".join(audio_chunks)
    return audio_bytes
