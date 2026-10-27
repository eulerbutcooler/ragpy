import base64
import json
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.pipeline.audio.stt import transcribe_audio
from app.pipeline.audio.tts import generate_speech, generate_speech_chunks

router = APIRouter(tags=["audio"])

class TTSRequest(BaseModel):
    text: str
    voice: str = "bm_george"  # British male voice ('George' - fast, authoritative military tone)
    speed: float = 1.0
    stream: bool = True

@router.post("/transcribe")
@router.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        text = transcribe_audio(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")

@router.post("/speak")
@router.post("/speak/")
async def speak(request: TTSRequest, stream: bool = True):
    try:
        should_stream = stream and request.stream
        if should_stream:
            def stream_generator():
                for i, chunk_bytes in enumerate(generate_speech_chunks(request.text, request.voice, request.speed)):
                    b64 = base64.b64encode(chunk_bytes).decode("ascii")
                    yield json.dumps({"index": i, "audio": b64}) + "\n"

            return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

        wav_bytes = generate_speech(request.text, request.voice, request.speed)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")
