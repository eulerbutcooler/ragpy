import io

from faster_whisper import WhisperModel

stt_model = None

def get_stt_model():
    global stt_model
    if stt_model is None:
        from app.llm.hf import resolve_local_hf_model

        _, local_files_only = resolve_local_hf_model(
            "Systran/faster-whisper-small",
            check_filename="model.bin",
        )
        try:
            stt_model = WhisperModel(
                "small",
                device="auto",
                compute_type="auto",
                local_files_only=local_files_only,
            )
        except Exception as e:
            print(f"Error loading Whisper: {e}. Falling back to CPU...")
            stt_model = WhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                local_files_only=local_files_only,
            )
    return stt_model

def transcribe_audio(audio_bytes: bytes) -> str:
    audio_file = io.BytesIO(audio_bytes)
    model = get_stt_model()
    segments, info = model.transcribe(audio_file, beam_size=5)
    text = " ".join([segment.text for segment in segments]).strip()
    return text
