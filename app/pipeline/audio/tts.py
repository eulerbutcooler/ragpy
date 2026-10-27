import io

import numpy as np
import soundfile as sf
from kokoro import KPipeline

tts_pipeline = None

def _prepare_en_g2p():
    """Ensure spacy model en_core_web_sm is available so misaki doesn't invoke a missing pip."""
    try:
        import spacy.util
        if not spacy.util.is_package("en_core_web_sm"):
            try:
                import pip  # noqa: F401
            except ImportError:
                import ensurepip
                ensurepip.bootstrap()
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
    except BaseException as e:
        print(f"Notice: unable to auto-download en_core_web_sm: {e}")

def get_tts_pipeline():
    global tts_pipeline
    if tts_pipeline is None:
        _prepare_en_g2p()
        try:
            tts_pipeline = KPipeline(lang_code='b') # British English for fast phonemization & formal naval tone
        except BaseException as e:
            print(f"Notice: Failed to load British English Kokoro pipeline ({e}), falling back to Hindi ('h')...")
            try:
                tts_pipeline = KPipeline(lang_code='h')
            except BaseException as e2:
                print(f"Failed to load fallback Kokoro TTS pipeline: {e2}")
                raise RuntimeError(f"TTS Init Error: {e2}")
    return tts_pipeline

def _resolve_voice(pipeline, voice: str) -> str:
    lang = getattr(pipeline, "lang_code", "b")
    if lang == "b":
        if not voice or voice.startswith("h"):
            return "bm_george"
    elif lang == "h":
        if not voice or voice.startswith("b"):
            return "hm_omega"
    return voice or ("bm_george" if lang == "b" else "hm_omega")

def generate_speech_chunks(text: str, voice: str = 'bm_george', speed: float = 1.0):
    pipeline = get_tts_pipeline()
    if not pipeline:
        raise RuntimeError("Kokoro TTS pipeline is not initialized.")
    voice = _resolve_voice(pipeline, voice)
        
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'[.!?]+\s*|\n+')
    
    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None and len(audio) > 0:
            out_io = io.BytesIO()
            sf.write(out_io, audio, 24000, format='WAV')
            yield out_io.getvalue()

def generate_speech(text: str, voice: str = 'bm_george', speed: float = 1.0) -> bytes:
    pipeline = get_tts_pipeline()
    if not pipeline:
        raise RuntimeError("Kokoro TTS pipeline is not initialized.")
    voice = _resolve_voice(pipeline, voice)
        
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'[.!?]+\s*|\n+')
    audio_chunks = []
    
    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None:
            audio_chunks.append(audio)
            
    if not audio_chunks:
        return b""
        
    final_audio = np.concatenate(audio_chunks)
    out_io = io.BytesIO()
    sf.write(out_io, final_audio, 24000, format='WAV')
    
    return out_io.getvalue()
