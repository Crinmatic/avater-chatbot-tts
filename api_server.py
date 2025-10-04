from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import io
import wave
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from kokoro.pipeline import KPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0

app = FastAPI(
    title="Kokoro TTS API",
    description="A REST API for Kokoro Text-to-Speech synthesis",
    version="1.0.0"
)

# Global pipeline cache to avoid reloading models
pipeline_cache = {}

# Language detection mapping
LANGUAGE_MAPPING = {
    'en': 'a',  # English -> American English
    'es': 'e',  # Spanish
    'fr': 'f',  # French
    'hi': 'h',  # Hindi
    'it': 'i',  # Italian
    'ja': 'j',  # Japanese
    'pt': 'p',  # Portuguese
    'zh': 'z',  # Chinese
    'zh-cn': 'z',  # Chinese (Simplified)
    'zh-tw': 'z',  # Chinese (Traditional)
}

# Default voices for each language (all female voices)
DEFAULT_VOICES = {
    'a': 'af_bella',    # American English Female
    'b': 'bf_alice',    # British English Female
    'e': 'ef_dora',     # Spanish Female
    'f': 'ff_siwis',    # French Female
    'h': 'hf_alpha',    # Hindi Female
    'i': 'if_sara',     # Italian Female
    'j': 'jf_alpha',    # Japanese Female
    'p': 'pf_dora',     # Portuguese Female
    'z': 'zf_xiaobei',  # Chinese Female
}

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # Will be auto-selected based on language if not provided
    language: Optional[str] = None  # Will be auto-detected if not provided
    speed: float = 1.0
    repo_id: Optional[str] = None
    auto_detect_language: bool = True  # Enable/disable automatic language detection

class VoiceInfo(BaseModel):
    voice_id: str
    language: str
    description: str

# Available voices mapping
AVAILABLE_VOICES = {
    # American Female
    "af_alloy": {"language": "a", "description": "American Female - Alloy"},
    "af_aoede": {"language": "a", "description": "American Female - Aoede"},
    "af_bella": {"language": "a", "description": "American Female - Bella"},
    "af_heart": {"language": "a", "description": "American Female - Heart"},
    "af_jessica": {"language": "a", "description": "American Female - Jessica"},
    "af_kore": {"language": "a", "description": "American Female - Kore"},
    "af_nicole": {"language": "a", "description": "American Female - Nicole"},
    "af_nova": {"language": "a", "description": "American Female - Nova"},
    "af_river": {"language": "a", "description": "American Female - River"},
    "af_sarah": {"language": "a", "description": "American Female - Sarah"},
    "af_sky": {"language": "a", "description": "American Female - Sky"},
    
    # American Male
    "am_adam": {"language": "a", "description": "American Male - Adam"},
    "am_echo": {"language": "a", "description": "American Male - Echo"},
    "am_eric": {"language": "a", "description": "American Male - Eric"},
    "am_fenrir": {"language": "a", "description": "American Male - Fenrir"},
    "am_liam": {"language": "a", "description": "American Male - Liam"},
    "am_michael": {"language": "a", "description": "American Male - Michael"},
    "am_onyx": {"language": "a", "description": "American Male - Onyx"},
    "am_puck": {"language": "a", "description": "American Male - Puck"},
    "am_santa": {"language": "a", "description": "American Male - Santa"},
    
    # British Female
    "bf_alice": {"language": "b", "description": "British Female - Alice"},
    "bf_emma": {"language": "b", "description": "British Female - Emma"},
    "bf_isabella": {"language": "b", "description": "British Female - Isabella"},
    "bf_lily": {"language": "b", "description": "British Female - Lily"},
    
    # British Male
    "bm_daniel": {"language": "b", "description": "British Male - Daniel"},
    "bm_fable": {"language": "b", "description": "British Male - Fable"},
    "bm_george": {"language": "b", "description": "British Male - George"},
    "bm_lewis": {"language": "b", "description": "British Male - Lewis"},
    
    # Other languages
    "ef_dora": {"language": "e", "description": "Spanish Female - Dora"},
    "em_alex": {"language": "e", "description": "Spanish Male - Alex"},
    "em_santa": {"language": "e", "description": "Spanish Male - Santa"},
    "ff_siwis": {"language": "f", "description": "French Female - Siwis"},
    "hf_alpha": {"language": "h", "description": "Hindi Female - Alpha"},
    "hf_beta": {"language": "h", "description": "Hindi Female - Beta"},
    "hm_omega": {"language": "h", "description": "Hindi Male - Omega"},
    "hm_psi": {"language": "h", "description": "Hindi Male - Psi"},
    "if_sara": {"language": "i", "description": "Italian Female - Sara"},
    "im_nicola": {"language": "i", "description": "Italian Male - Nicola"},
    "jf_alpha": {"language": "j", "description": "Japanese Female - Alpha"},
    "jf_gongitsune": {"language": "j", "description": "Japanese Female - Gongitsune"},
    "jf_nezumi": {"language": "j", "description": "Japanese Female - Nezumi"},
    "jf_tebukuro": {"language": "j", "description": "Japanese Female - Tebukuro"},
    "jm_kumo": {"language": "j", "description": "Japanese Male - Kumo"},
    "pf_dora": {"language": "p", "description": "Portuguese Female - Dora"},
    "pm_alex": {"language": "p", "description": "Portuguese Male - Alex"},
    "pm_santa": {"language": "p", "description": "Portuguese Male - Santa"},
    "zf_xiaobei": {"language": "z", "description": "Chinese Female - Xiaobei"},
    "zf_xiaoni": {"language": "z", "description": "Chinese Female - Xiaoni"},
    "zf_xiaoxiao": {"language": "z", "description": "Chinese Female - Xiaoxiao"},
    "zf_xiaoyi": {"language": "z", "description": "Chinese Female - Xiaoyi"},
    "zm_yunjian": {"language": "z", "description": "Chinese Male - Yunjian"},
    "zm_yunxi": {"language": "z", "description": "Chinese Male - Yunxi"},
    "zm_yunxia": {"language": "z", "description": "Chinese Male - Yunxia"},
    "zm_yunyang": {"language": "z", "description": "Chinese Male - Yunyang"},
}

def detect_language(text: str) -> str:
    """Detect language from text and return corresponding Kokoro language code."""
    try:
        # Clean text for better detection
        clean_text = text.strip()
        if len(clean_text) < 3:
            logger.warning("Text too short for reliable language detection, defaulting to English")
            return 'a'
        
        detected = detect(clean_text)
        logger.info(f"Detected language: {detected}")
        
        # Map to Kokoro language codes
        kokoro_lang = LANGUAGE_MAPPING.get(detected, 'a')  # Default to American English
        logger.info(f"Mapped to Kokoro language code: {kokoro_lang}")
        
        return kokoro_lang
        
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return 'a'
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}, defaulting to English")
        return 'a'

def select_voice(language: str, requested_voice: Optional[str] = None) -> str:
    """Select appropriate voice based on language and user preference."""
    if requested_voice and requested_voice in AVAILABLE_VOICES:
        # Check if the requested voice matches the language
        voice_lang = AVAILABLE_VOICES[requested_voice]["language"]
        if voice_lang != language:
            logger.warning(f"Voice '{requested_voice}' is for language '{voice_lang}' but text is '{language}'. Using requested voice anyway.")
        return requested_voice
    
    # Use default voice for the detected language
    default_voice = DEFAULT_VOICES.get(language, 'af_bella')
    logger.info(f"Selected default voice '{default_voice}' for language '{language}'")
    return default_voice

def get_or_create_pipeline(language: str, repo_id: Optional[str] = None) -> KPipeline:
    """Get or create a pipeline for the specified language."""
    cache_key = f"{language}_{repo_id or 'default'}"
    
    if cache_key not in pipeline_cache:
        logger.info(f"Creating new pipeline for language: {language}")
        try:
            pipeline_cache[cache_key] = KPipeline(
                lang_code=language,
                repo_id=repo_id
            )
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize TTS pipeline: {str(e)}")
    
    return pipeline_cache[cache_key]

def audio_to_wav_bytes(audio: torch.FloatTensor, sample_rate: int = 24000) -> bytes:
    """Convert audio tensor to WAV bytes."""
    # Convert to numpy and ensure correct format
    audio_np = audio.cpu().numpy()
    
    # Normalize audio to 16-bit range
    audio_16bit = (audio_np * 32767).astype('int16')
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_16bit.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Kokoro TTS API with Auto Language Detection",
        "version": "1.0.0",
        "features": [
            "Automatic language detection from text",
            "Auto voice selection based on detected language",
            "Support for 9 languages",
            "Multiple voices per language"
        ],
        "endpoints": {
            "/synthesize": "POST - Generate speech from text (with auto language detection)",
            "/synthesize-stream": "POST - Generate speech with streaming response",
            "/voices": "GET - List available voices",
            "/detect-language": "POST - Detect language from text",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}

@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List all available voices."""
    voices = []
    for voice_id, info in AVAILABLE_VOICES.items():
        voices.append(VoiceInfo(
            voice_id=voice_id,
            language=info["language"],
            description=info["description"]
        ))
    return voices

@app.post("/detect-language")
async def detect_text_language(request: dict):
    """
    Detect the language of the provided text.
    
    Returns the detected language code and recommended voice.
    """
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
            
        detected_lang = detect_language(text)
        recommended_voice = select_voice(detected_lang)
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "detected_language_code": detected_lang,
            "detected_language_name": {
                'a': 'American English',
                'b': 'British English', 
                'e': 'Spanish',
                'f': 'French',
                'h': 'Hindi',
                'i': 'Italian',
                'j': 'Japanese',
                'p': 'Portuguese',
                'z': 'Chinese'
            }.get(detected_lang, 'American English'),
            "recommended_voice": recommended_voice,
            "voice_description": AVAILABLE_VOICES[recommended_voice]["description"]
        }
        
    except Exception as e:
        logger.error(f"Error in detect_text_language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text.
    
    Returns audio as WAV file.
    """
    try:
        # Auto-detect language if not provided and auto-detection is enabled
        if request.auto_detect_language and not request.language:
            request.language = detect_language(request.text)
            logger.info(f"Auto-detected language: {request.language}")
        elif not request.language:
            request.language = 'a'  # Default to American English
            logger.info("Using default language: American English")
        
        # Auto-select voice if not provided
        if not request.voice:
            request.voice = select_voice(request.language)
            logger.info(f"Auto-selected voice: {request.voice}")
        
        # Validate voice exists
        if request.voice not in AVAILABLE_VOICES:
            raise HTTPException(
                status_code=400, 
                detail=f"Voice '{request.voice}' not available. Use /voices endpoint to see available voices."
            )
        
        # Get or create pipeline
        pipeline = get_or_create_pipeline(request.language, request.repo_id)
        
        # Generate speech
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        results = list(pipeline(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        ))
        
        if not results:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Concatenate all audio results
        audio_segments = [result.audio for result in results if result.audio is not None]
        
        if not audio_segments:
            raise HTTPException(status_code=500, detail="No audio segments generated")
        
        # Concatenate audio
        combined_audio = torch.cat(audio_segments, dim=-1)
        
        # Convert to WAV bytes
        wav_bytes = audio_to_wav_bytes(combined_audio)
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in synthesize_speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize-stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    Synthesize speech from text with streaming response.
    
    Returns audio as streaming WAV file.
    """
    try:
        # Auto-detect language if not provided and auto-detection is enabled
        if request.auto_detect_language and not request.language:
            request.language = detect_language(request.text)
            logger.info(f"Auto-detected language: {request.language}")
        elif not request.language:
            request.language = 'a'  # Default to American English
            logger.info("Using default language: American English")
        
        # Auto-select voice if not provided
        if not request.voice:
            request.voice = select_voice(request.language)
            logger.info(f"Auto-selected voice: {request.voice}")
        
        # Validate voice exists
        if request.voice not in AVAILABLE_VOICES:
            raise HTTPException(
                status_code=400, 
                detail=f"Voice '{request.voice}' not available. Use /voices endpoint to see available voices."
            )
        
        # Get or create pipeline
        pipeline = get_or_create_pipeline(request.language, request.repo_id)
        
        def generate_audio():
            """Generator for streaming audio."""
            logger.info(f"Streaming speech for text: {request.text[:50]}...")
            
            for result in pipeline(
                text=request.text,
                voice=request.voice,
                speed=request.speed
            ):
                if result.audio is not None:
                    wav_bytes = audio_to_wav_bytes(result.audio)
                    yield wav_bytes
        
        return StreamingResponse(
            generate_audio(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in synthesize_speech_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)