from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import io
import wave
import logging
import os
import numpy as np
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from kokoro.pipeline import KPipeline
import sherpa_onnx

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
    'fa': 'fa',  # Persian/Farsi
    'ur': 'fa',  # Urdu (often detected for Persian text) -> map to Persian
    'nan': 'nan',  # Min-nan (Southern Min/Taiwanese Hokkien)
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
    'fa': 'piper_fa_amir',  # Persian Female (Piper model)
    'nan': 'piper_nan',  # Min-nan (Piper model)
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
    
    # Persian/Farsi (Piper models)
    "piper_fa_amir": {"language": "fa", "description": "Persian (Farsi) - Amir (Piper)"},
    "piper_fa_gyro": {"language": "fa", "description": "Persian (Farsi) - Gyro (Piper)"},
    
    # Min-nan / Southern Min / Taiwanese Hokkien (Piper model)
    "piper_nan": {"language": "nan", "description": "Min-nan (Southern Min/Taiwanese Hokkien) - MMS (Piper)"},
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
    logger.info(f"select_voice called with language='{language}', requested_voice='{requested_voice}'")
    logger.info(f"DEFAULT_VOICES keys: {list(DEFAULT_VOICES.keys())}")
    logger.info(f"DEFAULT_VOICES.get('{language}'): {DEFAULT_VOICES.get(language)}")
    
    if requested_voice and requested_voice in AVAILABLE_VOICES:
        # Check if the requested voice matches the language
        voice_lang = AVAILABLE_VOICES[requested_voice]["language"]
        if voice_lang != language:
            logger.warning(f"Voice '{requested_voice}' is for language '{voice_lang}' but text is '{language}'. Using requested voice anyway.")
        return requested_voice
    
    # Use default voice for the detected language
    default_voice = DEFAULT_VOICES.get(language)
    
    # If no default voice found, try to find any voice for this language
    if not default_voice:
        logger.warning(f"No default voice found in DEFAULT_VOICES for language '{language}'")
        for voice_id, voice_info in AVAILABLE_VOICES.items():
            if voice_info["language"] == language:
                default_voice = voice_id
                logger.info(f"No default voice for language '{language}', using '{default_voice}'")
                break
    
    # If still no voice found, default to American English
    if not default_voice:
        default_voice = 'af_bella'
        logger.warning(f"No voice found for language '{language}', defaulting to English")
    else:
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

# Piper TTS configuration
PIPER_MODELS = {
    "piper_fa_amir": {
        "model": "models/vits-piper-fa_IR-amir-medium/fa_IR-amir-medium.onnx",
        "tokens": "models/vits-piper-fa_IR-amir-medium/tokens.txt",
        "data_dir": "models/vits-piper-fa_IR-amir-medium/espeak-ng-data",
    },
    "piper_fa_gyro": {
        "model": "models/vits-piper-fa_IR-gyro-medium/fa_IR-gyro-medium.onnx",
        "tokens": "models/vits-piper-fa_IR-gyro-medium/tokens.txt",
        "data_dir": "models/vits-piper-fa_IR-gyro-medium/espeak-ng-data",
    },
    "piper_nan": {
        "model": "models/vits-mms-nan/model.onnx",
        "tokens": "models/vits-mms-nan/tokens.txt",
        "data_dir": None,  # MMS models don't use espeak-ng-data
    },
}

piper_tts_cache = {}

def get_or_create_piper_tts(voice: str):
    """Get or create a Piper TTS instance for the specified voice."""
    if voice not in piper_tts_cache:
        if voice not in PIPER_MODELS:
            raise HTTPException(status_code=400, detail=f"Piper voice '{voice}' not found")
        
        model_config = PIPER_MODELS[voice]
        logger.info(f"Creating new Piper TTS for voice: {voice}")
        
        # Get the absolute path to the models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_config["model"])
        tokens_path = os.path.join(base_dir, model_config["tokens"])
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"Model file not found: {model_path}")
        if not os.path.exists(tokens_path):
            raise HTTPException(status_code=500, detail=f"Tokens file not found: {tokens_path}")
        
        # Handle data_dir (optional for MMS models)
        data_dir = None
        if model_config["data_dir"]:
            data_dir = os.path.join(base_dir, model_config["data_dir"])
            if not os.path.exists(data_dir):
                raise HTTPException(status_code=500, detail=f"Data directory not found: {data_dir}")
        
        try:
            vits_model_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_path,
                tokens=tokens_path,
                data_dir=data_dir or "",  # Empty string if None
                length_scale=1.0,
                noise_scale=0.667,
                noise_scale_w=0.8,
            )
            
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_model_config,
                num_threads=2,
                debug=False,
                provider="cpu",
            )
            
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                max_num_sentences=1,
            )
            
            piper_tts_cache[voice] = sherpa_onnx.OfflineTts(tts_config)
            logger.info(f"Successfully created Piper TTS for voice: {voice}")
        except Exception as e:
            logger.error(f"Failed to create Piper TTS: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Piper TTS: {str(e)}")
    
    return piper_tts_cache[voice]

def piper_synthesize(text: str, voice: str, speed: float = 1.0) -> bytes:
    """Synthesize speech using Piper TTS."""
    tts = get_or_create_piper_tts(voice)
    
    # Generate audio
    audio = tts.generate(text, speed=speed)
    
    # Convert to WAV bytes
    sample_rate = tts.sample_rate
    
    # audio.samples is a list, convert to numpy array
    audio_data = np.array(audio.samples, dtype=np.float32)
    
    # Normalize audio to 16-bit range
    audio_16bit = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_16bit.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

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
        "message": "Kokoro & Piper TTS API with Auto Language Detection",
        "version": "2.0.0",
        "features": [
            "Automatic language detection from text",
            "Auto voice selection based on detected language",
            "Support for 11 languages (including Persian and Min-nan)",
            "Multiple voices per language",
            "Dual backend: Kokoro TTS and Piper TTS"
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
                'z': 'Chinese',
                'fa': 'Persian (Farsi)',
                'nan': 'Min-nan (Southern Min/Taiwanese Hokkien)'
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
        
        # Check if this is a Piper voice (Persian)
        if request.voice.startswith("piper_"):
            logger.info(f"Using Piper TTS for voice: {request.voice}")
            wav_bytes = piper_synthesize(request.text, request.voice, request.speed)
        else:
            # Use Kokoro pipeline
            logger.info(f"Using Kokoro TTS for voice: {request.voice}")
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