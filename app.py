"""
FastAPI backend for real-time speech-to-text with AI grammar refinement.

This server provides:
1. WebSocket endpoint for real-time audio streaming and transcription
2. REST endpoint for text-only grammar refinement
3. Integration with Azure Speech Services and Azure OpenAI
"""

import os
import json
import asyncio
import base64
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Configuration
SPEECH_KEY = os.environ.get('SPEECH_KEY')
SPEECH_REGION = os.environ.get('SPEECH_REGION')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')

# Default relevant phrases for grammar correction context
DEFAULT_PHRASES = "Azure Cognitive Services, non-profit organization, speech recognition, OpenAI API"


# Pydantic models for request/response
class RefineRequest(BaseModel):
    text: str
    relevant_phrases: Optional[str] = DEFAULT_PHRASES


class RefineResponse(BaseModel):
    original: str
    refined: str


class TranscriptionResult(BaseModel):
    type: str  # "partial", "final", "error"
    text: str
    refined: Optional[str] = None


# Initialize Azure OpenAI client
openai_client: Optional[AzureOpenAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    global openai_client
    
    # Validate configuration
    missing_vars = []
    if not SPEECH_KEY:
        missing_vars.append("SPEECH_KEY")
    if not SPEECH_REGION:
        missing_vars.append("SPEECH_REGION")
    if not AZURE_OPENAI_ENDPOINT:
        missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing_vars.append("AZURE_OPENAI_API_KEY")
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Some features may not work correctly.")
    
    # Initialize OpenAI client
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        openai_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview"
        )
        print("✅ Azure OpenAI client initialized")
    
    print("🚀 VoiceRefine API server started")
    yield
    print("👋 Server shutting down")


# Create FastAPI app
app = FastAPI(
    title="VoiceRefine API",
    description="Real-time speech-to-text with AI grammar refinement",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rewrite_content(input_text: str, relevant_phrases: str = DEFAULT_PHRASES) -> str:
    """
    Refines the user's input sentence by fixing grammar issues.
    
    Args:
        input_text: The raw input sentence to rewrite.
        relevant_phrases: Context phrases for spelling correction.
    
    Returns:
        The refined sentence.
    """
    if not openai_client:
        return input_text  # Return original if no client
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant to help the user rewrite sentences. "
                "Please fix the grammar errors in the user-provided sentence and make it more readable. "
                "You can do minor rewriting but MUST NOT change the sentence's meaning. "
                "DO NOT make up new content. DO NOT answer questions. "
                f"Here are phrases relevant to the sentences: '{relevant_phrases}'. "
                "If they appear in the sentence and are misspelled, please fix them. "
                "Return ONLY the corrected sentence, nothing else.\n\n"
                "Example corrections:\n"
                "User: how ar you\nYour response: How are you?\n\n"
                "User: what yur name?\nYour response: What's your name?\n\n"
            )
        },
        {"role": "user", "content": input_text}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.3  # Lower temperature for more consistent corrections
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return input_text  # Return original on error


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "VoiceRefine API",
        "features": {
            "speech_to_text": bool(SPEECH_KEY and SPEECH_REGION),
            "grammar_refinement": bool(openai_client)
        }
    }


@app.post("/api/refine", response_model=RefineResponse)
async def refine_text(request: RefineRequest):
    """
    Refine text with AI grammar correction.
    
    Use this endpoint for text-only grammar refinement without speech recognition.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    refined = rewrite_content(request.text, request.relevant_phrases)
    
    return RefineResponse(
        original=request.text,
        refined=refined
    )


@app.get("/api/config")
async def get_config():
    """Get current configuration status (without sensitive data)."""
    return {
        "speech_region": SPEECH_REGION,
        "openai_model": OPENAI_MODEL,
        "speech_configured": bool(SPEECH_KEY),
        "openai_configured": bool(openai_client)
    }


# ============================================================================
# WebSocket Endpoint for Real-time Speech Recognition
# ============================================================================

class SpeechRecognitionSession:
    """Manages a speech recognition session for a WebSocket connection."""
    
    def __init__(self, websocket: WebSocket, relevant_phrases: str = DEFAULT_PHRASES):
        self.websocket = websocket
        self.relevant_phrases = relevant_phrases
        self.speech_recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.is_running = False
        
    async def send_result(self, result_type: str, text: str, refined: Optional[str] = None):
        """Send transcription result to client."""
        await self.websocket.send_json({
            "type": result_type,
            "text": text,
            "refined": refined
        })
    
    def setup_recognizer(self, language: str = "en-US"):
        """Initialize Azure Speech recognizer with microphone input."""
        if not SPEECH_KEY or not SPEECH_REGION:
            raise ValueError("Speech service not configured")
        
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY,
            region=SPEECH_REGION
        )
        speech_config.speech_recognition_language = language
        
        # Use default microphone
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Set up callbacks
        def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs):
            """Handle partial recognition results."""
            if evt.result.text:
                asyncio.run(self.send_result("partial", evt.result.text))
        
        def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs):
            """Handle final recognition results."""
            if evt.result.text:
                refined = rewrite_content(evt.result.text, self.relevant_phrases)
                asyncio.run(self.send_result("final", evt.result.text, refined))
        
        def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            """Handle recognition cancellation."""
            if evt.reason == speechsdk.CancellationReason.Error:
                asyncio.run(self.send_result("error", f"Error: {evt.error_details}"))
        
        self.speech_recognizer.recognizing.connect(on_recognizing)
        self.speech_recognizer.recognized.connect(on_recognized)
        self.speech_recognizer.canceled.connect(on_canceled)
    
    async def start(self, language: str = "en-US"):
        """Start continuous recognition."""
        self.setup_recognizer(language)
        self.is_running = True
        
        # Start recognition
        self.speech_recognizer.start_continuous_recognition_async().get()
        await self.send_result("status", "Recognition started")
    
    async def stop(self):
        """Stop continuous recognition."""
        if self.speech_recognizer and self.is_running:
            self.speech_recognizer.stop_continuous_recognition_async().get()
            self.is_running = False
            await self.send_result("status", "Recognition stopped")


@app.websocket("/ws/speech")
async def websocket_speech(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech recognition.
    
    Client messages:
    - {"action": "start", "language": "en-US", "relevant_phrases": "..."}
    - {"action": "stop"}
    - {"action": "refine", "text": "..."}
    
    Server messages:
    - {"type": "partial", "text": "..."}
    - {"type": "final", "text": "...", "refined": "..."}
    - {"type": "status", "text": "..."}
    - {"type": "error", "text": "..."}
    """
    await websocket.accept()
    session: Optional[SpeechRecognitionSession] = None
    
    try:
        await websocket.send_json({
            "type": "status",
            "text": "Connected to VoiceRefine API"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "start":
                # Start speech recognition
                language = data.get("language", "en-US")
                phrases = data.get("relevant_phrases", DEFAULT_PHRASES)
                
                if not SPEECH_KEY or not SPEECH_REGION:
                    await websocket.send_json({
                        "type": "error",
                        "text": "Speech service not configured on server"
                    })
                    continue
                
                session = SpeechRecognitionSession(websocket, phrases)
                await session.start(language)
                
            elif action == "stop":
                # Stop speech recognition
                if session:
                    await session.stop()
                    session = None
                    
            elif action == "refine":
                # Refine text without speech recognition
                text = data.get("text", "")
                phrases = data.get("relevant_phrases", DEFAULT_PHRASES)
                
                if text:
                    refined = rewrite_content(text, phrases)
                    await websocket.send_json({
                        "type": "final",
                        "text": text,
                        "refined": refined
                    })
                    
            elif action == "ping":
                # Keep-alive ping
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "text": str(e)
            })
        except:
            pass
    finally:
        # Clean up
        if session and session.is_running:
            try:
                await session.stop()
            except:
                pass


# ============================================================================
# Alternative: Push-to-Talk Style Recognition (Single utterance)
# ============================================================================

@app.websocket("/ws/speech-single")
async def websocket_speech_single(websocket: WebSocket):
    """
    WebSocket endpoint for single-utterance speech recognition.
    
    This is simpler than continuous recognition - it recognizes one phrase
    at a time when the client sends a "recognize" action.
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "status",
            "text": "Connected - send 'recognize' to start"
        })
        
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "recognize":
                language = data.get("language", "en-US")
                phrases = data.get("relevant_phrases", DEFAULT_PHRASES)
                
                if not SPEECH_KEY or not SPEECH_REGION:
                    await websocket.send_json({
                        "type": "error",
                        "text": "Speech service not configured"
                    })
                    continue
                
                # Configure speech recognition
                speech_config = speechsdk.SpeechConfig(
                    subscription=SPEECH_KEY,
                    region=SPEECH_REGION
                )
                speech_config.speech_recognition_language = language
                
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config,
                    audio_config=audio_config
                )
                
                await websocket.send_json({
                    "type": "status",
                    "text": "Listening..."
                })
                
                # Perform single recognition
                result = recognizer.recognize_once_async().get()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    refined = rewrite_content(result.text, phrases)
                    await websocket.send_json({
                        "type": "final",
                        "text": result.text,
                        "refined": refined
                    })
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    await websocket.send_json({
                        "type": "error",
                        "text": "No speech recognized"
                    })
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    await websocket.send_json({
                        "type": "error",
                        "text": f"Recognition canceled: {cancellation.reason}"
                    })
                    
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ============================================================================
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)