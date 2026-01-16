"""
FastAPI backend for real-time speech-to-text with AI grammar refinement.

This server provides:
1. WebSocket endpoint for real-time audio streaming and transcription
2. REST endpoint for text-only grammar refinement
3. Integration with Azure Speech Services and Azure OpenAI
4. AI-powered smart extraction for form auto-filling
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
OPENAI_MODEL = os.environ.get('AZURE_AI_DEPLOYMENT', 'gpt-5.2-chat')
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')  # For API authentication

# Default relevant phrases for grammar correction context
DEFAULT_PHRASES = "Azure Cognitive Services, non-profit organization, speech recognition, OpenAI API"


# Pydantic models for request/response
class RefineRequest(BaseModel):
    text: str
    relevant_phrases: Optional[str] = DEFAULT_PHRASES


class RefineResponse(BaseModel):
    original: str
    refined: str


class ExtractedData(BaseModel):
    speaker_name: Optional[str] = None
    topic: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    key_points: List[str] = []
    action_items: List[str] = []
    notes: Optional[str] = None


class SmartRefineResponse(BaseModel):
    original: str
    refined: str
    extracted: ExtractedData


class TranscriptionResult(BaseModel):
    type: str  # "partial", "final", "error"
    text: str
    refined: Optional[str] = None
    extracted: Optional[Dict[str, Any]] = None


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
    if not ACCESS_TOKEN:
        missing_vars.append("ACCESS_TOKEN")
    
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
    description="Real-time speech-to-text with AI grammar refinement and smart extraction",
    version="2.0.0",
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

# OAuth2 Bearer Token security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token."""
    token = credentials.credentials
    if not ACCESS_TOKEN or token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return token


def rewrite_content(input_text: str, relevant_phrases: str = DEFAULT_PHRASES) -> str:
    """
    Translates the user's input text to Chinese.
    
    Args:
        input_text: The raw input text to translate.
        relevant_phrases: Context phrases for better translation.
    
    Returns:
        The translated text in Chinese.
    """
    if not openai_client:
        return input_text  # Return original if no client
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that translates text to Chinese. "
                "Translate the user's input to natural, fluent Chinese. "
                "Keep the meaning accurate and context appropriate. "
                f"Relevant phrases for context: '{relevant_phrases}'. "
                "Return ONLY the translated text, nothing else."
            )
        },
        {"role": "user", "content": input_text}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return input_text  # Return original on error


def extract_smart_data(input_text: str, accumulated_context: str = "") -> Dict[str, Any]:
    """
    Uses AI to intelligently extract structured data from speech text.
    
    Args:
        input_text: The current speech text to analyze.
        accumulated_context: Previous context for better understanding.
    
    Returns:
        Dictionary with extracted structured data.
    """
    if not openai_client:
        return {}
    
    full_context = f"{accumulated_context}\n{input_text}".strip()
    
    messages = [
        {
            "role": "system",
            "content": """You are an intelligent assistant that extracts structured information from speech transcriptions.

Analyze the provided text and extract the following information if present. Be smart about inference - don't just look for exact phrases, understand the context.

Return a JSON object with these fields (use null for fields you can't determine):
{
    "speaker_name": "Name of the person speaking (if they introduce themselves or are mentioned)",
    "topic": "Main topic or subject being discussed (brief, 5-10 words max)",
    "category": "One of: meeting, interview, lecture, brainstorm, support, personal, technical, other",
    "priority": "One of: high, medium, low (based on urgency cues, deadlines, importance)",
    "key_points": ["Array of main points or important information mentioned"],
    "action_items": ["Array of tasks, to-dos, or things that need to be done"],
    "notes": "Any other relevant context or information worth noting"
}

Guidelines:
- speaker_name: Look for "my name is", "I'm", "this is X speaking", or contextual references
- topic: Identify the main subject - what is this conversation/speech about?
- category: 
  * "meeting" - agenda items, attendees, minutes, schedules
  * "interview" - candidate discussions, job-related, hiring
  * "lecture" - educational content, presentations, teaching
  * "brainstorm" - ideas, creativity, "what if", possibilities
  * "support" - customer issues, problems, troubleshooting
  * "personal" - personal matters, reminders, daily tasks
  * "technical" - code, systems, technical discussions
  * "other" - doesn't fit other categories
- priority:
  * "high" - urgent, ASAP, critical, emergency, deadline today/tomorrow
  * "medium" - important, soon, this week, should prioritize
  * "low" - when possible, no rush, eventually
- key_points: Important facts, decisions, or information stated
- action_items: Tasks prefixed with "need to", "must", "should", "will", "going to", "please", "make sure", "don't forget", "remember to"
- notes: Context that doesn't fit elsewhere but is valuable

Only include fields where you have reasonable confidence. For arrays, include only clear items.
Return ONLY valid JSON, no markdown formatting or explanation."""
        },
        {
            "role": "user",
            "content": f"Extract structured information from this speech text:\n\n{full_context}"
        }
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        extracted = json.loads(result_text.strip())
        
        # Ensure arrays are lists
        if "key_points" not in extracted or extracted["key_points"] is None:
            extracted["key_points"] = []
        if "action_items" not in extracted or extracted["action_items"] is None:
            extracted["action_items"] = []
            
        return extracted
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {}
    except Exception as e:
        print(f"Smart extraction error: {e}")
        return {}


def smart_rewrite_and_extract(input_text: str, relevant_phrases: str = DEFAULT_PHRASES, accumulated_context: str = "") -> Dict[str, Any]:
    """
    Combined function that both refines text and extracts structured data.
    Makes a single API call for efficiency.
    
    Args:
        input_text: The raw input text to process.
        relevant_phrases: Context phrases for better processing.
        accumulated_context: Previous context for better understanding.
    
    Returns:
        Dictionary with 'refined' text and 'extracted' data.
    """
    if not openai_client:
        return {"refined": input_text, "extracted": {}}
    
    full_context = f"{accumulated_context}\n{input_text}".strip() if accumulated_context else input_text
    
    messages = [
        {
            "role": "system",
            "content": f"""You are a bilingual assistant that performs two tasks:

TASK 1 - TRANSLATION:
Translate the user's input to natural, fluent Chinese. Keep meaning accurate.
Relevant context phrases: '{relevant_phrases}'

TASK 2 - SMART EXTRACTION:
Extract structured information for form auto-filling.

Return a JSON object with this exact structure:
{{
    "refined": "The Chinese translation of the input",
    "extracted": {{
        "speaker_name": "Name if mentioned (null if not)",
        "topic": "Main topic in 5-10 words (null if unclear)",
        "category": "meeting|interview|lecture|brainstorm|support|personal|technical|other (null if unclear)",
        "priority": "high|medium|low based on urgency (null if no urgency cues)",
        "key_points": ["Important points mentioned"],
        "action_items": ["Tasks or to-dos mentioned"],
        "notes": "Other relevant context (null if none)"
    }}
}}

Category guidelines:
- meeting: agenda, attendees, minutes, schedules
- interview: candidates, hiring, job discussions
- lecture: educational, presentations, teaching
- brainstorm: ideas, creativity, possibilities
- support: customer issues, troubleshooting
- personal: personal matters, daily tasks
- technical: code, systems, technical topics

Priority guidelines:
- high: urgent, ASAP, critical, today/tomorrow deadlines
- medium: important, this week, should prioritize
- low: when possible, no rush

Return ONLY valid JSON, no markdown or explanation."""
        },
        {
            "role": "user",
            "content": f"Process this text:\n\n{input_text}\n\nFull context for extraction:\n{full_context}"
        }
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result = json.loads(result_text.strip())
        
        # Ensure required structure
        if "refined" not in result:
            result["refined"] = input_text
        if "extracted" not in result:
            result["extracted"] = {}
        if "key_points" not in result["extracted"] or result["extracted"]["key_points"] is None:
            result["extracted"]["key_points"] = []
        if "action_items" not in result["extracted"] or result["extracted"]["action_items"] is None:
            result["extracted"]["action_items"] = []
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        # Fallback to basic translation
        refined = rewrite_content(input_text, relevant_phrases)
        return {"refined": refined, "extracted": {}}
    except Exception as e:
        print(f"Smart processing error: {e}")
        return {"refined": input_text, "extracted": {}}


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "VoiceRefine API",
        "version": "2.0.0",
        "features": {
            "speech_to_text": bool(SPEECH_KEY and SPEECH_REGION),
            "grammar_refinement": bool(openai_client),
            "smart_extraction": bool(openai_client)
        }
    }


@app.post("/api/refine", response_model=RefineResponse, dependencies=[Depends(verify_token)])
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


@app.post("/api/smart-refine", response_model=SmartRefineResponse, dependencies=[Depends(verify_token)])
async def smart_refine_text(request: RefineRequest):
    """
    Refine text and extract structured data in one call.
    
    Use this endpoint for combined translation and smart form filling.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = smart_rewrite_and_extract(request.text, request.relevant_phrases)
    
    return SmartRefineResponse(
        original=request.text,
        refined=result.get("refined", request.text),
        extracted=ExtractedData(**result.get("extracted", {}))
    )


@app.get("/api/config", dependencies=[Depends(verify_token)])
async def get_config():
    """Get current configuration status (without sensitive data)."""
    return {
        "speech_region": SPEECH_REGION,
        "openai_model": OPENAI_MODEL,
        "speech_configured": bool(SPEECH_KEY),
        "openai_configured": bool(openai_client),
        "smart_extraction_enabled": bool(openai_client)
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
        self.accumulated_context = ""  # Store context for smarter extraction
        
    async def send_result(self, result_type: str, text: str, refined: Optional[str] = None, extracted: Optional[Dict] = None):
        """Send transcription result to client."""
        message = {
            "type": result_type,
            "text": text,
            "refined": refined
        }
        if extracted:
            message["extracted"] = extracted
        await self.websocket.send_json(message)
    
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
                # Use smart extraction with accumulated context
                result = smart_rewrite_and_extract(
                    evt.result.text, 
                    self.relevant_phrases,
                    self.accumulated_context
                )
                
                # Update accumulated context for future extractions
                self.accumulated_context += " " + evt.result.text
                # Keep context manageable (last ~500 words)
                words = self.accumulated_context.split()
                if len(words) > 500:
                    self.accumulated_context = " ".join(words[-500:])
                
                asyncio.run(self.send_result(
                    "final", 
                    evt.result.text, 
                    result.get("refined"),
                    result.get("extracted")
                ))
        
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
        self.accumulated_context = ""  # Reset context on new session
        
        # Start recognition
        self.speech_recognizer.start_continuous_recognition_async().get()
        await self.send_result("status", "Recognition started")
    
    async def stop(self):
        """Stop continuous recognition."""
        if self.speech_recognizer and self.is_running:
            self.speech_recognizer.stop_continuous_recognition_async().get()
            self.is_running = False
            await self.send_result("status", "Recognition stopped")
    
    def clear_context(self):
        """Clear accumulated context."""
        self.accumulated_context = ""


@app.websocket("/ws/speech")
async def websocket_speech(websocket: WebSocket, token: str = Query(...)):
    """
    WebSocket endpoint for real-time speech recognition with smart extraction.
    
    Client messages:
    - {"action": "start", "language": "en-US", "relevant_phrases": "..."}
    - {"action": "stop"}
    - {"action": "refine", "text": "...", "context": "..."}
    - {"action": "clear_context"}
    
    Server messages:
    - {"type": "partial", "text": "..."}
    - {"type": "final", "text": "...", "refined": "...", "extracted": {...}}
    - {"type": "status", "text": "..."}
    - {"type": "error", "text": "..."}
    """
    await websocket.accept()
    session: Optional[SpeechRecognitionSession] = None
    accumulated_context = ""  # For text mode
    
    try:
        # Verify token
        if not ACCESS_TOKEN or token != ACCESS_TOKEN:
            await websocket.send_json({
                "type": "error",
                "text": "Invalid or missing token"
            })
            await websocket.close()
            return
        
        await websocket.send_json({
            "type": "status",
            "text": "Connected to VoiceRefine API v2.0"
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
                # Refine text with smart extraction (text mode)
                text = data.get("text", "")
                phrases = data.get("relevant_phrases", DEFAULT_PHRASES)
                context = data.get("context", accumulated_context)
                
                if text:
                    result = smart_rewrite_and_extract(text, phrases, context)
                    
                    # Update accumulated context
                    accumulated_context += " " + text
                    words = accumulated_context.split()
                    if len(words) > 500:
                        accumulated_context = " ".join(words[-500:])
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": text,
                        "refined": result.get("refined"),
                        "extracted": result.get("extracted", {})
                    })
            
            elif action == "clear_context":
                # Clear accumulated context
                accumulated_context = ""
                if session:
                    session.clear_context()
                await websocket.send_json({
                    "type": "status",
                    "text": "Context cleared"
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
    accumulated_context = ""
    
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
                    # Use smart extraction
                    processed = smart_rewrite_and_extract(result.text, phrases, accumulated_context)
                    
                    # Update context
                    accumulated_context += " " + result.text
                    words = accumulated_context.split()
                    if len(words) > 500:
                        accumulated_context = " ".join(words[-500:])
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": result.text,
                        "refined": processed.get("refined"),
                        "extracted": processed.get("extracted", {})
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
            
            elif action == "clear_context":
                accumulated_context = ""
                await websocket.send_json({
                    "type": "status",
                    "text": "Context cleared"
                })
                    
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ============================================================================
# Run with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
