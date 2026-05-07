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
class ProcessRequest(BaseModel):
    text: str
    relevant_phrases: Optional[str] = DEFAULT_PHRASES


class ExtractedData(BaseModel):
    speaker_name: Optional[str] = None
    topic: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    key_points: List[str] = []
    action_items: List[str] = []
    notes: Optional[str] = None


class ProcessResponse(BaseModel):
    original: str
    extracted: ExtractedData


class TranscriptionResult(BaseModel):
    type: str  # "partial", "final", "error"
    text: str
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


def analyze_realtime(full_transcript: str, previous_summary: str = "", relevant_phrases: str = DEFAULT_PHRASES) -> Dict[str, Any]:
    """
    Comprehensive real-time analysis that generates summary and extracts form data.
    Designed to be called periodically as speech comes in.
    
    Args:
        full_transcript: The complete transcript so far.
        previous_summary: The previous summary to build upon.
        relevant_phrases: Context phrases for better analysis.
    
    Returns:
        Dictionary with 'summary' and 'extracted' data.
    """
    if not openai_client or not full_transcript.strip():
        return {"summary": "", "extracted": {}}
    
    messages = [
        {
            "role": "system",
            "content": f"""You are an intelligent real-time speech analyst. Your job is to analyze ongoing speech and provide:

1. A LIVE SUMMARY - A concise, evolving summary of what's being discussed (2-4 sentences)
2. STRUCTURED DATA - Extract key information for form auto-filling

The speech is ongoing, so your summary should capture the essence so far and be ready to evolve.

{f"Previous summary to build upon: '{previous_summary}'" if previous_summary else "This is the start of the conversation."}

Context phrases that may be relevant: {relevant_phrases}

Return a JSON object with this exact structure:
{{
    "summary": "A concise 2-4 sentence summary of the conversation so far. Make it informative and capture the key points.",
    "extracted": {{
        "speaker_name": "Name if mentioned (null if not found)",
        "topic": "Main topic in 5-10 words (null if unclear)",
        "category": "meeting|interview|lecture|brainstorm|support|personal|technical|other",
        "priority": "high|medium|low (null if no urgency indicators)",
        "key_points": ["Array of key points mentioned so far"],
        "action_items": ["Array of tasks or to-dos mentioned"],
        "notes": "Any additional context worth noting"
    }}
}}

Guidelines for extraction:
- Be intelligent about inference - understand context, don't just pattern match
- speaker_name: Names can be mentioned in many ways ("I'm John", "This is Sarah", etc.)
- topic: What is this conversation fundamentally about?
- category: Choose the best fit based on overall content
- priority: Look for urgency words, deadlines, importance indicators
- key_points: Important information, decisions, facts (not just any sentence)
- action_items: Things that need to be done, tasks, follow-ups
- notes: Interesting context that doesn't fit elsewhere

Be concise but comprehensive. Return ONLY valid JSON."""
        },
        {
            "role": "user",
            "content": f"Analyze this ongoing speech transcript:\n\n{full_transcript}"
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
        if "summary" not in result:
            result["summary"] = ""
        if "extracted" not in result:
            result["extracted"] = {}
        if "key_points" not in result.get("extracted", {}) or result["extracted"].get("key_points") is None:
            result["extracted"]["key_points"] = []
        if "action_items" not in result.get("extracted", {}) or result["extracted"].get("action_items") is None:
            result["extracted"]["action_items"] = []
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in realtime analysis: {e}")
        return {"summary": previous_summary, "extracted": {}}
    except Exception as e:
        print(f"Realtime analysis error: {e}")
        return {"summary": previous_summary, "extracted": {}}


def process_speech_text(input_text: str, relevant_phrases: str = DEFAULT_PHRASES, accumulated_context: str = "") -> Dict[str, Any]:
    """
    Process speech text and extract structured data.
    
    Args:
        input_text: The raw input text to process.
        relevant_phrases: Context phrases for better processing.
        accumulated_context: Previous context for better understanding.
    
    Returns:
        Dictionary with 'text' and 'extracted' data.
    """
    if not openai_client:
        return {"text": input_text, "extracted": {}}
    
    # Extract structured data
    extracted = extract_smart_data(input_text, accumulated_context)
    
    return {"text": input_text, "extracted": extracted}


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
            "smart_extraction": bool(openai_client)
        }
    }


@app.post("/api/process", response_model=ProcessResponse, dependencies=[Depends(verify_token)])
async def process_text(request: ProcessRequest):
    """
    Process text and extract structured data.
    
    Use this endpoint for text-based smart form filling.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = process_speech_text(request.text, request.relevant_phrases)
    
    return ProcessResponse(
        original=request.text,
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
        self.accumulated_context = ""
        
    async def send_result(self, result_type: str, text: str, extracted: Optional[Dict] = None):
        """Send transcription result to client."""
        message = {
            "type": result_type,
            "text": text
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
                result = process_speech_text(
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
        self.accumulated_context = ""
        
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
async def websocket_speech(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech recognition with smart extraction.
    
    Client messages (first message must be auth):
    - {"action": "auth", "token": "..."}
    - {"action": "start", "language": "en-US", "relevant_phrases": "..."}
    - {"action": "stop"}
    - {"action": "process", "text": "...", "context": "..."}
    - {"action": "analyze", "text": "...", "previous_summary": "...", "relevant_phrases": "..."}
    - {"action": "clear_context"}
    
    Server messages:
    - {"type": "partial", "text": "..."}
    - {"type": "final", "text": "...", "extracted": {...}}
    - {"type": "analysis", "summary": "...", "extracted": {...}}
    - {"type": "status", "text": "..."}
    - {"type": "error", "text": "..."}
    """
    await websocket.accept()
    session: Optional[SpeechRecognitionSession] = None
    accumulated_context = ""  # For text mode
    current_summary = ""  # Track current summary
    
    try:
        # First message must be auth
        auth_data = await websocket.receive_json()
        if auth_data.get("action") != "auth" or not ACCESS_TOKEN or auth_data.get("token") != ACCESS_TOKEN:
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
                    
            elif action == "refine" or action == "process":
                # Process text with smart extraction (text mode)
                text = data.get("text", "")
                phrases = data.get("relevant_phrases", DEFAULT_PHRASES)
                context = data.get("context", accumulated_context)
                
                if text:
                    result = process_speech_text(text, phrases, context)
                    
                    # Update accumulated context
                    accumulated_context += " " + text
                    words = accumulated_context.split()
                    if len(words) > 500:
                        accumulated_context = " ".join(words[-500:])
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": text,
                        "extracted": result.get("extracted", {})
                    })
            
            elif action == "analyze":
                # Real-time analysis request - generates summary and extracts data
                text = data.get("text", "")
                previous_summary = data.get("previous_summary", current_summary)
                phrases = data.get("relevant_phrases", DEFAULT_PHRASES)
                
                if text:
                    result = analyze_realtime(text, previous_summary, phrases)
                    
                    # Update current summary
                    if result.get("summary"):
                        current_summary = result["summary"]
                    
                    await websocket.send_json({
                        "type": "analysis",
                        "summary": result.get("summary", ""),
                        "extracted": result.get("extracted", {})
                    })
            
            elif action == "clear_context":
                # Clear accumulated context
                accumulated_context = ""
                current_summary = ""
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
    First message must be: {"action": "auth", "token": "..."}
    """
    await websocket.accept()
    accumulated_context = ""
    
    try:
        # First message must be auth
        auth_data = await websocket.receive_json()
        if auth_data.get("action") != "auth" or not ACCESS_TOKEN or auth_data.get("token") != ACCESS_TOKEN:
            await websocket.send_json({
                "type": "error",
                "text": "Invalid or missing token"
            })
            await websocket.close()
            return

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
                    processed = process_speech_text(result.text, phrases, accumulated_context)
                    
                    # Update context
                    accumulated_context += " " + result.text
                    words = accumulated_context.split()
                    if len(words) > 500:
                        accumulated_context = " ".join(words[-500:])
                    
                    await websocket.send_json({
                        "type": "final",
                        "text": result.text,
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
