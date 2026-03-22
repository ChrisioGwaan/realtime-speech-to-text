# VoiceRefine

**Real-time speech capture with AI-powered summary and smart form auto-fill**

VoiceRefine is a prototype web application that demonstrates how live speech can be transcribed and analyzed in real time to support business workflows such as interviews, meetings, presentations, and note-taking sessions. Instead of recording audio first and processing it later, VoiceRefine captures speech as it happens, generates a live summary, and auto-fills structured fields such as speaker name, topic, category, priority, key points, action items, and notes. The project is built with a FastAPI backend, WebSocket-based real-time streaming, Azure Speech Services, and Azure OpenAI / Azure AI Foundry integration.

---

## Why this project

In many real-world scenarios, people need to listen carefully while also documenting important information. This is common in:

- interviews
- stakeholder meetings
- client discovery calls
- internal presentations
- support discussions
- note-heavy review sessions

Traditional audio recording can help, but it still creates extra post-meeting processing work. VoiceRefine explores a more direct workflow: live transcript in one panel, AI-generated structured capture in another, and transcript history for quick review during the conversation itself. The current prototype includes a live summary card, transcript history, a structured “Smart Capture” panel, text mode, speech mode, and downloadable report output.

---

## Core features

- Real-time speech transcription using Azure Speech Services
- AI-powered structured extraction from ongoing transcript
- Live evolving summary of the conversation
- Smart form auto-fill for:
  - speaker name
  - topic / subject
  - category
  - priority
  - key points
  - action items
  - notes / additional context
- Transcript history view with expandable modal
- Text mode for manual input and refinement
- Download captured session as TXT
- Light / dark mode UI
- WebSocket-based real-time communication between frontend and backend

The backend defines REST and WebSocket flows for text processing, speech recognition, structured extraction, and real-time analysis. The extraction schema explicitly includes `speaker_name`, `topic`, `category`, `priority`, `key_points`, `action_items`, and `notes`.

---

## Example use cases

### Interview support
Capture candidate responses in real time while auto-filling structured interview notes.

### Meeting note assistance
Track ongoing discussion and automatically identify important points and action items.

### Presentation capture
Summarize speaker content live and organize information into structured fields for later review.

### Operational documentation
Reduce the overhead of manually turning spoken discussion into usable written records.

---

## Tech stack

### Backend
- FastAPI
- WebSockets
- Azure Cognitive Services Speech SDK
- Azure OpenAI client
- Python
- Uvicorn
- Pydantic

The backend dependencies include `fastapi`, `uvicorn`, `websockets`, `azure-cognitiveservices-speech`, and `openai`, showing the application is designed as an Azure-backed real-time API service.

### Frontend
- HTML
- CSS
- Vanilla JavaScript
- Real-time WebSocket client UI

The current frontend is implemented as a single-page interface with sections for speech mode, text mode, live summary, transcript history, and smart form capture.

### Azure services
- Azure Speech Services
- Azure OpenAI / Azure AI Foundry

The backend reads Azure credentials and deployment configuration from environment variables including `SPEECH_KEY`, `SPEECH_REGION`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_AI_DEPLOYMENT`.

---

## Architecture overview

```text
Microphone / Text Input
        ↓
Frontend Web App
        ↓  (WebSocket / HTTP)
FastAPI Backend
        ↓
Azure Speech Services  → live transcription
        ↓
Azure OpenAI / AI Foundry → summary + structured extraction
        ↓
Frontend Smart Capture UI
        ↓
User review / download / transcript history