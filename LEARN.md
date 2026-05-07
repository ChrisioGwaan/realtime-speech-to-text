# Learning Guide — VoiceRefine

This document is for students, educators, and developers who want to understand **how and why** this project is built the way it is. It explains the concepts behind each layer of the stack, the design decisions made, and extension ideas you can use as learning exercises.

---

## What you will learn from this project

| Concept | Where it appears |
|---|---|
| WebSocket protocol | `backend/app.py` ↔ `frontend/index.html` |
| Real-time streaming data | Speech recognition callbacks → browser |
| REST API design with FastAPI | `/api/process`, `/api/config` endpoints |
| API authentication patterns | Access token via first WebSocket message |
| Prompt engineering | System prompts in `extract_smart_data()` and `analyze_realtime()` |
| Structured AI output (JSON mode) | OpenAI returning typed extraction schema |
| Azure cloud services | Azure Speech + Azure OpenAI |
| Environment variable security | `.env`, `python-dotenv`, `secret.py` |
| Async Python | `asyncio`, `async def`, `await` throughout the backend |
| Browser WebSocket API | `new WebSocket()`, `onopen`, `onmessage` in vanilla JS |

---

## Concepts explained

### 1. WebSockets — why not plain HTTP?

Normal HTTP follows a **request → response** cycle. The browser asks, the server replies, and the connection closes. That works fine for loading a page or submitting a form.

Real-time speech is different. Azure Speech sends you partial results every few hundred milliseconds as you speak — words appear before the sentence is finished. HTTP cannot push data to the browser unprompted. You would have to poll (ask "anything new?" every N milliseconds), which is wasteful and adds latency.

**WebSockets** solve this by keeping a persistent, two-way connection open. Once connected, the server can push messages to the browser at any time without the browser asking first. That is why every transcription result, summary update, and status message in VoiceRefine arrives instantly.

```
Browser ──── ws://localhost:8000/ws/speech ────► Backend
        ◄──── {"type": "partial", "text": "Hello"} ────
        ◄──── {"type": "final",   "text": "Hello world.", "extracted": {...}} ────
```

**Key files:** `backend/app.py` → `websocket_speech()`, `frontend/index.html` → `connect()`

---

### 2. The authentication handshake

Most WebSocket tutorials authenticate via a URL query parameter (`?token=abc`). That is convenient but insecure — the token appears in server access logs and browser history.

VoiceRefine uses a **first-message handshake** instead:

1. Browser connects (no token in URL)
2. Browser immediately sends `{"action": "auth", "token": "..."}`
3. Backend checks the token before processing any other message
4. If valid → sends `{"type": "status", "text": "Connected..."}` and continues
5. If invalid → sends error and closes the connection

This keeps the secret inside the encrypted WebSocket frame body and out of every log file.

```python
# backend/app.py
auth_data = await websocket.receive_json()
if auth_data.get("action") != "auth" or auth_data.get("token") != ACCESS_TOKEN:
    await websocket.send_json({"type": "error", "text": "Invalid or missing token"})
    await websocket.close()
    return
```

**Learning exercise:** What other information could the auth message carry? How would you extend this to support multiple users with different tokens?

---

### 3. Async Python and why it matters here

Python normally runs code line by line. If one operation blocks (e.g. waiting for Azure to return a response), nothing else can run.

`async`/`await` lets Python pause a function while it waits for I/O and resume it when the result arrives — without blocking other work. This is critical for a WebSocket server that may be handling multiple connections simultaneously.

```python
# Without async — the whole server would freeze while waiting:
result = openai_client.chat.completions.create(...)

# With async — the server can handle other messages while waiting:
result = await asyncio.to_thread(openai_client.chat.completions.create, ...)
```

FastAPI is built on top of **Starlette** and **asyncio**, so every route handler and WebSocket handler is an async function.

**Learning exercise:** Add `print("before")` and `print("after")` around an `await` call. Use two browser tabs to connect simultaneously. Observe that messages from both connections interleave in the terminal — that is async I/O working.

---

### 4. Prompt engineering — getting structured output from an LLM

Asking an LLM "summarise this text" gives you a paragraph. That is fine for humans to read but hard to use programmatically — you cannot reliably extract `action_items` from a paragraph.

VoiceRefine instructs the model to **always return a specific JSON schema**:

```python
content = """Return a JSON object with this exact structure:
{
    "summary": "...",
    "extracted": {
        "speaker_name": "...",
        "topic": "...",
        "category": "meeting|interview|lecture|...",
        "priority": "high|medium|low",
        "key_points": [...],
        "action_items": [...],
        "notes": "..."
    }
}
Return ONLY valid JSON, no markdown formatting."""
```

This technique is called **structured output prompting**. The model is told:
- Exactly what fields to return
- What the allowed values are for each field (e.g. `"high|medium|low"`)
- Not to include anything outside the JSON (no "Here is your answer:" preamble)

The backend then parses the response with `json.loads()` and maps it to a Pydantic model.

**Learning exercise:** What happens if you remove "Return ONLY valid JSON"? Try it and inspect what the raw response looks like when the model adds markdown code fences.

---

### 5. Accumulating context across a conversation

Each call to Azure Speech returns one sentence at a time. But extracting meaning from a single sentence ("Sure, I can do that by Friday") is nearly impossible without knowing what came before.

VoiceRefine maintains `accumulated_context` — a rolling window of the last 500 words of transcript — and passes it along with each new sentence to the AI:

```python
self.accumulated_context += " " + evt.result.text
words = self.accumulated_context.split()
if len(words) > 500:
    self.accumulated_context = " ".join(words[-500:])
```

This keeps the context window manageable (LLMs have token limits and cost per token) while still giving the model enough history to understand references like "that", "it", or "the deadline we discussed".

**Learning exercise:** What would happen if you used the full transcript instead of a 500-word window? When would that cause problems? How would you handle a 2-hour meeting?

---

### 6. Environment variables and secret management

Hard-coding API keys in source code is one of the most common security mistakes beginners make. Keys committed to a public GitHub repository are found by automated scanners within minutes.

This project uses three layers of protection:

| Layer | How |
|---|---|
| `.env` file | Keeps secrets out of source code |
| `python-dotenv` | Loads `.env` at runtime: `load_dotenv()` |
| `.gitignore` | Prevents `.env` from being committed |

The `secret.py` script generates the access token using Python's `secrets` module, which is designed for cryptographic use (unlike `random`, which is not):

```python
import secrets
token = secrets.token_urlsafe(48)  # 384 bits of entropy
```

**Learning exercise:** Run `git log --all --full-history -- .env` on any public repo that accidentally committed secrets. You will see that deleting the file from the latest commit does not remove it from history — it is permanently in the git object store. This is why `.gitignore` must come first.

---

### 7. FastAPI dependency injection for auth

REST endpoints use FastAPI's `Depends()` system for authentication:

```python
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials.credentials

@app.post("/api/process", dependencies=[Depends(verify_token)])
async def process_text(request: ProcessRequest):
    ...
```

`Depends(verify_token)` means FastAPI automatically calls `verify_token` before the route handler runs. If it raises an `HTTPException`, the route never executes. This is called a **guard** or **middleware dependency** — a clean way to enforce auth without repeating the same check in every route.

**Learning exercise:** Add a new endpoint that does not use `Depends(verify_token)`. Call it from the browser. Now add the dependency. What HTTP status code do you get when the token is wrong?

---

## Project structure walkthrough

```
realtime-speech-to-text/
├── backend/
│   ├── app.py          ← FastAPI server, WebSocket handlers, AI calls
│   ├── secret.py       ← Token generator utility
│   ├── requirements.txt
│   └── .env            ← Your secrets (never commit this)
├── frontend/
│   ├── index.html      ← Entire frontend: HTML + CSS + JS in one file
│   ├── middleware.js    ← Vercel Edge security headers
│   └── package.json
└── assets/             ← Screenshots for README
```

The frontend is intentionally a **single HTML file**. This makes it easy to share — anyone can open it in a browser without installing anything. The trade-off is that all the logic (state management, WebSocket client, UI rendering) lives in one place, which would not scale for a large application.

---

## Extension ideas

These are good starting points for coursework, hackathons, or personal projects:

### Beginner
- Add a **language selector** dropdown to the frontend and pass it in the `start` action so users can transcribe in languages other than English
- Add a **character/word counter** to the transcript panel
- Style the UI differently — try a different colour scheme or font

### Intermediate
- Replace the static access token with a **login form** that issues a short-lived session token (hint: look at `python-jose` for JWT)
- Add a **"Copy to clipboard"** button for each extracted field
- Save the session to `localStorage` so it survives a page refresh

### Advanced
- Add **speaker diarisation** — Azure Speech supports identifying multiple speakers; display each speaker's text in a different colour
- Stream the AI summary back to the frontend word-by-word using the OpenAI streaming API instead of waiting for the full response
- Replace the single-user access token with a proper **multi-user auth system** using FastAPI Users or Supabase Auth

---

## Further reading

- [WebSocket API — MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [FastAPI — official docs](https://fastapi.tiangolo.com/)
- [Azure AI Speech — quickstart](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text)
- [Azure OpenAI — structured outputs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs)
- [Python asyncio — official docs](https://docs.python.org/3/library/asyncio.html)
- [OWASP — Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)

---

## Author

Built by **Chrisio Gwaan** — shared with the education community to demonstrate real-world patterns in real-time AI applications.
