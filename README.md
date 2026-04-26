# HW02 — My Very Powerful Chatbot

> 作者：112550190 劉彥廷  
> 本專案以 HW01「Your own ChatGPT」為基礎，升級成支援 **RAG 文件問答、長期記憶、多模態輸入、自動模型路由、Tool Use / MCP、簡報生成與圖片生成** 的聊天機器人系統。

---

## 1. Project Overview

This project is a full-stack chatbot web application built with **FastAPI + HTML/CSS/JavaScript**.  
The backend connects to **NVIDIA NIM** through an OpenAI-compatible API and also integrates **Gemini API** for multimodal and image-related tasks.

Compared with HW01, this version adds:

- RAG document question answering
- Long-term memory
- Multimodal image / file upload
- Auto routing between models and tools
- MCP-style tool server and tool manager UI
- PowerPoint generation
- Image generation
- Agent mode with visible reasoning workflow
- TTS response reading and voice input (No demo because it is too noisy)

---

## 2. Main Features

### 2.1 Basic ChatGPT Functions

- Select LLM model from the UI
- Customize system prompt
- Adjust common API parameters:
  - temperature
  - top_p
  - max_tokens
- Enable or disable streaming response
- Enable short-term memory and adjust memory window size
- Save and reload chat sessions through SQLite
- Export current conversation as `.md` or `.json`
- Display token usage and token chart
- Support dark / light theme, accent color, and font size adjustment

---

### 2.2 RAG Document Question Answering

This project includes a custom RAG pipeline in `rag.py`.

Supported sources:

- PDF files
- TXT / Markdown / code files
- DOCX files
- Web URLs

RAG workflow:

1. User uploads a document or pastes a URL in the RAG panel.
2. Backend extracts readable text from the source.
3. Text is split into overlapping chunks.
4. Each chunk is embedded by NVIDIA embedding API.
5. If NVIDIA embedding is unavailable, the system falls back to a local TF-IDF-style embedding.
6. Chunks and embeddings are stored in SQLite.
7. When the user asks a question, the system retrieves the most relevant chunks within the current session.
8. Retrieved context is injected into the system prompt before generating the final answer.

Important implementation details:

| Item | Implementation |
|---|---|
| RAG module | `rag.py` |
| Database | `chat_history.db` |
| Tables | `rag_documents`, `rag_chunks` |
| Embedding model | `nvidia/nv-embedqa-e5-v5` |
| Fallback embedding | local TF-IDF hash vector |
| Chunk size | 500 characters |
| Chunk overlap | 80 characters |
| Retrieval | cosine similarity |
| Default top-k | 4 chunks |

RAG-related API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/rag/docs?session_id=...` | List RAG documents for a session |
| POST | `/api/rag/ingest?session_id=...` | Upload and index a document |
| POST | `/api/rag/ingest?session_id=...&url=...` | Fetch and index a URL |
| DELETE | `/api/rag/docs/{doc_id}` | Delete a RAG document and its chunks |

---

### 2.3 Long-Term Memory

The long-term memory module is implemented in `memory.py` and stored in SQLite.

The system can:

- Automatically detect user preferences or important facts
- Store memory as `profile`, `episodic`, or `semantic`
- Retrieve relevant memories before answering
- Inject memory context into the system prompt
- Let the user view and delete saved memories in the UI

Memory-related API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/memories` | List saved memories |
| POST | `/api/memories` | Manually add a memory |
| GET | `/api/memories/search?q=...` | Search memory by query |
| DELETE | `/api/memories/{memory_id}` | Delete a memory |

---

### 2.4 Multimodal Input

The app supports file upload through `/api/upload`.

Supported input types:

- Images: converted to base64 and sent to a vision-capable model
- Text / JSON / code files: decoded as text and appended as context
- PDF files: backend attempts text extraction before sending to the model

Vision-capable models used in the project:

- `gemini/gemini-2.5-flash`
- `gemini/gemini-2.5-pro`
- `meta/llama-3.2-11b-vision-instruct`
- `microsoft/phi-3.5-vision-instruct`

---

### 2.5 Auto Routing Between Models

The routing logic is implemented in `router.py`.

The router detects the task type and automatically chooses a suitable model or tool. For example:

| Task Type | Route |
|---|---|
| General chat | NVIDIA `openai/gpt-oss-120b` |
| Simple / low-cost tasks | NVIDIA `openai/gpt-oss-20b` |
| Long text analysis / summary | Gemini 2.5 Pro |
| Coding tasks | Gemini 2.5 Flash |
| Image input | Gemini multimodal model |
| Image generation | Gemini image tool |
| Math calculation | calculator tool |
| Weather / date / search | corresponding tool |
| Presentation generation | PowerPoint tool |
| YouTube summary | YouTube transcript tool |

The UI displays the selected route and reason so the user can see why a model or tool was chosen.

---

### 2.6 Tool Use and MCP Server

Tools are implemented in `tools.py` through a `ToolRegistry`.  
The MCP-style server is implemented in `mcp_server.py`.

Main supported tools:

- Web search
- GitHub repository search
- Calculator
- Weather lookup
- Date / time lookup
- Image generation
- PowerPoint generation
- URL fetch
- Unit converter
- Note manager
- QR code generator
- Timezone converter
- Hash generator
- Color converter
- Dictionary lookup
- Random generator
- YouTube transcript extraction

MCP-related endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/mcp` | JSON-RPC MCP endpoint |
| GET | `/mcp/sse` | SSE stream for real-time tool events |
| GET | `/mcp/tools` | List all registered tools |
| POST | `/mcp/tools/{name}/invoke` | Invoke a tool directly |
| POST | `/mcp/tools/{name}/toggle` | Enable or disable a tool |
| GET | `/mcp/logs` | View recent tool call logs |
| DELETE | `/mcp/logs` | Clear tool call logs |
| GET | `/mcp/health` | MCP server health check |

The standalone tool manager UI can be opened at:

```text
http://127.0.0.1:8000/mcp-ui
```

---

### 2.7 Agent Mode

Agent mode shows a visible task-solving pipeline in the UI:

```text
Analyze Task → Decide Actions → Tool Call → Synthesize → Verify → Final Answer
```

This makes the model routing and tool-use process easier to demonstrate during the homework demo.

---

### 2.8 Presentation and Image Generation

The project supports automatic PowerPoint generation through the `create_presentation` tool.

Generated files are saved under:

```text
static/generated/
```

Download endpoint:

```text
/api/download/{filename}
```

The project also supports image generation through Gemini image-related routing and the `generate_image` tool.

---

## 3. System Architecture

```text
User Browser
    |
    |  HTML / CSS / JavaScript
    v
FastAPI Backend (app.py)
    |
    |-- Session Manager / SQLite Chat History
    |-- Prompt Template Manager
    |-- Short-Term Memory
    |-- Long-Term Memory (memory.py)
    |-- RAG Manager (rag.py)
    |       |-- Text Extraction
    |       |-- Chunking
    |       |-- Embedding
    |       |-- Vector Retrieval
    |
    |-- Router (router.py)
    |       |-- Model Selection
    |       |-- Task Type Detection
    |       |-- Tool Routing
    |
    |-- Tool Registry (tools.py)
    |       |-- Web Search
    |       |-- GitHub Search
    |       |-- Calculator
    |       |-- Weather
    |       |-- PPT Generation
    |       |-- Image Generation
    |       |-- YouTube Transcript
    |
    |-- MCP Server (mcp_server.py)
    |
    v
External APIs
    |-- NVIDIA NIM API
    |-- NVIDIA Embedding API
    |-- Gemini API
    |-- DuckDuckGo Search
    |-- GitHub API
    |-- Weather / QR / YouTube-related APIs
```

---

## 4. Project Structure

```text
.
├── app.py                    # FastAPI main backend
├── router.py                 # Rule-based auto model / tool router
├── rag.py                    # RAG document ingestion, embedding, retrieval
├── memory.py                 # Long-term memory manager
├── tools.py                  # MCP-style tool registry and tool implementations
├── mcp_server.py             # MCP JSON-RPC / SSE server
├── tool_cache.py             # Tool result cache
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
├── chat_history.db           # SQLite database for sessions, memory, RAG, logs
├── templates/
│   └── index.html            # Main web UI
├── static/
│   ├── app.js                # Frontend interaction logic
│   ├── style.css             # UI styling
│   └── generated/            # Generated PPT files
├── mcp_ui.html               # MCP tool manager UI
└── README_HW2_CHANGES.md     # Summary of HW2 changes
```

---

## 5. Installation

### Step 1: Create virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

For full RAG / DOCX / URL / YouTube support, install optional packages:

```bash
pip install pymupdf pdfplumber mammoth python-docx beautifulsoup4 youtube-transcript-api
```

---

## 6. Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then fill in the required API keys:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta

DEFAULT_MODEL=openai/gpt-oss-120b
GEMINI_DEFAULT_MODEL=gemini/gemini-2.5-flash
GEMINI_PRO_MODEL=gemini/gemini-2.5-pro
GEMINI_IMAGE_MODEL=gemini-2.5-flash-image-preview

PORT=8000
```

Optional:

```env
GITHUB_TOKEN=your_github_token_here
NVIDIA_EMBED_MODEL=nvidia/nv-embedqa-e5-v5
```

Security note:

- Do not commit `.env` to GitHub.
- API keys are only used by the backend.
- Frontend code should not contain API keys.

---

## 7. Run the Project

```bash
uvicorn app:app --reload
```

Open the browser:

```text
http://127.0.0.1:8000
```

MCP tool manager:

```text
http://127.0.0.1:8000/mcp-ui
```

---

## 8. Demo Guide

A suggested demo flow:

1. Open the web page and show model selection.
2. Modify system prompt and API parameters.
3. Turn streaming on and ask a normal chat question.
4. Enable auto routing and ask different task types:
   - coding question
   - calculation question
   - weather or date question
   - web search question
5. Upload an image and ask the chatbot to analyze it.
6. Upload a PDF / TXT / DOCX file into the RAG panel.
7. Ask questions related to the uploaded document and show that the answer uses retrieved context.
8. Paste a URL into the RAG URL box and ask questions about the webpage.
9. Turn on long-term memory and demonstrate preference recall.
10. Open MCP Tools Manager and show available tools / tool logs.
11. Ask the bot to generate a PowerPoint file and download the `.pptx`.
12. Demonstrate Agent Mode and show the visible Plan → Route → Tool → Answer process.

---

## 9. Main API Endpoints

| Category | Method | Endpoint | Description |
|---|---|---|---|
| Page | GET | `/` | Main chatbot UI |
| Models | GET | `/api/models` | List available models |
| Health | GET | `/api/health` | Backend health check |
| Chat | POST | `/api/chat` | Non-streaming chat |
| Chat | POST | `/api/chat/stream` | Streaming chat with NDJSON |
| Upload | POST | `/api/upload` | Upload image / file |
| Sessions | GET | `/api/sessions` | List chat sessions |
| Sessions | POST | `/api/sessions` | Create new session |
| Sessions | GET | `/api/sessions/{session_id}/messages` | Get session messages |
| Sessions | PATCH | `/api/sessions/{session_id}` | Rename session |
| Sessions | DELETE | `/api/sessions/{session_id}` | Delete session |
| Share | POST | `/api/sessions/{session_id}/share` | Create read-only share link |
| Share | GET | `/share/{token}` | View shared conversation |
| Memory | GET | `/api/memories` | List long-term memories |
| Memory | POST | `/api/memories` | Add memory manually |
| Tools | GET | `/api/tools` | List registered tools |
| RAG | GET | `/api/rag/docs` | List RAG documents |
| RAG | POST | `/api/rag/ingest` | Ingest file or URL |
| RAG | DELETE | `/api/rag/docs/{doc_id}` | Delete indexed document |
| Download | GET | `/api/download/{filename}` | Download generated PPTX |
| MCP | POST | `/mcp` | MCP JSON-RPC endpoint |
| MCP | GET | `/mcp/tools` | List MCP tools |
| MCP | GET | `/mcp/sse` | MCP SSE event stream |

---

## 10. Implementation Notes

- The backend normalizes errors into JSON responses for easier frontend handling.
- Chat history, long-term memory, RAG documents, and RAG chunks are stored in local SQLite.
- RAG is session-scoped, so each chat session can have its own uploaded documents.
- The router can be disabled from the UI if the user wants to manually choose a model.
- Tool usage can also be disabled from the UI.
- When tool calling is enabled, the backend emits tool events so the frontend can display progress.
- Generated PowerPoint files are served through `/api/download/{filename}`.
- The project keeps API keys in `.env`, not in frontend JavaScript.
