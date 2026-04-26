# HW2 Feature Plus Changes

This version keeps the original HW1 chat UI/features and adds HW2 functions without replacing the base project.

## Preserved HW1 features
- LLM model selection
- Custom system prompt
- API parameters: temperature, top_p, max_tokens
- Streaming / non-streaming
- Short-term memory window
- Chat history, export, token chart, prompt templates, voice input

## Added HW2 features
- Long-term memory using `memory.py` + SQLite
- Multimodal image upload with backend auto-routing to Gemini multimodal
- Text/PDF upload; PDFs are text-extracted with `pypdf` when possible
- Auto routing between models using `router.py`
- Tool use / MCP-style `ToolRegistry` using `tools.py`
- GitHub repo search tool
- PowerPoint generation tool (`create_presentation`) that exports `.pptx`
- Deterministic image-generation routing for image prompts via Gemini Image / Nano Banana
- Agent Mode: visible Plan → Route → Memory/Tool → Answer steps
- TTS output: browser SpeechSynthesis reads assistant responses
- Read-only share links for conversations

## Bug fixes / reliability improvements
- Upload button is no longer disabled when a text model is selected.
- Browser cache busting for `app.js` / `style.css` was updated.
- Image generation prompts are routed directly to the image tool instead of relying on the LLM to call a tool.
- Raw PDF bytes are no longer passed as `image_url`; the backend attempts text extraction instead.
- Shared session tokens are deleted when the source session is deleted.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env
# fill NVIDIA_API_KEY and GEMINI_API_KEY in .env
uvicorn app:app --reload
```

After replacing files, use Ctrl+F5 in the browser to avoid cached JavaScript.
