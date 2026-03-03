# AI Medical Image Generator

Turn clinical ideas into exam-ready medical illustrations. The app uses an LLM (with optional RAG) to craft image prompts and Google Gemini to generate and edit images.

---

## Quick start

1. **Clone and enter the project**
   ```bash
   git clone <your-repo-url>
   cd image-generator
   ```

2. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Configure environment**  
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key
   # Optional for RAG: MONGODB_URI=your-mongodb-uri
   ```
   Do not commit `.env`; it should be in `.gitignore`.

4. **Run the app**
   ```bash
   python server.py
   ```

5. **Open in browser**  
   **http://localhost:5001**  
   (Port can be overridden with the `PORT` environment variable.)

---

## Image generation flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User question  │────▶│  Prompt builder  │────▶│  Image prompt   │
│  (medical topic)│     │  (LLM + RAG)     │     │  (editable)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
        │                            │                      │
        │                            │                      ▼
        │                            │             ┌─────────────────┐
        │                            │             │  Gemini         │
        │                            │             │  (generate /    │
        │                            │             │   edit image)   │
        │                            │             └────────┬────────┘
        │                            │                      │
        ▼                            ▼                      ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ System instruction│    │ Vector store     │     │  Image URL      │
│ (how prompts     │     │ (MongoDB Atlas)  │     │  (preview +     │
│  are crafted)   │     │ optional context │     │   download)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

- **System instruction** + **user question** → sent to backend.
- **Backend**: optionally retrieves context from the vector store, then calls the LLM to produce an **image prompt**.
- You can **edit the prompt** in the UI before generating.
- **Generate image** sends the prompt to Gemini; the app returns an image URL and shows it in the UI.
- **Edit image** sends the current image + change description to Gemini for iterative edits.

---

## Main components

| Component | Role |
|-----------|------|
| **Flask** | Backend API; serves the UI and handles `/generate-prompt`, `/generate-image`, `/edit-image`, `/images/<filename>`. |
| **OpenAI (GPT-4)** | LLM used to turn your question + optional RAG context into a detailed image prompt. |
| **MongoDB Atlas Vector Search** | Optional RAG: stores medical documents; retrieves relevant snippets to improve prompts. |
| **OpenAI Embeddings** | Used for RAG (embedding queries and documents for vector search). |
| **Google Gemini** | Generates images from text prompts and edits existing images from natural-language change requests. |
| **Single-page UI** (`index.html`) | System instruction, question/prompt boxes, image preview, chat history, and multi-session support. |

---

## Environment

- **Required**: `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`
- **Optional**: `MONGODB_URI` for RAG (falls back to prompt-only generation if unset or if the vector store is unavailable)
- **Optional**: `PORT` to change the server port (default `5001`)

Images are served from memory (and optionally from `static/images/` when writable) via `GET /images/<filename>`.
