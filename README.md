## AI Medical Image Generator

Turn clinical ideas into exam-ready medical illustrations. The Studio UI sends prompts directly to Google Gemini for generation and editing; Doc chat uses an LLM with RAG for Q&A over your PDFs.

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
   # Optional: SERPER_API_KEY=your-serper-api-key
   ```
   Do not commit `.env`; it should be in `.gitignore`.

4. **Run the app**
   ```bash
   python server.py
   ```

5. **Open in browser**  
   `http://localhost:5001`  
   (Port can be overridden with the `PORT` environment variable.)

---

## Overall application flow

- **Frontend (`index.html`, `docs_chat.html`, `upload_edit.html`, `static/app.js`, `static/styles.css`)**
  - **Studio** (`/`): user writes or pastes an image prompt and calls `/generate-image`; history, `/edit-image`, and `/get-accurate` support iteration.
  - **Doc chat** (`/docs-chat`): user selects documents and asks questions; the frontend calls `/chat-with-docs` (RAG over MongoDB + session uploads).
  - **Upload & edit** (`/upload-edit`): user uploads an image and describes edits via `/edit-image`.

- **Backend (`server.py` + routes)**
  - `server.py` loads config, initializes LLM, Gemini, MongoDB vectorstore, Serper, and shared `AppState`, then registers routes.
  - `routes/main_routes.py` serves HTML pages and exposes `/health`.
  - `routes/rag_routes.py` exposes:
    - `/chat-with-docs` – answers from retrieved chunks using the OpenAI chat model.
    - `/doc-names` – lists library and session document names.
    - `/upload-doc`, `/session/reset` – session-scoped PDF ingest and cleanup.
  - `routes/image_routes.py` exposes:
    - `/generate-image` – sends the prompt to Gemini and stores image bytes (in-memory + optional disk).
    - `/images/<filename>` – serves generated images from memory or `static/images/`.
    - `/edit-image` – loads an existing image (from filename or data URL), applies Gemini edits, and stores the new version.

- **Data & RAG**
  - `db.init_mongo()` connects to MongoDB Atlas, configures `MongoDBAtlasVectorSearch`, builds a retriever, and loads known `doc_name`s.
  - `services.rag_service` normalizes source selection, runs vector + web retrieval, and builds the combined context string for doc chat and retrieval.
  - `services.image_service` wraps Gemini calls and image storage; `backend/image_utils.py` handles data URL and PNG extraction.

---

## Modules and files (brief)

- **`server.py`**: Application entry point; initializes config, logging, `AppState`, LLM, Gemini, Serper, MongoDB/vectorstore, and registers all routes.
- **`config.py`**: Loads environment variables; defines API keys, MongoDB settings, RAG options, Serper key, image store (`IMAGE_STORE`, `IMAGES_DIR`, `IS_SERVERLESS`).
- **`app_state.py`**: Defines `AppState` singleton holding runtime clients (LLM, Gemini, MongoDB, vectorstore, retriever, Serper, known doc names).
- **`clients.py`**: Creates OpenAI `ChatOpenAI` client, Google Gemini client (`genai.Client`), and Google Serper wrapper.
- **`db.py`**: Connects to MongoDB, initializes `MongoDBAtlasVectorSearch` and retriever, fetches distinct `doc_name`s.
- **`backend/image_utils.py`**: Utility helpers for converting image bytes ↔ data URLs and extracting PNG bytes from Gemini responses.

- **`routes/main_routes.py`**: `/` (serves `index.html`) and `/health` (reports config/RAG readiness).
- **`routes/rag_routes.py`**: `/chat-with-docs`, `/doc-names`, `/upload-doc`, `/session/reset`.
- **`routes/image_routes.py`**: `/generate-image`, `/edit-image`, `/images/<filename>` for image generation, editing, and serving.
  It also exposes `/get-accurate`, which runs a multi-step accuracy refinement loop:
  GPT-4o vision first detects labeling/arrow flaws in an image, then Gemini applies up to five targeted correction passes.

- **`services/rag_service.py`**: Implements retrieval logic (doc name validation, NO RAG / WEB_RETRIEVAL flags, Serper + web scraping, vector search, context assembly).
- **`services/image_service.py`**: Implements Gemini-based image generation and editing, the `/get-accurate` iterative flaw-detection-and-fix pipeline, plus in-memory/disk storage and retrieval of image bytes.

- **`index.html`**, **`docs_chat.html`**, **`upload_edit.html`**: Studio, document Q&A, and upload-to-edit flows.
- **`static/app.js`**: Shared frontend logic (theme, chat sessions, doc selection, `/chat-with-docs`, `/generate-image`, `/edit-image`, `/get-accurate`, uploads).
- **`static/styles.css`**: Modern light/dark theme styling, layout, prompt/editor sections, conversation history, RAG panel, and image preview/fullscreen UI.
- **`static/images/`**: Optional directory where generated images are persisted when not running in a serverless environment.

---

## Environment

- **Required**: `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`
- **Optional (RAG)**: `MONGODB_URI` (falls back to direct LLM prompts if unavailable)
- **Optional (web retrieval)**: `SERPER_API_KEY` (used for Google web search via Serper)
- **Optional**: `PORT` (default `5001`)

Images are always stored in-memory via `IMAGE_STORE`, and additionally written to `static/images/` when the filesystem is writable and the app is not running serverless.
