## AI Medical Image Generator

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

- **Frontend (`index.html`, `static/app.js`, `static/styles.css`)**
  - User sets a **system instruction**, enters a **medical question/topic**, and clicks **Generate image prompt**.
  - Frontend calls `/generate-prompt`, optionally using RAG controls (source documents, web retrieval, NO RAG).
  - The generated prompt is editable; clicking **Generate image** calls `/generate-image`.
  - Conversation history tracks prompts and image versions; users can request edits inline, which call `/edit-image`.

- **Backend (`server.py` + routes)**
  - `server.py` loads config, initializes LLM, Gemini, MongoDB vectorstore, Serper, and shared `AppState`, then registers routes.
  - `routes/main_routes.py` serves `index.html` (`/`) and exposes `/health`.
  - `routes/rag_routes.py` exposes:
    - `/generate-prompt` – builds a detailed image prompt using OpenAI + optional RAG (MongoDB vectorstore + web).
    - `/re-run-retrieval` – re-runs retrieval for updated doc selection/query.
    - `/doc-names` – lists available `doc_name` values from MongoDB.
  - `routes/image_routes.py` exposes:
    - `/generate-image` – sends the prompt to Gemini and stores image bytes (in-memory + optional disk).
    - `/images/<filename>` – serves generated images from memory or `static/images/`.
    - `/edit-image` – loads an existing image (from filename or data URL), applies Gemini edits, and stores the new version.

- **Data & RAG**
  - `db.init_mongo()` connects to MongoDB Atlas, configures `MongoDBAtlasVectorSearch`, builds a retriever, and loads known `doc_name`s.
  - `services.rag_service` normalizes source selection, runs vector + web retrieval, and builds the combined context string for the LLM.
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
- **`routes/rag_routes.py`**: `/generate-prompt`, `/re-run-retrieval`, `/doc-names` for RAG-driven prompt generation and retrieval control.
- **`routes/image_routes.py`**: `/generate-image`, `/edit-image`, `/images/<filename>` for image generation, editing, and serving.

- **`services/rag_service.py`**: Implements retrieval logic (doc name validation, NO RAG / WEB_RETRIEVAL flags, Serper + web scraping, vector search, context assembly, structured retrieval query building).
- **`services/image_service.py`**: Implements Gemini-based image generation and editing, plus in-memory/disk storage and retrieval of image bytes.

- **`index.html`**: Single-page UI with system instruction, question, RAG controls, editable prompt, image preview, chat sessions, and full-screen image viewer.
- **`static/app.js`**: Frontend logic for theme toggle, chat sessions, RAG controls, calling `/generate-prompt`, `/re-run-retrieval`, `/generate-image`, `/edit-image`, and updating the UI.
- **`static/styles.css`**: Modern light/dark theme styling, layout, prompt/editor sections, conversation history, RAG panel, and image preview/fullscreen UI.
- **`static/images/`**: Optional directory where generated images are persisted when not running in a serverless environment.

---

## Environment

- **Required**: `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`
- **Optional (RAG)**: `MONGODB_URI` (falls back to direct LLM prompts if unavailable)
- **Optional (web retrieval)**: `SERPER_API_KEY` (used for Google web search via Serper)
- **Optional**: `PORT` (default `5001`)

Images are always stored in-memory via `IMAGE_STORE`, and additionally written to `static/images/` when the filesystem is writable and the app is not running serverless.
