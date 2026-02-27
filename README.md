## AI Medical Image Generator

This app lets you turn clinical ideas into exam‑ready medical illustrations. It combines:

- **LLM prompt engineering with RAG** (OpenAI + MongoDB Atlas Vector Search) to craft high‑quality image prompts.
- **Image generation and editing** (Google Gemini) to create and refine images.
- A **single‑page UI** (`index.html`) to manage system instructions, prompts, images, and chat‑style history.

---

## High‑level workflow

### 1. System instruction (prompt “policy”)

- **What it is**: A long, opinionated system prompt that defines how image prompts should be written (USMLE level, clinical realism, mechanism focus, etc.).
- **Where**: The **“System Instruction (How prompts are crafted)”** textarea at the top‑left of the UI (`systemInstruction` in `index.html`).
- **How it’s used**:
  - Sent to the backend as `system_instruction` when you click **“Generate image prompt”**.
  - Passed to the OpenAI Chat model as the `system` message to shape the final prompt style and content for all subsequent generations in that session (until you change it).

### 2. User question / medical topic

- **What it is**: The specific case, topic, or vignette you want an illustration for.
- **Where**: The **“Your Question / Medical Topic”** textarea (`userQuestion`).
- **How it’s used**:
  - Sent as `user_question` to `/generate-prompt`.
  - The backend may:
    - Use a smaller LLM call to **extract high‑yield retrieval keywords**.
    - Use MongoDB Atlas Vector Search (if configured) to **retrieve context documents** from `medical_rag.medical_documents`.
    - Combine those documents and your question into a **construction prompt** for the main LLM call.

### 3. Prompt generation (OpenAI + optional RAG)

**Endpoint**: `POST /generate-prompt` (see `server.py`).

1. **Input payload**:
   - `system_instruction`: text from the system instruction box.
   - `user_question`: text from the question/topic box.
2. **RAG branch (if MongoDB + embeddings available)**:
   - Uses `ChatOpenAI` to extract retrieval cues from `user_question`.
   - Calls the `retriever` (backed by `MongoDBAtlasVectorSearch`) to get top‑k documents.
   - Builds a combined **context + user request** construction prompt.
   - Calls the main `ChatOpenAI` model (`gpt-4`) with:
     - `system`: your system instruction.
     - `user`: the constructed context + request.
3. **Fallback branch (no RAG / failure)**:
   - Directly calls `ChatOpenAI` with:
     - `system`: your system instruction.
     - `user`: `Create a detailed medical illustration prompt for: <user_question>`.
4. **Output**:
   - JSON: `{ "prompt": "<generated prompt>", "success": true }`.
   - The UI writes this into the **“Image Generation Prompt (Editable)”** textarea (`generatedPrompt`), where you can freely edit before generating images.

### 4. Image generation (Gemini)

**Endpoint**: `POST /generate-image`.

1. **Input**:
   - `prompt`: the contents of the **“Image Generation Prompt (Editable)”** box (either auto‑generated, hand‑written, or edited).
2. **Backend flow**:
   - Uses `google.genai` client with `model="gemini-3-pro-image-preview"`.
   - Calls `gemini_client.models.generate_content(contents=[prompt])`.
   - Extracts `inline_data` from the first candidate’s parts and decodes it into an image.
   - Stores the image (in-memory on serverless, or also under `static/images/image_<timestamp>.png` when writable).
3. **Output**:
   - JSON: `{ "image_url": "<full URL>", "filename": "<filename>", "success": true }`.
   - The UI:
     - Updates the **“Latest image preview”** card (large pinned preview).
     - Adds the image to the **conversation history** (right column) as an assistant message.
     - Increments the **per‑chat image count** and enables **“Download latest image”**.

### 5. Image editing (Gemini with existing image)

**Endpoint**: `POST /edit-image`.

1. **Input**:
   - `filename`: the last generated/edited image filename (the UI derives this from the current image URL).
   - `changes`: natural‑language description of what you want to change (e.g., zoom, labels, lighting, focus, mechanism tweaks).
2. **Backend flow**:
   - Loads the existing image from the server (in-memory store or `static/images/<filename>` when available).
   - Builds a prompt: `"Edit the following image based on the requested changes:\n\nChanges: <changes>"`.
   - Calls `gemini_client.models.generate_content(contents=[prompt, image])` to condition on both the instructions and the original image.
   - Extracts a new image from the response and stores it (in-memory and optionally as `static/images/edited_<timestamp>.png`).
3. **Output**:
   - JSON: `{ "image_url": "<new URL>", "filename": "<new filename>", "success": true }`.
   - The UI:
     - Updates the pinned preview to the new image.
     - Records an **assistant “edited image”** entry in the conversation history.
     - Keeps the original image and all edits visible in the chat timeline so you can compare iterations.

### 6. Chat sessions and history

- **Multiple chats**:
  - The right column supports **multiple independent chat sessions** (Chat 1, Chat 2, …).
  - Each chat has its own:
    - Message history (user prompts, edit requests, and images).
    - Current pinned image URL.
    - Image count.
- **Controls**:
  - **Chat selector**: drop‑down to switch between chats.
  - **“New chat”**: starts a fresh conversation with empty history and blank prompt/edit boxes (system instruction text is not automatically reset).

---

## Backend endpoints overview

All endpoints are implemented in `server.py`:

- **`GET /`**: Serves `index.html`.
- **`GET /health`**: Returns health/status JSON (API keys, MongoDB connectivity, retriever state, etc.).
- **`POST /generate-prompt`**: Generates image prompts via OpenAI with optional RAG.
- **`POST /generate-image`**: Generates an image from a prompt via Gemini; stores in memory (and optionally under `static/images` when the filesystem is writable).
- **`POST /edit-image`**: Edits an existing image via Gemini and stores the new image (in-memory and optionally on disk).
- **`GET /images/<filename>`**: Serves generated and edited images from the in-memory store or, locally, from `static/images`.

---

## Environment configuration (`.env`)

Create a `.env` file in the project root (`image-generator/`) to hold API keys and (optionally) database configuration.

### Required keys

- **`OPENAI_API_KEY`**: Used by LangChain’s `ChatOpenAI` and `OpenAIEmbeddings`.
- **`GOOGLE_GENERATIVE_AI_API_KEY`**: Used by the Gemini image generation/editing client.

Example `.env` (do **not** commit real keys):

```bash
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key-here

# Optional: override MongoDB connection string for RAG
# MONGODB_URI=your-mongodb-connection-uri-here
```

Notes:

- `server.py` automatically loads `.env` with `python-dotenv` if the file exists.
- If `.env` is missing, the app falls back to **system environment variables** with the same names.
- Ensure `.env` is **ignored by git** (add `.env` to `.gitignore` if not already present).

---

## MongoDB RAG configuration (optional but recommended)

If you want RAG‑enhanced prompts:

- **`MONGODB_URI`**:
  - Connection string to MongoDB Atlas (or compatible).
  - Used to initialize `MongoClient` and `MongoDBAtlasVectorSearch`.
  - If not set, a default URI is used (see `MONGODB_URI` in `server.py`), but you should override this in production.
- **Database & collection**:
  - Database: `medical_rag`
  - Collection: `medical_documents`
  - Vector index name: `vector_index`
- **Vector store**:
  - Uses `OpenAIEmbeddings` (`text-embedding-3-small`) plus `MongoDBAtlasVectorSearch`.
  - The retriever runs similarity search with `k=3` for relevant context snippets.
- If the collection is empty or the vector store fails to initialize:
  - The server logs a warning.
  - `/generate-prompt` falls back gracefully to **direct prompt generation without RAG**.

You’ll need to populate MongoDB separately (e.g., via a `build_mongo_vectorstore.py` script or similar ingestion pipeline).

---

## Local development setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd image-generator

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Create `.env`

Create `image-generator/.env` with:

```bash
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key-here
# Optional:
# MONGODB_URI=your-mongodb-connection-uri-here
```

Make sure `.env` is **not committed** to version control.

### 3. Run the Flask server

```bash
python server.py
```

- The server starts on `http://localhost:5001` by default.
- You can override the port with the `PORT` environment variable (useful for platforms like Render/Railway).

### 4. Use the UI

1. Open `http://localhost:5001` in your browser.
2. Optionally adjust the **System Instruction** at the top.
3. Enter a **Question / Medical Topic** and click **“Generate image prompt”** to produce a refined prompt.
4. Review and edit the **Image Generation Prompt (Editable)** box.
5. Click **“Generate image”** to create an image and see it in both:
   - The **Latest image preview** card.
   - The **Conversation history** as an assistant message.
6. Use **“Suggest Changes to Generated Image”**:
   - Describe edits.
   - Click **“Apply changes to latest image”** to create iterated versions.
7. Manage multiple explorations via the **chat session selector** and **“New chat”**.

---

## Deployment notes

- **Environment variables**:
  - Set `OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, and (optionally) `MONGODB_URI` in your hosting platform’s config.
  - Do not deploy `.env` files; rely on platform‑level environment variable configuration.
- **Server binding**:
  - `server.py` binds to `0.0.0.0` and reads the `PORT` env var, which is compatible with most PaaS providers.
- **Image storage (serverless-friendly)**:
  - On Vercel (or when the filesystem is read-only), images are stored **in memory** and served via `GET /images/<filename>`. No disk writes.
  - Locally, images are also written to `static/images/` when writable. The same `/images/<filename>` endpoint serves from memory first, then from disk.
  - Image URLs returned by the API use the `/images/<filename>` path so they work in both environments.

This README should give you the full picture from **system prompt design → RAG prompt generation → Gemini image creation → iterative image editing**, plus how to configure and run the app safely with API keys in `.env`.

