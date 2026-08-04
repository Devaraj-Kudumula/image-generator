## AI Medical Image Generator (Figure Studio)

Production-grade SaaS frontend for generating publication-ready medical and scientific illustrations. Built with **Next.js 15**, **React 19**, **TypeScript**, and **shadcn/ui**. Python **Flask** powers image generation, editing, vectorization, and RAG document Q&A.

---

## Quick start (local development)

### 1. Install dependencies

```bash
# Frontend
npm install

# Backend
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
# Optional local-only features (OpenCV, Tesseract, matplotlib):
pip install -r requirements-local.txt
```

### 2. Configure environment

Create `.env` in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key
GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key
# Optional RAG:
MONGODB_URI=your-mongodb-uri
SERPER_API_KEY=your-serper-api-key
```

### 3. Run both servers

```bash
# Terminal 1 — Flask API (port 5002)
python server.py

# Terminal 2 — Next.js frontend (port 3000)
npm run dev
```

Open **http://localhost:3000**

Next.js proxies `/api/*` requests to Flask at `http://127.0.0.1:5002` via `next.config.ts` rewrites.

---

## Architecture

```
Browser → Next.js App Router (pages + /api/chat streaming)
       → /api/generate-image, /api/edit-image, … → Flask (Python)
       → Gemini (images) + OpenAI (chat, vision QA, RAG)
```

### Frontend routes

| Route | Description |
|-------|-------------|
| `/` | Studio — prompt-based generation (FigureLabs-style UI) |
| `/edit` | Upload & edit workspace |
| `/gallery` | Image history |
| `/chat` | AI Chat with streaming (Vercel AI SDK) |
| `/docs` | Document Q&A (RAG) |
| `/settings` | Theme, defaults, system status |

### Backend API (Flask, prefixed with `/api`)

- `POST /api/generate-image` — Gemini text→image
- `POST /api/edit-image` — Gemini image editing
- `POST /api/get-accurate` — Vision QA + iterative fixes
- `POST /api/refined-prompt-image` — Vision QA + full prompt regen
- `POST /api/vectorize-image` — PNG→SVG for canvas editor
- `GET/POST /api/ai-chat-*` — Themed medical chat (Flask fallback)
- `POST /api/chat-with-docs` — RAG document Q&A (when MongoDB configured)

---

## Deploy on Vercel

1. Connect the repo to Vercel
2. Set environment variables (`OPENAI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, optional `MONGODB_URI`)
3. Deploy — Next.js builds the frontend; Python runs at `api/index.py`

`vercel.json` routes Flask API paths to the Python serverless function.

---

## Tech stack

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, Zustand, TanStack Query, Vercel AI SDK, Fabric.js
- **Backend:** Python 3.12, Flask, Google Gemini, OpenAI, MongoDB Atlas (RAG)
- **Deploy:** Vercel (Next.js + Python serverless)

---

## Future roadmap

See `lib/future/platform.ts` for planned integrations: Clerk auth, Vercel Blob storage, Stripe subscriptions, team workspaces, real-time job tracking.
