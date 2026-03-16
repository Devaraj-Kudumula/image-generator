"""
Application configuration: environment variables, constants, and image storage.
"""
import os
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
_env_path = Path('.env')
if _env_path.exists():
    logger.info("Loading environment variables from %s", _env_path.resolve())
    load_dotenv(_env_path)
else:
    logger.info(".env file not found; relying on system environment variables")

# API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')

# MongoDB
MONGODB_URI = os.getenv(
    'MONGODB_URI',
    'mongodb+srv://db_user:db_user@cluster0.9a1tk8o.mongodb.net/?retryWrites=true&w=majority',
)
DB_NAME = "medical_vector_db"
# COLLECTION_NAME = "vector_chunks_embeds"
COLLECTION_NAME = "new_vector_chunks"
INDEX_NAME = "default"
# INDEX_NAME = "vector_index"

# RAG options
NO_RAG_OPTION_VALUE = "NO_RAG"
WEB_RETRIEVAL_OPTION_VALUE = "WEB_RETRIEVAL"
WEB_RETRIEVAL_RESULT_COUNT = 10

# LLM metrics/cost defaults (USD per 1M tokens). Override with env vars if needed.
OPENAI_PRICING_PER_1M_TOKENS = {
    "gpt-4": {
        "input": float(os.getenv("OPENAI_GPT4_INPUT_PER_1M", "30.0")),
        "output": float(os.getenv("OPENAI_GPT4_OUTPUT_PER_1M", "60.0")),
    },
    "gpt-5": {
        "input": float(os.getenv("OPENAI_GPT5_INPUT_PER_1M", "5.0")),
        "output": float(os.getenv("OPENAI_GPT5_OUTPUT_PER_1M", "15.0")),
    },
    "default": {
        "input": float(os.getenv("OPENAI_DEFAULT_INPUT_PER_1M", "0.0")),
        "output": float(os.getenv("OPENAI_DEFAULT_OUTPUT_PER_1M", "0.0")),
    },
}

GEMINI_PRICING_PER_1M_TOKENS = {
    "gemini-3": {
        "input": float(os.getenv("GEMINI3_INPUT_PER_1M", "3.5")),
        "output": float(os.getenv("GEMINI3_OUTPUT_PER_1M", "10.5")),
    },
    "gemini-2": {
        "input": float(os.getenv("GEMINI2_INPUT_PER_1M", "1.25")),
        "output": float(os.getenv("GEMINI2_OUTPUT_PER_1M", "5.0")),
    },
    "default": {
        "input": float(os.getenv("GEMINI_DEFAULT_INPUT_PER_1M", "0.0")),
        "output": float(os.getenv("GEMINI_DEFAULT_OUTPUT_PER_1M", "0.0")),
    },
}

LLM_METRICS_RECENT_CALL_LIMIT = int(os.getenv("LLM_METRICS_RECENT_CALL_LIMIT", "100"))

# Serper (web search)
SERPER_API_KEY = (
    os.getenv("SERPER_API_KEY")
    or "8a46c8ecdb405e3ed59ef2655fd7ec228f46792e"
).strip()

# Image storage: in-memory for serverless (Vercel has read-only filesystem)
IMAGE_STORE = {}  # filename -> bytes
IMAGES_DIR = Path('static') / 'images'
IS_SERVERLESS = bool(
    os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
)
if not IS_SERVERLESS:
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Images directory ready: %s", IMAGES_DIR.resolve())
    except OSError as e:
        logger.warning("Images directory not writable (%s); using in-memory store only", e)
        IS_SERVERLESS = True
else:
    logger.info("Serverless environment detected; using in-memory image store only")
