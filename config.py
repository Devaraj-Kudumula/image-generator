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

# OpenAI chat models (override via environment for newer API releases)
OPENAI_RAG_MODEL = os.getenv('OPENAI_RAG_MODEL', 'gpt-4o')
OPENAI_RAG_TEMPERATURE = float(os.getenv('OPENAI_RAG_TEMPERATURE', '0.2'))
OPENAI_RAG_MAX_OUTPUT = int(os.getenv('OPENAI_RAG_MAX_OUTPUT', '2048'))

# Standalone AI chat (ai_chat page): strongest text model by default; trim if env unset for older accounts.
OPENAI_CONVERSATION_MODEL = os.getenv('OPENAI_CONVERSATION_MODEL', 'gpt-5.5')
OPENAI_CONVERSATION_TEMPERATURE = float(os.getenv('OPENAI_CONVERSATION_TEMPERATURE', '0.7'))
OPENAI_CONVERSATION_MAX_OUTPUT = int(os.getenv('OPENAI_CONVERSATION_MAX_OUTPUT', '8192'))
# Approximate input token budget for history + system (completion budget is separate via max_tokens).
OPENAI_CONVERSATION_MAX_CONTEXT_TOKENS = int(
    os.getenv('OPENAI_CONVERSATION_MAX_CONTEXT_TOKENS', '200000')
)
OPENAI_CONVERSATION_REQUEST_TIMEOUT = int(os.getenv('OPENAI_CONVERSATION_REQUEST_TIMEOUT', '180'))

# Refined-prompt regeneration (vision QA → text refinement → new Gemini image)
OPENAI_REFINED_REGEN_VISION_MODEL = os.getenv(
    'OPENAI_REFINED_REGEN_VISION_MODEL', 'gpt-5.4'
)
OPENAI_REFINED_REGEN_TEXT_MODEL = os.getenv(
    'OPENAI_REFINED_REGEN_TEXT_MODEL', 'gpt-5.5'
)

# MongoDB
MONGODB_URI = os.getenv(
    'MONGODB_URI',
    'mongodb+srv://devarajkudumulanew:12345@cluster0.9iljp.mongodb.net/',
).strip().strip('"').strip("'")

DB_NAME = "medical_vector_db"
COLLECTION_NAME = "new_vector_chunks"
INDEX_NAME = "default"

# Session-scoped on-demand document store
ONDEMAND_MONGODB_URI = os.getenv(
    'ONDEMAND_MONGODB_URI',
    'mongodb+srv://Devaa:Devaa@dev.pa9pov2.mongodb.net/?appName=DEV',
).strip().strip('"').strip("'")
ONDEMAND_DB_NAME = os.getenv('ONDEMAND_DB_NAME', 'medical_vector_db')
ONDEMAND_INDEX_NAME = os.getenv('ONDEMAND_INDEX_NAME', 'default')

# RAG options
NO_RAG_OPTION_VALUE = "NO_RAG"
WEB_RETRIEVAL_OPTION_VALUE = "WEB_RETRIEVAL"
WEB_RETRIEVAL_RESULT_COUNT = 10

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
