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
SVG_STORE = {}  # filename -> svg string (optional cache for downloads)
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

# --- PNG → SVG vectorization (vtracer + preprocessing, fidelity-first) ---
# Serverless defaults are lower to limit memory/time on Vercel/Lambda.
_trace_max_default = '1400' if IS_SERVERLESS else '2048'
_trace_target_default = '0' if IS_SERVERLESS else '2048'  # 0 = disable upscale target

TRACE_BACKEND = os.getenv('TRACE_BACKEND', 'vtracer').strip().lower()
# vtracer | vectorizer_ai | recraft

TRACE_MAX_DIMENSION = int(os.getenv('TRACE_MAX_DIMENSION', _trace_max_default))
TRACE_TARGET_DIMENSION = int(os.getenv('TRACE_TARGET_DIMENSION', _trace_target_default))
TRACE_UPSCALE_ENABLED = os.getenv('TRACE_UPSCALE_ENABLED', 'true').lower() in (
    '1', 'true', 'yes',
)

# Super-resolution (OpenCV dnn_superres); falls back to LANCZOS if model missing
TRACE_SUPERRES_ENABLED = os.getenv('TRACE_SUPERRES_ENABLED', 'true').lower() in (
    '1', 'true', 'yes',
)
_superres_model_default = 'FSRCNN_x2' if IS_SERVERLESS else 'FSRCNN_x3'
TRACE_SUPERRES_MODEL = os.getenv('TRACE_SUPERRES_MODEL', _superres_model_default)
# FSRCNN_x2 | FSRCNN_x3 | EDSR_x4
TRACE_SUPERRES_MODELS_DIR = Path(
    os.getenv('TRACE_SUPERRES_MODELS_DIR', str(Path(__file__).resolve().parent / 'models' / 'superres'))
)
TRACE_SUPERRES_AUTO_DOWNLOAD = os.getenv('TRACE_SUPERRES_AUTO_DOWNLOAD', 'false').lower() in (
    '1', 'true', 'yes',
)

TRACE_FLATTEN_ALPHA = os.getenv('TRACE_FLATTEN_ALPHA', 'true').lower() in ('1', 'true', 'yes')
TRACE_ALPHA_THRESHOLD = int(os.getenv('TRACE_ALPHA_THRESHOLD', '128'))
# Legacy median denoise (off by default for clean AI art)
TRACE_DENOISE = os.getenv('TRACE_DENOISE', 'false').lower() in ('1', 'true', 'yes')
# Edge-preserving smoothing: bilateral | edge_preserving | none
TRACE_SMOOTH_METHOD = os.getenv('TRACE_SMOOTH_METHOD', 'bilateral').strip().lower()
TRACE_BILATERAL_D = int(os.getenv('TRACE_BILATERAL_D', '9'))
TRACE_BILATERAL_SIGMA_COLOR = int(os.getenv('TRACE_BILATERAL_SIGMA_COLOR', '75'))
TRACE_BILATERAL_SIGMA_SPACE = int(os.getenv('TRACE_BILATERAL_SIGMA_SPACE', '75'))
TRACE_EDGE_PRESERVE_SIGMA_S = float(os.getenv('TRACE_EDGE_PRESERVE_SIGMA_S', '60'))
TRACE_EDGE_PRESERVE_SIGMA_R = float(os.getenv('TRACE_EDGE_PRESERVE_SIGMA_R', '0.4'))

TRACE_SHARPEN = os.getenv('TRACE_SHARPEN', 'false').lower() in ('1', 'true', 'yes')
TRACE_SHARPEN_RADIUS = float(os.getenv('TRACE_SHARPEN_RADIUS', '0.8'))
TRACE_SHARPEN_PERCENT = int(os.getenv('TRACE_SHARPEN_PERCENT', '40'))
TRACE_SHARPEN_THRESHOLD = int(os.getenv('TRACE_SHARPEN_THRESHOLD', '2'))

# K-means color quantization before tracing (reduces gradient banding)
TRACE_QUANTIZE_ENABLED = os.getenv('TRACE_QUANTIZE_ENABLED', 'true').lower() in (
    '1', 'true', 'yes',
)
TRACE_QUANTIZE_COLORS = int(os.getenv('TRACE_QUANTIZE_COLORS', '20'))
TRACE_QUANTIZE_SAMPLE_MAX = int(os.getenv('TRACE_QUANTIZE_SAMPLE_MAX', '12000'))

TRACE_SVG_SCOUR = os.getenv('TRACE_SVG_SCOUR', 'true').lower() in ('1', 'true', 'yes')
TRACE_SVG_SEAM_FILL = os.getenv('TRACE_SVG_SEAM_FILL', 'true').lower() in ('1', 'true', 'yes')
TRACE_SVG_SEAM_STROKE_WIDTH = float(os.getenv('TRACE_SVG_SEAM_STROKE_WIDTH', '0.35'))
TRACE_SVG_MIN_PATH_AREA = float(os.getenv('TRACE_SVG_MIN_PATH_AREA', '4.0'))

# Retuned defaults for cleaner stacked output
VTRACER_COLOR_PRECISION = int(os.getenv('VTRACER_COLOR_PRECISION', '7'))
VTRACER_LAYER_DIFFERENCE = int(os.getenv('VTRACER_LAYER_DIFFERENCE', '20'))
VTRACER_FILTER_SPECKLE = int(os.getenv('VTRACER_FILTER_SPECKLE', '6'))
VTRACER_PATH_PRECISION = int(os.getenv('VTRACER_PATH_PRECISION', '8'))
VTRACER_LENGTH_THRESHOLD = float(os.getenv('VTRACER_LENGTH_THRESHOLD', '4.5'))
VTRACER_CORNER_THRESHOLD = int(os.getenv('VTRACER_CORNER_THRESHOLD', '60'))
VTRACER_SPLICE_THRESHOLD = int(os.getenv('VTRACER_SPLICE_THRESHOLD', '45'))
VTRACER_MAX_ITERATIONS = int(os.getenv('VTRACER_MAX_ITERATIONS', '10'))

# Paid vectorization APIs (optional; set TRACE_BACKEND + API key)
VECTORIZER_AI_API_KEY = os.getenv('VECTORIZER_AI_API_KEY', '').strip()
VECTORIZER_AI_API_URL = os.getenv(
    'VECTORIZER_AI_API_URL',
    'https://vectorizer.ai/api/v1/vectorize',
).strip()
RECRAFT_API_KEY = os.getenv('RECRAFT_API_KEY', '').strip()
RECRAFT_VECTORIZE_URL = os.getenv(
    'RECRAFT_VECTORIZE_URL',
    'https://external.api.recraft.ai/v1/vectorization',
).strip()

VECTORIZE_DEBUG_DIR = Path(os.getenv('VECTORIZE_DEBUG_DIR', str(IMAGES_DIR)))
