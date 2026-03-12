"""
AI Medical Image Generator Server — main entry point.

Initializes configuration, logging, Flask app, CORS, database and clients,
then registers route modules. Run with: python server.py
"""
import logging
import os
import sys

from flask import Flask
from flask_cors import CORS

import config
from app_state import state
from db import init_mongo
from clients import init_llm, init_gemini, init_serper
from routes import main_routes, rag_routes, image_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Starting AI Medical Image Generator Server")
logger.info("=" * 60)

app = Flask(__name__)
CORS(app)
logger.info("Flask app and CORS initialized")

# Log API key status
if config.OPENAI_API_KEY:
    logger.info("OpenAI API key found (length: %d)", len(config.OPENAI_API_KEY))
else:
    logger.error("OpenAI API key not found!")

if config.GOOGLE_API_KEY:
    logger.info("Google API key found (length: %d)", len(config.GOOGLE_API_KEY))
else:
    logger.error("Google API key not found!")

# Initialize LLM and Gemini
logger.info("Initializing LLM...")
state.llm = init_llm()
state.openai_api_key = config.OPENAI_API_KEY
state.google_api_key = config.GOOGLE_API_KEY

logger.info("Initializing Gemini client...")
state.gemini_client = init_gemini()

logger.info("Initializing Serper (web search)...")
state.google_serper_wrapper = init_serper()

# Initialize MongoDB, vectorstore, retriever
logger.info("Initializing MongoDB connection...")
(
    state.mongo_client,
    state.vectorstore,
    state.retriever,
    state.known_doc_names,
    state.embedding_model,
) = init_mongo()

# Register route modules
main_routes.register(app)
rag_routes.register(app)
image_routes.register(app)
logger.info("Routes registered")

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("AI Prompt to Image Generator Server")
    logger.info("=" * 60)
    logger.info(
        "IMPORTANT: Make sure to create a .env file with your API keys, for example:"
    )
    logger.info("  OPENAI_API_KEY=your-openai-api-key")
    logger.info("  GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key")
    logger.info(
        "The server will automatically load environment variables from .env if present."
    )

    port = int(os.environ.get('PORT', 5001))
    logger.info("Server starting on http://localhost:%s", port)
    logger.info("Open your browser and navigate to http://localhost:%s", port)
    logger.info("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=port)
