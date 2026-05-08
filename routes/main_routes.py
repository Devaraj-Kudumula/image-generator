"""
Main routes: index and health check.
"""
import logging

from flask import send_from_directory, jsonify

import config
from app_state import state

logger = logging.getLogger(__name__)


def register(app):
    @app.route('/')
    def index():
        logger.info("Serving index.html")
        return send_from_directory('.', 'index.html')

    @app.route('/docs-chat')
    def docs_chat():
        logger.info("Serving docs_chat.html")
        return send_from_directory('.', 'docs_chat.html')

    @app.route('/upload-edit')
    def upload_edit():
        logger.info("Serving upload_edit.html")
        return send_from_directory('.', 'upload_edit.html')

    @app.route('/ai-chat')
    def ai_chat():
        logger.info("Serving ai_chat.html")
        return send_from_directory('.', 'ai_chat.html')

    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint for monitoring"""
        doc_count = 0
        if state.mongo_client:
            try:
                collection = state.mongo_client[config.DB_NAME][config.COLLECTION_NAME]
                doc_count = collection.count_documents({})
            except Exception:
                doc_count = -1

        status = {
            'status': 'healthy',
            'openai_configured': config.OPENAI_API_KEY is not None,
            'google_configured': config.GOOGLE_API_KEY is not None,
            'mongodb_connected': state.mongo_client is not None,
            'vectorstore_loaded': state.vectorstore is not None,
            'retriever_ready': state.retriever is not None,
            'doc_name_catalog_ready': len(state.known_doc_names) > 0,
            'vectorstore_doc_count': doc_count,
            'gemini_client_ready': state.gemini_client is not None,
            'rag_available': state.retriever is not None and doc_count > 0,
        }
        logger.info("Health check: %s", status)
        return jsonify(status), 200
