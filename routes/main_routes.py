"""
Main routes: health check and lightweight stubs for disabled RAG endpoints.
"""
import logging

from flask import jsonify, request

import config
from app_state import state
from routes.constants import API_PREFIX

logger = logging.getLogger(__name__)


def register(app):
    @app.route(f'{API_PREFIX}/health', methods=['GET'])
    def health():
        """Health check endpoint for monitoring"""
        status = {
            'status': 'healthy',
            'openai_configured': config.OPENAI_API_KEY is not None,
            'google_configured': config.GOOGLE_API_KEY is not None,
            'conversation_llm_ready': state.conversation_llm is not None,
            'gemini_client_ready': state.gemini_client is not None,
            'rag_available': state.llm is not None,
            'is_serverless': config.IS_SERVERLESS,
        }
        logger.info("Health check: %s", status)
        return jsonify(status), 200


def register_rag_stubs(app):
    """Register no-op RAG endpoints when the full RAG stack is unavailable."""

    @app.route(f'{API_PREFIX}/doc-names', methods=['GET'])
    def get_doc_names_stub():
        session_id = (request.args.get('session_id') or '').strip()
        return jsonify({
            'doc_names': [],
            'base_doc_names': [],
            'session_doc_names': [],
            'count': 0,
            'session_id': session_id,
            'disabled': True,
        }), 200

    @app.route(f'{API_PREFIX}/session/reset', methods=['POST'])
    def reset_session_stub():
        return jsonify({'success': True, 'cleared': False, 'disabled': True}), 200

    @app.route(f'{API_PREFIX}/chat-with-docs', methods=['POST'])
    def chat_with_docs_stub():
        return jsonify({
            'error': 'Document chat is not available on this deployment.',
            'disabled': True,
        }), 503

    @app.route(f'{API_PREFIX}/upload-doc', methods=['POST'])
    def upload_doc_stub():
        return jsonify({
            'error': 'Document upload is not available on this deployment.',
            'disabled': True,
        }), 503
