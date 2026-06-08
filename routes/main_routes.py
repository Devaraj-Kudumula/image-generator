"""
Main routes: pages, health check, and lightweight stubs for disabled RAG endpoints.
"""
import logging

from flask import send_from_directory, jsonify, redirect, request

import config
from app_state import state

logger = logging.getLogger(__name__)


def register(app):
    @app.route('/static/<path:filename>')
    def static_assets(filename):
        """Serve JS/CSS when not already handled by Vercel public/ CDN."""
        return send_from_directory('static', filename)

    @app.route('/')
    def index():
        logger.info("Redirecting / to /ai-chat")
        return redirect('/ai-chat', code=302)

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
        status = {
            'status': 'healthy',
            'openai_configured': config.OPENAI_API_KEY is not None,
            'google_configured': config.GOOGLE_API_KEY is not None,
            'conversation_llm_ready': state.conversation_llm is not None,
            'gemini_client_ready': state.gemini_client is not None,
            'rag_available': False,
            'is_serverless': config.IS_SERVERLESS,
        }
        logger.info("Health check: %s", status)
        return jsonify(status), 200

    @app.route('/doc-names', methods=['GET'])
    def get_doc_names_stub():
        """Stub: RAG doc catalog disabled on Vercel slim deploy."""
        session_id = (request.args.get('session_id') or '').strip()
        return jsonify({
            'doc_names': [],
            'base_doc_names': [],
            'session_doc_names': [],
            'count': 0,
            'session_id': session_id,
            'disabled': True,
        }), 200

    @app.route('/session/reset', methods=['POST'])
    def reset_session_stub():
        """Stub: session doc reset is a no-op when RAG is disabled."""
        return jsonify({'success': True, 'cleared': False, 'disabled': True}), 200

    @app.route('/chat-with-docs', methods=['POST'])
    def chat_with_docs_stub():
        """Stub: document Q&A is disabled on Vercel slim deploy."""
        return jsonify({
            'error': 'Document chat is not available on this deployment.',
            'disabled': True,
        }), 503

    @app.route('/upload-doc', methods=['POST'])
    def upload_doc_stub():
        """Stub: PDF upload for RAG is disabled on Vercel slim deploy."""
        return jsonify({
            'error': 'Document upload is not available on this deployment.',
            'disabled': True,
        }), 503
