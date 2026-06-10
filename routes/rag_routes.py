"""
RAG routes: chat with docs, doc names, session uploads.
"""
import time
import traceback
import logging

from flask import request, jsonify

from app_state import state
from db import fetch_distinct_doc_names
from services import rag_service
from services import ondemand_docs_service
from prompts import CHAT_WITH_DOCS_SYSTEM, CHAT_WITH_DOCS_USER_TEMPLATE

from routes.constants import API_PREFIX

logger = logging.getLogger(__name__)


def register(app):
    @app.route(f'{API_PREFIX}/chat-with-docs', methods=['POST'])
    def chat_with_docs():
        """Answer direct user questions using retrieved document chunks."""
        request_start = time.time()
        logger.info("[/chat-with-docs] Request received")
        try:
            data = request.get_json() or {}
            session_id = rag_service.extract_session_id(data)
            user_question = str((data or {}).get('user_question') or '').strip()
            raw_selected_doc_names = (data or {}).get('selected_doc_names')
            chat_history_text = str((data or {}).get('chat_history') or '').strip()

            if not user_question:
                return jsonify({'error': 'user_question is required'}), 400
            if rag_service.is_no_rag_selected(raw_selected_doc_names):
                return jsonify({'error': 'NO RAG is selected; enable documents to chat.'}), 400
            if state.llm is None:
                return jsonify({'error': 'LLM is not initialized'}), 503

            selected_doc_names = rag_service.sanitize_selected_doc_names(
                raw_selected_doc_names,
                session_id=session_id,
            )
            selected_sources = selected_doc_names

            lowered_question = user_question.lower()
            image_intent = (
                ("image" in lowered_question)
                and any(
                    token in lowered_question
                    for token in ["generate", "create", "make", "draw"]
                )
            )
            if image_intent:
                return jsonify({
                    'answer': (
                        'This page is for document Q&A only. To create medical illustrations, '
                        'open the Studio page and use image generation there.'
                    ),
                    'search_query': user_question,
                    'chunks': [],
                    'selected_doc_names': selected_sources,
                    'session_id': session_id,
                }), 200

            if not selected_doc_names:
                return jsonify({
                    'answer': 'Please select one or more source documents to chat with docs.',
                    'search_query': user_question,
                    'chunks': [],
                    'selected_doc_names': selected_sources,
                    'session_id': session_id,
                }), 200

            vector_docs = rag_service.retrieve_docs_with_timeout(
                user_question,
                timeout_sec=20,
                selected_doc_names=selected_doc_names,
                total_k=10,
                session_id=session_id,
                equal_per_selected_doc=True,
            )

            if not vector_docs:
                return jsonify({
                    'answer': 'I could not find relevant information in the selected documents for this question.',
                    'search_query': user_question,
                    'chunks': [],
                    'selected_doc_names': selected_sources,
                    'session_id': session_id,
                }), 200

            context = rag_service.build_combined_context(vector_docs, [])
            qa_prompt = CHAT_WITH_DOCS_USER_TEMPLATE.format(
                chat_history=chat_history_text,
                user_question=user_question,
                context=context,
            )
            response = state.llm.invoke([
                {"role": "system", "content": CHAT_WITH_DOCS_SYSTEM},
                {"role": "user", "content": qa_prompt},
            ])

            chunks_payload = []
            for doc in vector_docs:
                metadata = getattr(doc, "metadata", {}) or {}
                metadata["source_type"] = metadata.get("source_type") or "vector"
                chunks_payload.append({
                    "content": doc.page_content,
                    "metadata": metadata,
                })

            elapsed = time.time() - request_start
            logger.info(
                "[/chat-with-docs] Returned answer with %d chunks in %.2fs",
                len(chunks_payload),
                elapsed,
            )
            return jsonify({
                'answer': (response.content or '').strip(),
                'search_query': user_question,
                'chunks': chunks_payload,
                'selected_doc_names': selected_sources,
                'session_id': session_id,
            }), 200
        except TimeoutError as e:
            return jsonify({'error': str(e)}), 504
        except Exception as e:
            logger.error("[/chat-with-docs] Error: %s", e)
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.route(f'{API_PREFIX}/doc-names', methods=['GET'])
    def get_doc_names():
        """Return distinct source document names available in the vector store."""
        session_id = (request.args.get('session_id') or '').strip()
        latest_names = fetch_distinct_doc_names(state.mongo_client)
        if latest_names:
            state.known_doc_names = latest_names
        base_names = sorted([
            name
            for name in state.known_doc_names
            if isinstance(name, str) and name.strip()
        ])
        session_names = ondemand_docs_service.list_session_doc_names(session_id)
        return jsonify({
            'doc_names': base_names,
            'base_doc_names': base_names,
            'session_doc_names': session_names,
            'count': len(base_names) + len(session_names),
            'session_id': session_id,
        }), 200

    @app.route(f'{API_PREFIX}/upload-doc', methods=['POST'])
    def upload_doc():
        """Upload a PDF for the current browser session and ingest chunks into MongoDB."""
        session_id = (request.form.get('session_id') or '').strip()
        upload = request.files.get('file')
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        if upload is None:
            return jsonify({'error': 'file is required'}), 400

        try:
            result = ondemand_docs_service.upload_pdf_for_session(session_id, upload)
            return jsonify({
                'success': True,
                'session_id': session_id,
                'doc_name': result['doc_name'],
                'chunks_inserted': result['chunks_inserted'],
                'session_doc_names': result['session_doc_names'],
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route(f'{API_PREFIX}/session/reset', methods=['POST'])
    def reset_session_docs():
        """Drop session-scoped on-demand document collection and forget in-memory session state."""
        data = request.get_json(silent=True) or {}
        session_id = str((data or {}).get('session_id') or '').strip()
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        cleared = ondemand_docs_service.clear_session(session_id)
        return jsonify({'success': True, 'session_id': session_id, 'cleared': bool(cleared)}), 200
