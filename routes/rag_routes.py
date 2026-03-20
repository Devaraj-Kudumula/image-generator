"""
RAG routes: re-run retrieval, doc names, generate prompt.
"""
import time
import traceback
import logging
from typing import List

from flask import request, jsonify
from langchain_core.documents import Document

import config
from app_state import state
from db import fetch_distinct_doc_names
from services import rag_service
from services.llm_metrics_service import record_langchain_openai_call

logger = logging.getLogger(__name__)


def _invoke_openai_and_track(messages):
    response = state.llm.invoke(messages)
    model_name = getattr(state.llm, "model_name", None) or "gpt-4"
    record_langchain_openai_call(response, model_name)
    return response


def register(app):
    @app.route('/re-run-retrieval', methods=['POST'])
    def re_run_retrieval():
        """
        Re-run retrieval with a user-provided search query. Returns chunks for the frontend
        to display and use in downstream prompt synthesis.
        """
        request_start = time.time()
        logger.info("[/re-run-retrieval] Request received")
        try:
            data = request.get_json()
            search_query = (data or {}).get('search_query', '').strip()
            raw_selected_doc_names = (data or {}).get('selected_doc_names')
            web_retrieval_selected = rag_service.is_web_retrieval_selected(
                raw_selected_doc_names
            )
            disable_rag = bool((data or {}).get('disable_rag')) or rag_service.is_no_rag_selected(
                raw_selected_doc_names
            )
            selected_doc_names = rag_service.sanitize_selected_doc_names(
                raw_selected_doc_names
            )
            selected_sources = rag_service.normalize_selected_sources(
                raw_selected_doc_names
            )
            run_doc_retrieval = rag_service.should_run_doc_retrieval(
                raw_selected_doc_names,
                selected_doc_names,
                web_retrieval_selected,
            )
            if not search_query:
                return jsonify({
                    'error': 'search_query is required and must be non-empty'
                }), 400
            if disable_rag:
                return jsonify({
                    'error': 'Retrieval is disabled when NO RAG is selected'
                }), 400
            if run_doc_retrieval and not state.retriever:
                return jsonify({'error': 'RAG retriever not available'}), 503
            retrieval_query = rag_service.build_structured_retrieval_query(
                search_query
            )
            logger.info(
                "Re-running retrieval with structured query: %s...",
                retrieval_query[:120],
            )
            vector_docs: List[Document] = []
            web_docs: List[Document] = []

            if run_doc_retrieval:
                if selected_doc_names:
                    logger.info(
                        "Using selected doc_name filters: %s",
                        selected_doc_names,
                    )
                vector_docs = rag_service.retrieve_docs_with_timeout(
                    retrieval_query,
                    selected_doc_names=selected_doc_names,
                    total_k=10,
                )

            if web_retrieval_selected:
                web_docs = rag_service.retrieve_web_docs(
                    retrieval_query,
                    max_results=config.WEB_RETRIEVAL_RESULT_COUNT,
                )

            docs: List[Document] = []
            docs.extend(vector_docs)
            docs.extend(web_docs)

            if not docs:
                return jsonify({
                    'search_query': retrieval_query,
                    'chunks': [],
                    'selected_doc_names': selected_sources,
                    'message': 'No documents found for this query',
                }), 200

            chunks_payload = []
            for doc in vector_docs:
                metadata = getattr(doc, "metadata", {}) or {}
                metadata["source_type"] = metadata.get("source_type") or "vector"
                chunks_payload.append({
                    "content": doc.page_content,
                    "metadata": metadata,
                })
            for doc in web_docs:
                metadata = getattr(doc, "metadata", {}) or {}
                metadata["source_type"] = metadata.get("source_type") or "web"
                chunks_payload.append({
                    "content": doc.page_content,
                    "metadata": metadata,
                })

            elapsed = time.time() - request_start
            logger.info(
                "[/re-run-retrieval] Returned %d chunks in %.2fs",
                len(chunks_payload),
                elapsed,
            )
            return jsonify({
                'search_query': retrieval_query,
                'chunks': chunks_payload,
                'selected_doc_names': selected_sources,
            }), 200
        except TimeoutError as e:
            return jsonify({'error': str(e)}), 504
        except Exception as e:
            logger.error("[/re-run-retrieval] Error: %s", e)
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.route('/generate-prompt', methods=['POST'])
    def generate_prompt():
        """
        Generate a detailed image prompt using RAG-enhanced LLM based on system instruction
        """
        request_start = time.time()
        logger.info("=" * 50)
        logger.info("[/generate-prompt] Request received")

        try:
            data = request.get_json()
            logger.info(
                "Request data keys: %s",
                list(data.keys()) if data else 'None',
            )
            system_instruction = data.get('system_instruction', '')
            user_question = data.get(
                'user_question', 'A serene landscape at sunset'
            )
            raw_selected_doc_names = (data or {}).get('selected_doc_names')
            web_retrieval_selected = rag_service.is_web_retrieval_selected(
                raw_selected_doc_names
            )
            disable_rag = bool((data or {}).get('disable_rag')) or rag_service.is_no_rag_selected(
                raw_selected_doc_names
            )
            selected_doc_names = rag_service.sanitize_selected_doc_names(
                raw_selected_doc_names
            )
            selected_sources = rag_service.normalize_selected_sources(
                raw_selected_doc_names
            )
            run_doc_retrieval = rag_service.should_run_doc_retrieval(
                raw_selected_doc_names,
                selected_doc_names,
                web_retrieval_selected,
            )

            if not system_instruction:
                logger.warning("Request missing system instruction")
                return jsonify({
                    'error': 'System instruction is required'
                }), 400

            if not config.OPENAI_API_KEY:
                logger.error("OpenAI API key not configured")
                return jsonify({
                    'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'
                }), 500

            logger.info("Question: %s...", user_question[:100])
            structured_user_query = rag_service.build_structured_retrieval_query(
                user_question
            )

            override_search_query = (data or {}).get('search_query')
            override_chunks = (data or {}).get('chunks')
            use_provided_context = (
                override_search_query is not None
                and isinstance(override_chunks, list)
                and len(override_chunks) > 0
            )

            if (state.retriever is not None or web_retrieval_selected) and not disable_rag:
                logger.info(
                    "Attempting RAG retrieval with timeout protection..."
                )
                try:
                    if use_provided_context:
                        retrieval_query = (
                            override_search_query.strip()
                            if isinstance(override_search_query, str)
                            else str(override_search_query)
                        )
                        docs = [
                            Document(
                                page_content=c.get("content", "")
                                if isinstance(c, dict)
                                else str(c),
                                metadata=c.get("metadata", {})
                                if isinstance(c, dict)
                                else {},
                            )
                            for c in override_chunks
                        ]
                        logger.info(
                            "Using provided context: query length %d, %d chunks",
                            len(retrieval_query),
                            len(docs),
                        )
                        vector_docs = [
                            doc
                            for doc in docs
                            if isinstance(
                                getattr(doc, "metadata", {}), dict
                            )
                            and (getattr(doc, "metadata", {}) or {}).get(
                                "source_type"
                            )
                            != "web"
                        ]
                        web_docs = [
                            doc
                            for doc in docs
                            if isinstance(
                                getattr(doc, "metadata", {}), dict
                            )
                            and (getattr(doc, "metadata", {}) or {}).get(
                                "source_type"
                            )
                            == "web"
                        ]
                        vector_docs_for_prompt = vector_docs
                    else:
                        retrieval_query = structured_user_query
                        vector_docs = []
                        web_docs = []

                        if run_doc_retrieval:
                            logger.info(
                                "Starting similarity retrieval from structured query..."
                            )
                            if selected_doc_names:
                                logger.info(
                                    "Applying selected doc_name filters: %s",
                                    selected_doc_names,
                                )
                            vector_docs = rag_service.retrieve_docs_with_timeout(
                                retrieval_query,
                                timeout_sec=20,
                                selected_doc_names=selected_doc_names,
                                total_k=10,
                            )

                        if web_retrieval_selected:
                            logger.info(
                                "Starting web retrieval from structured query..."
                            )
                            web_docs = rag_service.retrieve_web_docs(
                                retrieval_query,
                                max_results=config.WEB_RETRIEVAL_RESULT_COUNT,
                            )

                        docs = []
                        docs.extend(vector_docs)
                        docs.extend(web_docs)
                        if not docs:
                            logger.info(
                                "No documents retrieved; switching to direct prompt generation"
                            )
                            response = _invoke_openai_and_track([
                                {"role": "system", "content": system_instruction},
                                {
                                    "role": "user",
                                    "content": (
                                        "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                                        f"Structured Clinical Query: {structured_user_query}\n"
                                        f"Original User Question: {user_question}"
                                    ),
                                },
                            ])
                            generated_prompt = response.content.strip()
                            return jsonify({
                                'prompt': generated_prompt,
                                'success': True,
                                'search_query': structured_user_query,
                                'selected_doc_names': selected_sources,
                                'disable_rag': False,
                                'chunks': [],
                            })
                        logger.info(
                            "Retrieved %d documents in time",
                            len(docs),
                        )
                        doc_names = sorted({
                            rag_service.extract_doc_name_from_doc(doc)
                            for doc in docs
                            if rag_service.extract_doc_name_from_doc(doc)
                        })
                        if doc_names:
                            logger.info(
                                "Filtered doc_name values used for retrieval: %s",
                                doc_names,
                            )
                        if docs:
                            logger.info(
                                "Doc 1: %s...",
                                docs[0].page_content[:80],
                            )

                        vector_docs_for_prompt = vector_docs

                    logger.info(
                        "Using %d/%d vector chunks for prompt context",
                        len(vector_docs_for_prompt),
                        len(vector_docs),
                    )

                    context = rag_service.build_combined_context(
                        vector_docs_for_prompt, web_docs
                    )
                    logger.info(
                        "Context assembled (%d chars)",
                        len(context),
                    )

                    construction_prompt = f"""
                    Retrieved High-Yield Medical Context:
                    {context}

                    Structured Clinical Query:
                    {structured_user_query}

                    Original User Question:
                    {user_question}

                    Return a complete structured and detailed image generation prompt following the system instruction guidelines.
                    """

                    logger.info("Generating prompt with RAG context...")
                    response = _invoke_openai_and_track([
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": construction_prompt},
                    ])

                    generated_prompt = response.content.strip()
                    logger.info(
                        "Generated prompt with RAG (%d chars)",
                        len(generated_prompt),
                    )

                    chunks_payload = []
                    for doc in vector_docs:
                        metadata = getattr(doc, "metadata", {}) or {}
                        metadata["source_type"] = (
                            metadata.get("source_type") or "vector"
                        )
                        chunks_payload.append({
                            "content": doc.page_content,
                            "metadata": metadata,
                        })
                    for doc in web_docs:
                        metadata = getattr(doc, "metadata", {}) or {}
                        metadata["source_type"] = (
                            metadata.get("source_type") or "web"
                        )
                        chunks_payload.append({
                            "content": doc.page_content,
                            "metadata": metadata,
                        })

                    return jsonify({
                        'prompt': generated_prompt,
                        'success': True,
                        'search_query': retrieval_query,
                        'selected_doc_names': selected_sources,
                        'disable_rag': False,
                        'chunks': chunks_payload,
                    })

                except Exception as e:
                    logger.warning(
                        "RAG failed (%s), falling back to direct generation",
                        e,
                    )
                    logger.warning(traceback.format_exc())
                    response = _invoke_openai_and_track([
                        {"role": "system", "content": system_instruction},
                        {
                            "role": "user",
                            "content": (
                                "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                                f"Structured Clinical Query: {structured_user_query}\n"
                                f"Original User Question: {user_question}"
                            ),
                        },
                    ])
                    generated_prompt = response.content.strip()
                    logger.info(
                        "Fallback prompt generated (%d chars)",
                        len(generated_prompt),
                    )
            elif disable_rag:
                logger.info(
                    "NO RAG mode selected; generating prompt from structured query and original user question"
                )
                response = _invoke_openai_and_track([
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": (
                            "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                            f"Structured Clinical Query: {structured_user_query}\n"
                            f"Original User Question: {user_question}"
                        ),
                    },
                ])
                generated_prompt = response.content.strip()
                logger.info(
                    "Direct prompt generated (NO RAG mode) (%d chars)",
                    len(generated_prompt),
                )
            else:
                logger.warning(
                    "RAG system not available, using direct generation without retrieval"
                )
                response = _invoke_openai_and_track([
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": (
                            "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                            f"Structured Clinical Query: {structured_user_query}\n"
                            f"Original User Question: {user_question}"
                        ),
                    },
                ])
                generated_prompt = response.content.strip()
                logger.info(
                    "Direct prompt generated (%d chars)",
                    len(generated_prompt),
                )

            request_time = time.time() - request_start
            logger.info(
                "[/generate-prompt] Success in %.2fs",
                request_time,
            )
            logger.info("=" * 50)

            return jsonify({
                'prompt': generated_prompt,
                'success': True,
                'search_query': structured_user_query,
                'selected_doc_names': selected_sources,
                'disable_rag': disable_rag,
                'chunks': [],
            })

        except Exception as e:
            request_time = time.time() - request_start
            logger.error(
                "[/generate-prompt] Error after %.2fs: %s",
                request_time,
                e,
            )
            logger.error(traceback.format_exc())
            logger.info("=" * 50)
            return jsonify({
                'error': f'Error generating prompt: {str(e)}'
            }), 500

    @app.route('/doc-names', methods=['GET'])
    def get_doc_names():
        """Return distinct source document names available in the vector store."""
        latest_names = fetch_distinct_doc_names(state.mongo_client)
        if latest_names:
            state.known_doc_names = latest_names
        names = sorted([
            name
            for name in state.known_doc_names
            if isinstance(name, str) and name.strip()
        ])
        return jsonify({'doc_names': names, 'count': len(names)}), 200
