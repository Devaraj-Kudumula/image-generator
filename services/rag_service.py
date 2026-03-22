"""
RAG: retrieval helpers, source normalization, web search, and context building.
"""
import re
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Any, Optional, Tuple

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

import config
from app_state import state
from db import fetch_distinct_doc_names
from services import ondemand_docs_service
from services.llm_metrics_service import record_langchain_openai_call

logger = logging.getLogger(__name__)

BASE_DOC_PREFIX = "base::"
SESSION_DOC_PREFIX = "session::"


def _with_vector_metadata(doc: Document) -> Document:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["source_type"] = metadata.get("source_type") or "vector"
    return Document(page_content=doc.page_content, metadata=metadata)


def _similarity_search(
    query: str,
    k: int,
    pre_filter: Optional[Dict[str, Any]] = None,
    vectorstore: Any = None,
) -> List[Document]:
    """Run plain vector similarity search and normalize metadata."""
    target_vectorstore = vectorstore or state.vectorstore
    if target_vectorstore is None:
        return []

    search_kwargs: Dict[str, Any] = {"query": query, "k": max(1, int(k))}
    if pre_filter is not None:
        search_kwargs["pre_filter"] = pre_filter

    try:
        docs = target_vectorstore.similarity_search(**search_kwargs)
    except TypeError:
        if "pre_filter" in search_kwargs:
            docs = target_vectorstore.similarity_search(
                query=search_kwargs["query"],
                k=search_kwargs["k"],
            )
        else:
            docs = []
    except Exception:
        docs = []

    return [
        _with_vector_metadata(doc)
        for doc in (docs or [])
        if isinstance(doc, Document)
    ]


def _similarity_search_by_doc_name(
    query: str,
    k: int,
    doc_name: str,
    vectorstore: Any,
) -> List[Document]:
    """Try known doc_name metadata paths to avoid filter misses across schemas."""
    if not doc_name:
        return []

    filter_paths = (
        "metadata.doc_name",
        "metadata.metadata.doc_name",
        "doc_name",
    )

    for path in filter_paths:
        filtered_docs = _similarity_search(
            query=query,
            k=k,
            pre_filter={path: {"$eq": doc_name}},
            vectorstore=vectorstore,
        )
        if filtered_docs:
            logger.info(
                "Filtered retrieval matched path '%s' for doc_name='%s'",
                path,
                doc_name,
            )
            return filtered_docs

    return []


def extract_doc_name_from_doc(doc: Document) -> Optional[str]:
    metadata = getattr(doc, "metadata", {}) or {}
    if isinstance(metadata, dict):
        top_level = metadata.get("doc_name")
        if isinstance(top_level, str) and top_level.strip():
            return top_level
        nested = metadata.get("metadata")
        if isinstance(nested, dict):
            nested_doc_name = nested.get("doc_name")
            if isinstance(nested_doc_name, str) and nested_doc_name.strip():
                return nested_doc_name
    return None


def extract_session_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("session_id") or "").strip()


def _split_selected_sources(selected_doc_names: Any) -> Tuple[List[str], List[str]]:
    if not isinstance(selected_doc_names, list):
        return ([], [])

    base_candidates: List[str] = []
    session_candidates: List[str] = []
    for raw_value in selected_doc_names:
        if not isinstance(raw_value, str):
            continue
        value = raw_value.strip()
        if not value:
            continue
        normalized_upper = value.upper()
        if normalized_upper in {
            config.NO_RAG_OPTION_VALUE,
            config.WEB_RETRIEVAL_OPTION_VALUE,
        }:
            continue
        if value.startswith(SESSION_DOC_PREFIX):
            session_value = value[len(SESSION_DOC_PREFIX):].strip()
            if session_value:
                session_candidates.append(session_value)
            continue
        if value.startswith(BASE_DOC_PREFIX):
            base_value = value[len(BASE_DOC_PREFIX):].strip()
            if base_value:
                base_candidates.append(base_value)
            continue
        base_candidates.append(value)
    return (base_candidates, session_candidates)


def sanitize_selected_doc_names(
    selected_doc_names: Any,
    session_id: str = "",
) -> List[str]:
    if not isinstance(selected_doc_names, list):
        return []

    selected_sources: List[str] = []
    if is_no_rag_selected(selected_doc_names):
        return [config.NO_RAG_OPTION_VALUE]
    if is_web_retrieval_selected(selected_doc_names):
        selected_sources.append(config.WEB_RETRIEVAL_OPTION_VALUE)

    base_candidates, session_candidates = _split_selected_sources(
        selected_doc_names
    )

    candidate_names = [name for name in base_candidates if name]
    if not state.known_doc_names:
        state.known_doc_names = fetch_distinct_doc_names(state.mongo_client)

    if any(name not in set(state.known_doc_names) for name in candidate_names):
        latest_doc_names = fetch_distinct_doc_names(state.mongo_client)
        if latest_doc_names:
            state.known_doc_names = latest_doc_names

    valid_names = set(state.known_doc_names)
    selected_sources.extend(
        f"{BASE_DOC_PREFIX}{name}"
        for name in candidate_names
        if name in valid_names
    )

    if session_id:
        session_valid_names = set(
            ondemand_docs_service.list_session_doc_names(session_id)
        )
        selected_sources.extend(
            f"{SESSION_DOC_PREFIX}{name}"
            for name in session_candidates
            if name in session_valid_names
        )

    return selected_sources


def is_no_rag_selected(selected_doc_names: Any) -> bool:
    if not isinstance(selected_doc_names, list):
        return False
    return any(
        isinstance(name, str)
        and name.strip().upper() == config.NO_RAG_OPTION_VALUE
        for name in selected_doc_names
    )


def is_web_retrieval_selected(selected_doc_names: Any) -> bool:
    if not isinstance(selected_doc_names, list):
        return False
    return any(
        isinstance(name, str)
        and name.strip().upper() == config.WEB_RETRIEVAL_OPTION_VALUE
        for name in selected_doc_names
    )


def normalize_selected_sources(
    selected_doc_names: Any,
    session_id: str = "",
) -> List[str]:
    """Preserve selected sources while validating known document names."""
    return sanitize_selected_doc_names(selected_doc_names, session_id=session_id)


def should_run_doc_retrieval(
    raw_selected_sources: Any,
    sanitized_doc_names: List[str],
    web_selected: bool,
) -> bool:
    """Decide whether vector retrieval should run based on selected sources."""
    base_selected, session_selected = _split_selected_sources(sanitized_doc_names)
    if base_selected or session_selected:
        return True
    if not isinstance(raw_selected_sources, list) or len(raw_selected_sources) == 0:
        return True

    normalized = [
        str(value).strip().upper()
        for value in raw_selected_sources
        if isinstance(value, str)
    ]
    non_special_selected = [
        value
        for value in normalized
        if value
        not in {config.NO_RAG_OPTION_VALUE, config.WEB_RETRIEVAL_OPTION_VALUE}
    ]
    if non_special_selected:
        return True

    if (
        web_selected
        and len(normalized) == 1
        and normalized[0] == config.WEB_RETRIEVAL_OPTION_VALUE
    ):
        return False

    return True


def _strip_html_to_text(html: str) -> str:
    """Convert raw HTML into plain text with light cleanup."""
    content = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    content = re.sub(r"(?is)<style.*?>.*?</style>", " ", content)
    content = re.sub(r"(?is)<[^>]+>", " ", content)
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def _fetch_webpage_text(
    url: str, timeout_sec: int = 10, max_chars: int = 2500
) -> str:
    """Fetch and extract plain text from a webpage URL using LangChain loader."""
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            requests_kwargs={"timeout": timeout_sec},
        )
        loaded_docs = loader.load()
        if not loaded_docs:
            return ""
        text = re.sub(
            r"\s+", " ", loaded_docs[0].page_content or ""
        ).strip()
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
    except Exception as e:
        logger.warning("Could not fetch webpage text for '%s': %s", url, e)
        return ""


def _search_web_with_serper_direct(
    query_text: str, result_limit: int
) -> List[Dict[str, str]]:
    if not config.SERPER_API_KEY:
        return []

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": config.SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "q": query_text,
                "gl": "us",
                "hl": "en",
                "num": max(1, min(result_limit, 10)),
            },
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError:
        logger.warning(
            "Direct Serper request failed with HTTP %s: %s",
            response.status_code,
            response.text[:200],
        )
        return []
    except Exception as direct_serper_error:
        logger.warning("Direct Serper request failed: %s", direct_serper_error)
        return []

    normalized_direct_results: List[Dict[str, str]] = []
    for item in payload.get("organic") or []:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not link:
            continue
        normalized_direct_results.append({
            "link": link,
            "title": title,
            "snippet": snippet,
        })

    logger.info(
        "Direct Serper request returned %d results",
        len(normalized_direct_results),
    )
    return normalized_direct_results


def search_web_with_langchain(
    query: str, max_results: int = None
) -> List[Dict[str, str]]:
    """Run Google CSE through LangChain wrapper and normalize results."""
    if max_results is None:
        max_results = config.WEB_RETRIEVAL_RESULT_COUNT

    def _search_web_with_serper(
        query_text: str, result_limit: int
    ) -> List[Dict[str, str]]:
        if state.google_serper_wrapper is None:
            logger.warning(
                "GoogleSerperAPIWrapper unavailable; trying direct Serper request"
            )
            return _search_web_with_serper_direct(query_text, result_limit)

        try:
            results = state.google_serper_wrapper.results(query_text)
        except Exception as serper_error:
            logger.warning(
                "GoogleSerperAPIWrapper search failed: %s; trying direct Serper request",
                serper_error,
            )
            return _search_web_with_serper_direct(query_text, result_limit)

        organic = (
            results.get("organic") or []
            if isinstance(results, dict)
            else (results or [])
        )

        normalized_results: List[Dict[str, str]] = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            link = str(item.get("link") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(
                item.get("snippet") or item.get("text") or ""
            ).strip()
            if not link:
                continue
            normalized_results.append({
                "link": link,
                "title": title,
                "snippet": snippet,
            })
            if len(normalized_results) >= max(1, min(result_limit, 10)):
                break

        logger.info(
            "GoogleSerperAPIWrapper returned %d results",
            len(normalized_results),
        )
        return normalized_results

    return _search_web_with_serper(query, max_results)


def retrieve_web_docs(
    query: str, max_results: int = None
) -> List[Document]:
    """Retrieve web search results and extracted content through LangChain tools."""
    if max_results is None:
        max_results = config.WEB_RETRIEVAL_RESULT_COUNT
    items = search_web_with_langchain(query, max_results=max_results)
    if not items:
        return []

    try:
        docs: List[Document] = []

        for item in items:
            link = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            if not link:
                continue

            extracted_text = _fetch_webpage_text(link)
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if snippet:
                content_parts.append(f"Snippet: {snippet}")
            if extracted_text:
                content_parts.append(f"Extracted Content: {extracted_text}")

            combined_content = "\n".join(content_parts).strip()
            if not combined_content:
                continue

            docs.append(
                Document(
                    page_content=combined_content,
                    metadata={
                        "source_type": "web",
                        "url": link,
                        "title": title,
                    },
                )
            )

        logger.info("Web retrieval returned %d documents", len(docs))
        return docs
    except Exception as e:
        logger.warning("Web retrieval failed: %s", e)
        return []


def build_combined_context(
    vector_docs: List[Document], web_docs: List[Document]
) -> str:
    """Build the single context string expected by prompt synthesis with clear separators."""
    sections: List[str] = []

    if vector_docs:
        retrieved_chunks = "\n\n".join(
            doc.page_content
            for doc in vector_docs
            if getattr(doc, "page_content", "")
        )
        sections.append(f"Retrieved Chunks:\n{retrieved_chunks}")

    if web_docs:
        web_content = "\n\n".join(
            doc.page_content
            for doc in web_docs
            if getattr(doc, "page_content", "")
        )
        sections.append(f"Web Retrieved Content:\n{web_content}")

    return "\n\n==============================\n\n".join(
        section for section in sections if section.strip()
    )


def build_structured_retrieval_query(user_question: str) -> str:
    """Build a concise clinical retrieval query from the raw user question."""
    base_question = (user_question or "").strip()
    if not base_question:
        return ""

    if state.llm is None:
        return base_question

    extract_system = """You are a medical retrieval query generator.

Your task is to convert the user’s medical question into a single, highly optimized search query for retrieving relevant medical textbook content.

The query must be dense, specific, and medically rich.

Instructions:

• Identify the core medical concept (anatomy, disease, pathology, physiology, mechanism)
• Expand with related anatomical structures, systems, and pathways
• Include mechanism-related terms (flow, obstruction, degeneration, signaling, etc.)
• Include synonyms or closely related medical terminology
• Remove conversational phrases (e.g., "explain", "what is")

Focus on retrieving:
• anatomical relationships
• structural hierarchy
• underlying mechanisms
• system-level context

Output Rules:

• Return ONE single-line query
• Do NOT explain anything
• Do NOT return bullets or JSON

Example:

User: anatomy of subclavian steal syndrome

Output:
subclavian steal syndrome anatomy vertebral artery subclavian artery stenosis basilar artery posterior circulation retrograde blood flow vascular pathway"""

    try:
        logger.info("Extracting structured retrieval query...")
        extract_response = state.llm.invoke([
            {"role": "system", "content": extract_system},
            {"role": "user", "content": base_question},
        ])
        model_name = getattr(state.llm, "model_name", None) or "gpt-4"
        record_langchain_openai_call(extract_response, model_name)
        structured_query = (extract_response.content or "").strip()
        if structured_query:
            logger.info(
                "Structured retrieval query: %s...",
                structured_query,
            )
            return structured_query
        logger.warning(
            "Structured retrieval query was empty; falling back to raw user question"
        )
        return base_question
    except Exception as extract_error:
        logger.warning(
            "Failed to extract structured retrieval query: %s",
            extract_error,
        )
        return base_question


def retrieve_docs_with_timeout(
    query: str,
    timeout_sec: int = 20,
    selected_doc_names: Optional[List[str]] = None,
    total_k: int = 10,
    session_id: str = "",
    equal_per_selected_doc: bool = False,
) -> List[Document]:
    """Run retrieval with timeout across base and session-scoped vector stores."""

    def retrieve_docs():
        try:
            sanitized_sources = sanitize_selected_doc_names(
                selected_doc_names,
                session_id=session_id,
            )
            selected_base, selected_session = _split_selected_sources(
                sanitized_sources
            )

            chunk_count = max(1, int(total_k))
            docs: List[Document] = []

            session_vectorstore = (
                ondemand_docs_service.get_session_vectorstore(session_id)
                if session_id
                else None
            )
            session_doc_names = (
                ondemand_docs_service.list_session_doc_names(session_id)
                if session_id
                else []
            )

            if not selected_base and not selected_session:
                if session_vectorstore is not None and session_doc_names:
                    # Prefer ephemeral session docs when no explicit source is selected.
                    selected_session = session_doc_names
                elif state.retriever is not None:
                    logger.info("Using base retriever similarity search")
                    base_docs = state.retriever.invoke(query)
                    return [
                        _with_vector_metadata(doc)
                        for doc in (base_docs or [])
                        if isinstance(doc, Document)
                    ]
                elif session_vectorstore is not None:
                    return _similarity_search(
                        query=query,
                        k=chunk_count,
                        vectorstore=session_vectorstore,
                    )
                raise ValueError("Retriever not available")

            selected_entries: List[Tuple[str, str]] = []
            selected_entries.extend(("base", name) for name in selected_base)
            selected_entries.extend(("session", name) for name in selected_session)

            if equal_per_selected_doc and selected_entries:
                base_k, remainder = divmod(chunk_count, len(selected_entries))
                for index, (source_type, doc_name) in enumerate(selected_entries):
                    per_doc_k = base_k + (1 if index < remainder else 0)
                    if per_doc_k <= 0:
                        continue
                    target_vectorstore = (
                        state.vectorstore
                        if source_type == "base"
                        else session_vectorstore
                    )
                    if target_vectorstore is None:
                        continue
                    filtered_docs = _similarity_search_by_doc_name(
                        query=query,
                        k=per_doc_k,
                        doc_name=doc_name,
                        vectorstore=target_vectorstore,
                    )
                    if (
                        source_type == "session"
                        and len(filtered_docs) == 0
                        and session_id
                    ):
                        filtered_docs = ondemand_docs_service.retrieve_docs_bruteforce(
                            session_id=session_id,
                            query=query,
                            doc_name=doc_name,
                            k=per_doc_k,
                        )
                    logger.info(
                        "Equal retrieval (%s): doc_name='%s', requested_k=%s, returned=%d",
                        source_type,
                        doc_name,
                        per_doc_k,
                        len(filtered_docs),
                    )
                    docs.extend(filtered_docs)
                return docs[:chunk_count]

            active_groups = 0
            if selected_base:
                active_groups += 1
            if selected_session:
                active_groups += 1
            group_k = max(1, chunk_count // max(1, active_groups))

            if selected_base and state.vectorstore is not None:
                per_doc_k = max(1, group_k // len(selected_base))
                for doc_name in selected_base:
                    filtered_docs = _similarity_search_by_doc_name(
                        query=query,
                        k=per_doc_k,
                        doc_name=doc_name,
                        vectorstore=state.vectorstore,
                    )
                    logger.info(
                        "Base filter retrieval: doc_name='%s', requested_k=%s, returned=%d",
                        doc_name,
                        per_doc_k,
                        len(filtered_docs),
                    )
                    docs.extend(filtered_docs)

            if selected_session and session_vectorstore is not None:
                per_doc_k = max(1, group_k // len(selected_session))
                for doc_name in selected_session:
                    filtered_docs = _similarity_search_by_doc_name(
                        query=query,
                        k=per_doc_k,
                        doc_name=doc_name,
                        vectorstore=session_vectorstore,
                    )
                    logger.info(
                        "Session filter retrieval: doc_name='%s', requested_k=%s, returned=%d",
                        doc_name,
                        per_doc_k,
                        len(filtered_docs),
                    )
                    docs.extend(filtered_docs)

            return docs[:chunk_count]
        except Exception as e:
            logger.error("Retriever invoke error: %s", e)
            raise

    with ThreadPoolExecutor(max_workers=5) as executor:
        future = executor.submit(retrieve_docs)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            logger.error(
                "Document retrieval timed out after %s seconds",
                timeout_sec,
            )
            raise TimeoutError("Embedding API timeout - please retry")
