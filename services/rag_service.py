"""
RAG: retrieval helpers, source normalization, web search, and context building.
"""
import re
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Any, Optional

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

import config
from app_state import state
from db import fetch_distinct_doc_names
from services.llm_metrics_service import record_langchain_openai_call

logger = logging.getLogger(__name__)


def _as_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _with_similarity_metadata(
    doc: Document,
    score: Optional[float],
    score_direction: Optional[str] = None,
) -> Document:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    metadata["source_type"] = metadata.get("source_type") or "vector"
    similarity_score = _as_float_or_none(score)
    if similarity_score is not None:
        metadata["similarity_score"] = similarity_score
    if score_direction in {"higher_is_better", "lower_is_better"}:
        metadata["similarity_score_direction"] = score_direction
    return Document(page_content=doc.page_content, metadata=metadata)


def _similarity_search_scored(
    query: str,
    k: int,
    pre_filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Run vector search and attach similarity scores when available."""
    if state.vectorstore is None:
        return []

    search_kwargs: Dict[str, Any] = {"query": query, "k": max(1, int(k))}
    if pre_filter is not None:
        search_kwargs["pre_filter"] = pre_filter

    scored_methods = [
        "similarity_search_with_relevance_scores",
        "similarity_search_with_score",
    ]

    for method_name in scored_methods:
        method = getattr(state.vectorstore, method_name, None)
        if not callable(method):
            continue
        try:
            pairs = method(**search_kwargs)
        except TypeError:
            if "pre_filter" in search_kwargs:
                fallback_kwargs = {
                    "query": search_kwargs["query"],
                    "k": search_kwargs["k"],
                }
                try:
                    pairs = method(**fallback_kwargs)
                except Exception:
                    continue
            else:
                continue
        except Exception:
            continue

        score_direction = (
            "higher_is_better"
            if method_name == "similarity_search_with_relevance_scores"
            else "lower_is_better"
        )

        docs: List[Document] = []
        for pair in pairs or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            doc, score = pair[0], pair[1]
            if not isinstance(doc, Document):
                continue
            docs.append(
                _with_similarity_metadata(
                    doc,
                    score,
                    score_direction=score_direction,
                )
            )

        if docs:
            return docs

    try:
        docs = state.vectorstore.similarity_search(**search_kwargs)
    except TypeError:
        if "pre_filter" in search_kwargs:
            docs = state.vectorstore.similarity_search(
                query=search_kwargs["query"],
                k=search_kwargs["k"],
            )
        else:
            docs = []
    except Exception:
        docs = []

    return [
        _with_similarity_metadata(doc, None)
        for doc in (docs or [])
        if isinstance(doc, Document)
    ]


def select_top_vector_docs_for_prompt(
    vector_docs: List[Document],
    top_n: int = 5,
) -> List[Document]:
    """Use similarity scores when available; otherwise keep retrieval order."""
    if top_n <= 0:
        return []
    if not vector_docs:
        return []

    has_any_score = any(
        isinstance(getattr(doc, "metadata", {}), dict)
        and (getattr(doc, "metadata", {}) or {}).get("similarity_score")
        is not None
        for doc in vector_docs
    )
    if not has_any_score:
        return vector_docs[:top_n]

    directions = {
        (getattr(doc, "metadata", {}) or {}).get("similarity_score_direction")
        for doc in vector_docs
        if isinstance(getattr(doc, "metadata", {}), dict)
    }
    directions.discard(None)
    if len(directions) != 1:
        return vector_docs[:top_n]

    direction = next(iter(directions))

    def _score(doc: Document) -> float:
        metadata = getattr(doc, "metadata", {}) or {}
        value = _as_float_or_none(metadata.get("similarity_score"))
        if value is None:
            return float("inf") if direction == "lower_is_better" else float("-inf")
        return value

    sorted_docs = sorted(
        vector_docs,
        key=_score,
        reverse=(direction == "higher_is_better"),
    )
    return sorted_docs[:top_n]


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


def sanitize_selected_doc_names(selected_doc_names: Any) -> List[str]:
    if not isinstance(selected_doc_names, list):
        return []
    candidate_names = [
        str(name).strip()
        for name in selected_doc_names
        if isinstance(name, str) and str(name).strip()
    ]
    if not candidate_names:
        return []

    if not state.known_doc_names:
        state.known_doc_names = fetch_distinct_doc_names(state.mongo_client)

    if any(name not in set(state.known_doc_names) for name in candidate_names):
        latest_doc_names = fetch_distinct_doc_names(state.mongo_client)
        if latest_doc_names:
            state.known_doc_names = latest_doc_names

    valid_names = set(state.known_doc_names)
    if not valid_names:
        return []
    return [name for name in candidate_names if name in valid_names]


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


def normalize_selected_sources(selected_doc_names: Any) -> List[str]:
    """Preserve selected sources while validating known document names."""
    if not isinstance(selected_doc_names, list):
        return []

    normalized_sources: List[str] = []
    if is_no_rag_selected(selected_doc_names):
        return [config.NO_RAG_OPTION_VALUE]
    if is_web_retrieval_selected(selected_doc_names):
        normalized_sources.append(config.WEB_RETRIEVAL_OPTION_VALUE)

    sanitized_doc_names = sanitize_selected_doc_names(selected_doc_names)
    normalized_sources.extend(sanitized_doc_names)
    return normalized_sources


def should_run_doc_retrieval(
    raw_selected_sources: Any,
    sanitized_doc_names: List[str],
    web_selected: bool,
) -> bool:
    """Decide whether vector retrieval should run based on selected sources."""
    if sanitized_doc_names:
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

    extract_system = """Extract:
        1. Primary medical condition
        2. Mechanism keywords
        3. Clinical keywords

        Return short structured text only.
    """

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
                structured_query[:150],
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
) -> List[Document]:
    """Run retrieval with timeout; supports explicit selected doc_name filtering."""
    if not state.retriever:
        raise ValueError("Retriever not available")

    def retrieve_docs():
        try:
            sanitized_doc_names = sanitize_selected_doc_names(selected_doc_names)

            if sanitized_doc_names:
                if state.vectorstore is None:
                    raise ValueError(
                        "Vectorstore not available for doc_name filtered retrieval"
                    )

                chunk_count = max(1, int(total_k))
                doc_count = len(sanitized_doc_names)
                base_k = chunk_count // doc_count
                remainder = chunk_count % doc_count

                docs: List[Document] = []
                for index, doc_name in enumerate(sanitized_doc_names):
                    per_doc_k = base_k + (1 if index < remainder else 0)
                    if per_doc_k <= 0:
                        continue
                    pre_filter = {"metadata.doc_name": {"$eq": doc_name}}
                    filtered_docs = _similarity_search_scored(
                        query=query,
                        k=per_doc_k,
                        pre_filter=pre_filter,
                    )
                    logger.info(
                        "Doc filter retrieval: doc_name='%s', requested_k=%s, returned=%d",
                        doc_name,
                        per_doc_k,
                        len(filtered_docs),
                    )
                    docs.extend(filtered_docs)

                return docs

            if state.vectorstore is not None:
                logger.info("Using scored vector similarity search")
                docs = _similarity_search_scored(
                    query=query,
                    k=max(1, int(total_k)),
                )
                if docs:
                    return docs

            logger.warning(
                "Scored vector search unavailable/empty; using retriever fallback"
            )
            docs = state.retriever.invoke(query)
            return [
                _with_similarity_metadata(doc, None)
                for doc in (docs or [])
                if isinstance(doc, Document)
            ]
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
