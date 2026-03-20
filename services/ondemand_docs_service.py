"""
Session-scoped on-demand document ingestion and retrieval support.
"""
import logging
import re
import tempfile
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

import config
from app_state import state

logger = logging.getLogger(__name__)

SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,80}$")


def _sanitize_session_id(session_id: Any) -> Optional[str]:
    value = str(session_id or "").strip()
    if not value:
        return None
    if not SESSION_ID_RE.match(value):
        return None
    return value


def _collection_name_for_session(session_id: str) -> str:
    return f"session_{session_id}_chunks"


def _get_or_create_embedding_model() -> Optional[OpenAIEmbeddings]:
    if state.embedding_model is not None:
        return state.embedding_model
    if not config.OPENAI_API_KEY:
        return None
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=config.OPENAI_API_KEY,
        request_timeout=45,
        max_retries=2,
        show_progress_bar=False,
    )


def _create_vector_index_if_missing(collection: Any, index_name: str) -> None:
    try:
        existing = [idx.get("name") for idx in collection.list_search_indexes()]
    except Exception:
        existing = []

    if index_name in existing:
        return

    definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine",
            },
            {
                "type": "filter",
                "path": "metadata.doc_name",
            },
        ]
    }

    try:
        collection.database.command(
            {
                "createSearchIndexes": collection.name,
                "indexes": [
                    {
                        "name": index_name,
                        "type": "vectorSearch",
                        "definition": definition,
                    }
                ],
            }
        )
        logger.info(
            "Created vector search index '%s' for collection '%s'",
            index_name,
            collection.name,
        )
    except Exception as e:
        logger.warning(
            "Could not create vector search index '%s' for '%s': %s",
            index_name,
            collection.name,
            e,
        )


def _build_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    if not config.ONDEMAND_MONGODB_URI:
        return None

    mongo_client = MongoClient(
        config.ONDEMAND_MONGODB_URI,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        retryWrites=True,
    )
    mongo_client.admin.command("ping")

    collection_name = _collection_name_for_session(session_id)
    collection = mongo_client[config.ONDEMAND_DB_NAME][collection_name]
    _create_vector_index_if_missing(collection, config.ONDEMAND_INDEX_NAME)

    embedding_model = _get_or_create_embedding_model()
    if embedding_model is None:
        mongo_client.close()
        return None

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name=config.ONDEMAND_INDEX_NAME,
        text_key="chunk_text",
    )

    context = {
        "session_id": session_id,
        "mongo_client": mongo_client,
        "collection": collection,
        "collection_name": collection_name,
        "vectorstore": vectorstore,
        "known_doc_names": [],
        "last_access": time.time(),
    }
    return context


def get_session_context(session_id: Any, create: bool = True) -> Optional[Dict[str, Any]]:
    normalized = _sanitize_session_id(session_id)
    if not normalized:
        return None

    cached = state.ondemand_sessions.get(normalized)
    if cached is not None:
        cached["last_access"] = time.time()
        return cached

    if not create:
        return None

    try:
        context = _build_session_context(normalized)
    except Exception as e:
        logger.error("Failed to initialize on-demand session context: %s", e)
        return None

    if context is None:
        return None

    state.ondemand_sessions[normalized] = context
    return context


def _refresh_session_doc_names(context: Dict[str, Any]) -> List[str]:
    collection = context["collection"]
    names = set()
    try:
        for value in collection.distinct("metadata.doc_name"):
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    except Exception:
        pass

    sorted_names = sorted(names)
    context["known_doc_names"] = sorted_names
    return sorted_names


def list_session_doc_names(session_id: Any) -> List[str]:
    context = get_session_context(session_id, create=True)
    if context is None:
        return []
    return _refresh_session_doc_names(context)


def upload_pdf_for_session(session_id: Any, file_storage: Any) -> Dict[str, Any]:
    context = get_session_context(session_id, create=True)
    if context is None:
        raise ValueError("Invalid or unavailable session context")

    if file_storage is None:
        raise ValueError("No file provided")

    filename = str(getattr(file_storage, "filename", "") or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")

    doc_name = Path(filename).stem.strip() or "uploaded_document"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        file_storage.save(tmp_path)

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        if not documents:
            raise ValueError("Uploaded PDF did not contain readable text")

        normalized_docs: List[Document] = []
        for doc in documents:
            content = str(getattr(doc, "page_content", "") or "").strip()
            if not content:
                continue
            metadata = dict(getattr(doc, "metadata", {}) or {})
            normalized_docs.append(
                Document(page_content=content, metadata=metadata)
            )

        combined_text = "\n\n".join(doc.page_content for doc in normalized_docs).strip()
        if not combined_text:
            raise ValueError(
                "Could not extract readable text from this PDF. "
                "If this is a scanned/image-only PDF, run OCR first and upload the OCR text PDF."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
        )
        chunked_docs = text_splitter.split_documents(normalized_docs)
        if not chunked_docs:
            fallback_chunks = text_splitter.split_text(combined_text)
            chunked_docs = [
                Document(page_content=chunk.strip(), metadata={})
                for chunk in (fallback_chunks or [])
                if isinstance(chunk, str) and chunk.strip()
            ]
        if not chunked_docs:
            raise ValueError("Could not split uploaded PDF into chunks")

        embedding_model = _get_or_create_embedding_model()
        if embedding_model is None:
            raise ValueError("Embedding model is not available")

        collection = context["collection"]
        collection.delete_many({"metadata.doc_name": doc_name})

        mongo_docs = []
        for idx, chunk in enumerate(chunked_docs):
            content = (chunk.page_content or "").strip()
            if not content:
                continue
            embed_vector = embedding_model.embed_query(content)
            raw_meta = getattr(chunk, "metadata", {}) or {}
            mongo_docs.append(
                {
                    "chunk_id": idx,
                    "chunk_text": content,
                    "embedding": embed_vector,
                    "metadata": {
                        "doc_name": doc_name,
                        "chunk_size": len(content),
                        "page": raw_meta.get("page"),
                        "source": raw_meta.get("source"),
                    },
                }
            )

        if not mongo_docs:
            raise ValueError("No valid text chunks were extracted from the PDF")

        collection.insert_many(mongo_docs)
        session_doc_names = _refresh_session_doc_names(context)
        return {
            "doc_name": doc_name,
            "chunks_inserted": len(mongo_docs),
            "session_doc_names": session_doc_names,
        }
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def get_session_vectorstore(session_id: Any):
    context = get_session_context(session_id, create=True)
    if context is None:
        return None
    return context.get("vectorstore")


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        af = float(a)
        bf = float(b)
        dot += af * bf
        norm_a += af * af
        norm_b += bf * bf
    if norm_a <= 0.0 or norm_b <= 0.0:
        return -1.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def retrieve_docs_bruteforce(
    session_id: Any,
    query: str,
    doc_name: str,
    k: int,
) -> List[Document]:
    """Fallback retrieval for session docs when vector index is unavailable."""
    context = get_session_context(session_id, create=True)
    if context is None:
        return []

    embedding_model = _get_or_create_embedding_model()
    if embedding_model is None:
        return []

    query_embedding = embedding_model.embed_query(query)
    cursor = context["collection"].find(
        {"metadata.doc_name": doc_name},
        {"chunk_text": 1, "metadata": 1, "embedding": 1},
    )

    scored: List[Dict[str, Any]] = []
    for item in cursor:
        chunk_embedding = item.get("embedding") or []
        score = _cosine_similarity(query_embedding, chunk_embedding)
        scored.append({"doc": item, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: max(1, int(k))]
    docs: List[Document] = []
    for entry in top:
        payload = entry.get("doc") or {}
        metadata = dict(payload.get("metadata") or {})
        metadata["source_type"] = "vector"
        metadata["similarity_score"] = float(entry.get("score") or 0.0)
        docs.append(
            Document(
                page_content=str(payload.get("chunk_text") or ""),
                metadata=metadata,
            )
        )
    return docs


def clear_session(session_id: Any) -> bool:
    normalized = _sanitize_session_id(session_id)
    if not normalized:
        return False

    context = state.ondemand_sessions.pop(normalized, None)
    if context is None:
        # Best effort cleanup even if cache entry is not present.
        try:
            temp_client = MongoClient(
                config.ONDEMAND_MONGODB_URI,
                serverSelectionTimeoutMS=15000,
                connectTimeoutMS=15000,
                socketTimeoutMS=15000,
                retryWrites=True,
            )
            temp_client[config.ONDEMAND_DB_NAME].drop_collection(
                _collection_name_for_session(normalized)
            )
            temp_client.close()
            return True
        except Exception:
            return False

    try:
        db = context["mongo_client"][config.ONDEMAND_DB_NAME]
        db.drop_collection(context["collection_name"])
    except Exception as e:
        logger.warning("Could not drop on-demand collection during cleanup: %s", e)
    finally:
        try:
            context["mongo_client"].close()
        except Exception:
            pass

    return True
