"""
MongoDB connection, vectorstore, retriever, and document name catalog.
"""
import logging
import ssl
import time
import traceback
from typing import List, Any, Optional, Tuple

import certifi
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

import config

logger = logging.getLogger(__name__)


def fetch_distinct_doc_names(mongo_client: Any) -> List[str]:
    """Fetch distinct document names from known metadata locations in MongoDB."""
    if not mongo_client:
        return []

    try:
        collection = mongo_client[config.DB_NAME][config.COLLECTION_NAME]
    except Exception as e:
        logger.warning("Could not access collection for doc_name lookup: %s", e)
        return []

    field_paths = (
        "metadata.doc_name",
        "metadata.metadata.doc_name",
        "doc_name",
    )
    names = set()
    for field_path in field_paths:
        try:
            for value in collection.distinct(field_path):
                if isinstance(value, str) and value.strip():
                    names.add(value.strip())
        except Exception as distinct_error:
            logger.debug(
                "Could not load distinct values for '%s': %s",
                field_path,
                distinct_error,
            )

    return sorted(names)


def init_mongo() -> Tuple[
    Optional[Any],
    Optional[Any],
    Optional[Any],
    List[str],
    Optional[OpenAIEmbeddings],
]:
    """
    Initialize MongoDB connection, embeddings, vectorstore, and retriever.
    Returns (mongo_client, vectorstore, retriever, known_doc_names, embedding_model).
    """
    start_time = time.time()
    mongo_client = None
    vectorstore = None
    retriever = None
    known_doc_names: List[str] = []
    embedding_model = None

    if config.MONGODB_URI:
        masked = (
            config.MONGODB_URI[:30] + "..." + config.MONGODB_URI[-20:]
            if len(config.MONGODB_URI) > 50
            else "URI_SET"
        )
        logger.info("MongoDB URI configured: %s", masked)
    else:
        logger.error("MONGODB_URI environment variable not set!")

    logger.info(
        "Target database: %s, collection: %s, index: %s",
        config.DB_NAME,
        config.COLLECTION_NAME,
        config.INDEX_NAME,
    )

    try:
        logger.info("Initializing embedding model...")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=config.OPENAI_API_KEY,
            request_timeout=45,
            max_retries=2,
            show_progress_bar=False,
        )
        logger.info("Embedding model initialized")

        logger.info("Connecting to MongoDB Atlas...")
        try:
            mongo_client = MongoClient(
                config.MONGODB_URI,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
            )
            mongo_client.admin.command('ping')
            logger.info("MongoDB connection successful")
        except Exception as conn_error:
            logger.error("Connection attempt failed: %s", conn_error)
            logger.error("OpenSSL version: %s", ssl.OPENSSL_VERSION)
            logger.error("certifi bundle: %s", certifi.where())
            raise

        collection = mongo_client[config.DB_NAME][config.COLLECTION_NAME]
        doc_count = collection.count_documents({})
        logger.info("Found %d documents in MongoDB collection", doc_count)

        try:
            known_doc_names = fetch_distinct_doc_names(mongo_client)
            logger.info("Loaded %d distinct doc_name values", len(known_doc_names))
        except Exception as distinct_error:
            logger.warning("Could not load distinct doc_name values: %s", distinct_error)
            known_doc_names = []

        if doc_count == 0:
            logger.warning(
                "MongoDB collection is empty - run build_mongo_vectorstore.py first"
            )
        else:
            logger.info("Initializing MongoDB Atlas Vector Search...")
            vectorstore = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embedding_model,
                index_name=config.INDEX_NAME,
                text_key="chunk_text",
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10},
            )
            load_time = time.time() - start_time
            logger.info("MongoDB vectorstore loaded successfully in %.2fs", load_time)
            logger.info("Retriever configured: similarity search with k=10")

    except Exception as e:
        logger.error("Failed to initialize MongoDB vectorstore: %s", e)
        logger.error("Error type: %s", type(e).__name__)
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        logger.error("TROUBLESHOOTING:")
        logger.error("1. Verify MONGODB_URI environment variable is set in Render")
        logger.error("2. Check MongoDB Atlas network access allows 0.0.0.0/0")
        logger.error("3. Verify database credentials are correct")
        logger.error("4. Ensure vector search index '%s' exists", config.INDEX_NAME)
        logger.error("5. Use Python linked with OpenSSL 1.1.1+ (recommended OpenSSL 3.x)")
        logger.error("=" * 60)
        vectorstore = None
        retriever = None
        known_doc_names = []
        if mongo_client:
            try:
                mongo_client.close()
            except Exception:
                pass
            mongo_client = None

    total_init_time = time.time() - start_time
    logger.info("Total initialization time: %.2fs", total_init_time)

    return (mongo_client, vectorstore, retriever, known_doc_names, embedding_model)
