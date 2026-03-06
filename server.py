from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import sys
import logging
import time
import re
from datetime import datetime
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import certifi
from dotenv import load_dotenv

# LangChain and OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient

# Google Gemini for image generation
from google import genai
from google.genai import types
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("Starting AI Medical Image Generator Server")
logger.info("="*60)

app = Flask(__name__)
CORS(app)

logger.info("Flask app and CORS initialized")

# Load environment variables from .env file if present
env_path = Path('.env')
if env_path.exists():
    logger.info(f"Loading environment variables from {env_path.resolve()}")
    load_dotenv(env_path)
else:
    logger.info(".env file not found; relying on system environment variables")

# Configure API keys
logger.info("Checking API keys...")
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')

if openai_api_key:
    logger.info(f"✓ OpenAI API key found (length: {len(openai_api_key)})")
else:
    logger.error("✗ OpenAI API key not found!")

if google_api_key:
    logger.info(f"✓ Google API key found (length: {len(google_api_key)})")
else:
    logger.error("✗ Google API key not found!")

# Image storage: in-memory for serverless (Vercel has read-only filesystem)
# Local dev can also write to static/images when writable
IMAGE_STORE = {}  # filename -> bytes
IMAGES_DIR = Path('static') / 'images'
IS_SERVERLESS = os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME')
if IS_SERVERLESS:
    logger.info("Serverless environment detected; using in-memory image store only")
else:
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Images directory ready: {IMAGES_DIR.resolve()}")
    except OSError as e:
        logger.warning(f"Images directory not writable ({e}); using in-memory store only")
        IS_SERVERLESS = True

# Initialize LLM
logger.info("Initializing LLM...")
try:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        api_key=openai_api_key,
        request_timeout=60  # 60 second timeout
    )
    logger.info("✓ LLM initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize LLM: {str(e)}")
    llm = None

# Initialize Google Gemini client for image generation
logger.info("Initializing Gemini client...")
try:
    gemini_client = genai.Client(api_key=google_api_key) if google_api_key else None
    if gemini_client:
        logger.info("✓ Gemini client initialized successfully")
    else:
        logger.warning("⚠ Gemini client not initialized (no API key)")
except Exception as e:
    logger.error(f"✗ Failed to initialize Gemini client: {str(e)}")
    gemini_client = None

# MongoDB Configuration
logger.info("Initializing MongoDB connection...")
start_time = time.time()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://db_user:db_user@cluster0.9a1tk8o.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = "medical_vector_db"
COLLECTION_NAME = "vector_chunks_embeds"
INDEX_NAME = "default"
NO_RAG_OPTION_VALUE = "NO_RAG"

# Log MongoDB connection details (masked)
if MONGODB_URI:
    masked_uri = MONGODB_URI[:30] + "..." + MONGODB_URI[-20:] if len(MONGODB_URI) > 50 else "URI_SET"
    logger.info(f"MongoDB URI configured: {masked_uri}")
else:
    logger.error("✗ MONGODB_URI environment variable not set!")
    
logger.info(f"Target database: {DB_NAME}, collection: {COLLECTION_NAME}, index: {INDEX_NAME}")

vectorstore = None
retriever = None
known_doc_names: List[str] = []
mongo_client = None


def _fetch_distinct_doc_names() -> List[str]:
    """Fetch distinct document names from known metadata locations in MongoDB."""
    if not mongo_client:
        return []

    try:
        collection = mongo_client[DB_NAME][COLLECTION_NAME]
    except Exception as e:
        logger.warning(f"⚠ Could not access collection for doc_name lookup: {str(e)}")
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
                f"Could not load distinct values for '{field_path}': {str(distinct_error)}"
            )

    return sorted(names)

try:
    # Initialize embeddings
    logger.info("Initializing embedding model...")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
        request_timeout=45,
        max_retries=2,
        show_progress_bar=False
    )
    logger.info("✓ Embedding model initialized")
    
    # Connect to MongoDB with SSL configuration for Render compatibility
    logger.info(f"Connecting to MongoDB Atlas...")
    try:
        # Try with tlsAllowInvalidCertificates for Render free tier
        mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            tlsAllowInvalidCertificates=True,  # Bypass cert validation for free tier
            retryWrites=True
        )
        
        # Test connection
        mongo_client.admin.command('ping')
        logger.info("✓ MongoDB connection successful")
    except Exception as conn_error:
        logger.error(f"Connection attempt failed: {str(conn_error)}")
        # Try fallback with ssl=false in connection string
        logger.info("Attempting fallback connection without SSL...")
        fallback_uri = MONGODB_URI
        if '?' in fallback_uri:
            fallback_uri += '&tls=false&tlsInsecure=true'
        else:
            fallback_uri += '?tls=false&tlsInsecure=true'
        
        mongo_client = MongoClient(
            fallback_uri,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        mongo_client.admin.command('ping')
        logger.info("✓ MongoDB connection successful (fallback mode)")
    
    # Get collection
    collection = mongo_client[DB_NAME][COLLECTION_NAME]
    doc_count = collection.count_documents({})
    logger.info(f"✓ Found {doc_count} documents in MongoDB collection")

    try:
        known_doc_names = _fetch_distinct_doc_names()
        logger.info(f"✓ Loaded {len(known_doc_names)} distinct doc_name values")
    except Exception as distinct_error:
        logger.warning(f"⚠ Could not load distinct doc_name values: {str(distinct_error)}")
        known_doc_names = []
    
    if doc_count == 0:
        logger.warning("⚠ MongoDB collection is empty - run build_mongo_vectorstore.py first")
    else:
        # Create vectorstore
        logger.info("Initializing MongoDB Atlas Vector Search...")
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_model,
            index_name=INDEX_NAME,
            text_key="chunk_text"
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        load_time = time.time() - start_time
        logger.info(f"✓ MongoDB vectorstore loaded successfully in {load_time:.2f}s")
        logger.info(f"✓ Retriever configured: similarity search with k=10")
        
except Exception as e:
    logger.error(f"✗ Failed to initialize MongoDB vectorstore: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(traceback.format_exc())
    logger.error("="*60)
    logger.error("TROUBLESHOOTING:")
    logger.error("1. Verify MONGODB_URI environment variable is set in Render")
    logger.error("2. Check MongoDB Atlas network access allows 0.0.0.0/0")
    logger.error("3. Verify database credentials are correct")
    logger.error("4. Ensure vector search index 'vector_index' exists")
    logger.error("="*60)
    vectorstore = None
    retriever = None
    known_doc_names = []
    if mongo_client:
        try:
            mongo_client.close()
        except:
            pass
        mongo_client = None

total_init_time = time.time() - start_time
logger.info(f"Total initialization time: {total_init_time:.2f}s")


@app.route('/')
def index():
    logger.info("Serving index.html")
    return send_from_directory('.', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    doc_count = 0
    if mongo_client:
        try:
            collection = mongo_client[DB_NAME][COLLECTION_NAME]
            doc_count = collection.count_documents({})
        except:
            doc_count = -1  # error getting count
    
    status = {
        'status': 'healthy',
        'openai_configured': openai_api_key is not None,
        'google_configured': google_api_key is not None,
        'mongodb_connected': mongo_client is not None,
        'vectorstore_loaded': vectorstore is not None,
        'retriever_ready': retriever is not None,
        'doc_name_catalog_ready': len(known_doc_names) > 0,
        'vectorstore_doc_count': doc_count,
        'gemini_client_ready': gemini_client is not None,
        'rag_available': retriever is not None and doc_count > 0
    }
    logger.info(f"Health check: {status}")
    return jsonify(status), 200


def _extract_doc_name_from_doc(doc: Document) -> Optional[str]:
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


def _sanitize_selected_doc_names(selected_doc_names: Any) -> List[str]:
    global known_doc_names
    if not isinstance(selected_doc_names, list):
        return []
    candidate_names = [str(name).strip() for name in selected_doc_names if isinstance(name, str) and str(name).strip()]
    if not candidate_names:
        return []

    if not known_doc_names:
        known_doc_names = _fetch_distinct_doc_names()

    if any(name not in set(known_doc_names) for name in candidate_names):
        latest_doc_names = _fetch_distinct_doc_names()
        if latest_doc_names:
            known_doc_names = latest_doc_names

    valid_names = set(known_doc_names)
    if not valid_names:
        return []
    return [name for name in candidate_names if name in valid_names]


def _is_no_rag_selected(selected_doc_names: Any) -> bool:
    if not isinstance(selected_doc_names, list):
        return False
    return any(
        isinstance(name, str) and name.strip().upper() == NO_RAG_OPTION_VALUE
        for name in selected_doc_names
    )


def _build_structured_retrieval_query(user_question: str) -> str:
    """Build a concise clinical retrieval query from the raw user question."""
    base_question = (user_question or "").strip()
    if not base_question:
        return ""

    if llm is None:
        return base_question

    extract_system = """Extract:
        1. Primary medical condition
        2. Mechanism keywords
        3. Clinical keywords

        Return short structured text only.
    """

    try:
        logger.info("Extracting structured retrieval query...")
        extract_response = llm.invoke([
            {"role": "system", "content": extract_system},
            {"role": "user", "content": base_question}
        ])
        structured_query = (extract_response.content or "").strip()
        if structured_query:
            logger.info(f"Structured retrieval query: {structured_query[:150]}...")
            return structured_query
        logger.warning("Structured retrieval query was empty; falling back to raw user question")
        return base_question
    except Exception as extract_error:
        logger.warning(f"Failed to extract structured retrieval query: {str(extract_error)}")
        return base_question


def _retrieve_docs_with_timeout(
    query: str,
    timeout_sec: int = 20,
    selected_doc_names: Optional[List[str]] = None,
    total_k: int = 10,
) -> List[Document]:
    """Run retrieval with timeout; supports explicit selected doc_name filtering."""
    if not retriever:
        raise ValueError("Retriever not available")

    def retrieve_docs():
        try:
            sanitized_doc_names = _sanitize_selected_doc_names(selected_doc_names)

            if sanitized_doc_names:
                if vectorstore is None:
                    raise ValueError("Vectorstore not available for doc_name filtered retrieval")

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
                    filtered_docs = vectorstore.similarity_search(
                        query=query,
                        k=per_doc_k,
                        pre_filter=pre_filter,
                    )
                    logger.info(
                        f"Doc filter retrieval: doc_name='{doc_name}', requested_k={per_doc_k}, returned={len(filtered_docs)}"
                    )
                    docs.extend(filtered_docs)

                return docs

            logger.info("Using base similarity retriever (no doc_name filters)")
            docs = retriever.invoke(query)
            if docs:
                return docs
            if vectorstore is not None:
                logger.warning("Retriever returned 0 docs; retrying direct similarity search")
                return vectorstore.similarity_search(query=query, k=max(1, int(total_k)))
            return docs
        except Exception as e:
            logger.error(f"Retriever invoke error: {str(e)}")
            raise
    with ThreadPoolExecutor(max_workers=5) as executor:
        future = executor.submit(retrieve_docs)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            logger.error(f"Document retrieval timed out after {timeout_sec} seconds")
            raise TimeoutError("Embedding API timeout - please retry")


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
        disable_rag = bool((data or {}).get('disable_rag')) or _is_no_rag_selected(raw_selected_doc_names)
        selected_doc_names = _sanitize_selected_doc_names(raw_selected_doc_names)
        if not search_query:
            return jsonify({'error': 'search_query is required and must be non-empty'}), 400
        if disable_rag:
            return jsonify({'error': 'Retrieval is disabled when NO RAG is selected'}), 400
        if not retriever:
            return jsonify({'error': 'RAG retriever not available'}), 503
        retrieval_query = _build_structured_retrieval_query(search_query)
        logger.info(f"Re-running retrieval with structured query: {retrieval_query[:120]}...")
        if selected_doc_names:
            logger.info(f"Using selected doc_name filters: {selected_doc_names}")
        docs = _retrieve_docs_with_timeout(
            retrieval_query,
            selected_doc_names=selected_doc_names,
            total_k=10,
        )
        if not docs:
            return jsonify({
                'search_query': retrieval_query,
                'chunks': [],
                'selected_doc_names': selected_doc_names,
                'message': 'No documents found for this query'
            }), 200
        chunks_payload = [
            {"content": doc.page_content, "metadata": getattr(doc, "metadata", {}) or {}}
            for doc in docs
        ]
        elapsed = time.time() - request_start
        logger.info(f"[/re-run-retrieval] Returned {len(chunks_payload)} chunks in {elapsed:.2f}s")
        return jsonify({
            'search_query': retrieval_query,
            'chunks': chunks_payload,
            'selected_doc_names': selected_doc_names,
        }), 200
    except TimeoutError as e:
        return jsonify({'error': str(e)}), 504
    except Exception as e:
        logger.error(f"[/re-run-retrieval] Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    """
    Generate a detailed image prompt using RAG-enhanced LLM based on system instruction
    """
    request_start = time.time()
    logger.info("="*50)
    logger.info("[/generate-prompt] Request received")
    
    try:
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        system_instruction = data.get('system_instruction', '')
        user_question = data.get('user_question', 'A serene landscape at sunset')
        raw_selected_doc_names = (data or {}).get('selected_doc_names')
        disable_rag = bool((data or {}).get('disable_rag')) or _is_no_rag_selected(raw_selected_doc_names)
        selected_doc_names = _sanitize_selected_doc_names(raw_selected_doc_names)
        
        if not system_instruction:
            logger.warning("Request missing system instruction")
            return jsonify({'error': 'System instruction is required'}), 400
        
        if not openai_api_key:
            logger.error("OpenAI API key not configured")
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'}), 500
        
        logger.info(f"Question: {user_question[:100]}...")
        structured_user_query = _build_structured_retrieval_query(user_question)
        
        # Optional: use provided search query and chunks (e.g. from frontend after user edit)
        override_search_query = (data or {}).get('search_query')
        override_chunks = (data or {}).get('chunks')
        use_provided_context = (
            override_search_query is not None
            and isinstance(override_chunks, list)
            and len(override_chunks) > 0
        )

        # Use RAG if retriever is available and NO RAG mode is not selected
        if retriever and not disable_rag:
            logger.info("Attempting RAG retrieval with timeout protection...")
            try:
                if use_provided_context:
                    # Use frontend-provided query and chunks; skip extract and retrieval
                    retrieval_query = override_search_query.strip() if isinstance(override_search_query, str) else str(override_search_query)
                    docs = [
                        Document(page_content=c.get("content", "") if isinstance(c, dict) else str(c), metadata=c.get("metadata", {}) if isinstance(c, dict) else {})
                        for c in override_chunks
                    ]
                    logger.info(f"Using provided context: query length {len(retrieval_query)}, {len(docs)} chunks")
                else:
                    # Retrieve documents with timeout protection (20 seconds for cold starts)
                    retrieval_query = structured_user_query
                    logger.info("Starting similarity retrieval from structured query...")
                    if selected_doc_names:
                        logger.info(f"Applying selected doc_name filters: {selected_doc_names}")
                    docs = _retrieve_docs_with_timeout(
                        retrieval_query,
                        timeout_sec=20,
                        selected_doc_names=selected_doc_names,
                        total_k=10,
                    )
                    if not docs:
                        logger.warning("⚠ No documents retrieved, using direct generation")
                        raise ValueError("No documents found")
                    logger.info(f"✓ Retrieved {len(docs)} documents in time")
                    doc_names = sorted({_extract_doc_name_from_doc(doc) for doc in docs if _extract_doc_name_from_doc(doc)})
                    if doc_names:
                        logger.info(f"Filtered doc_name values used for retrieval: {doc_names}")
                    if docs:
                        logger.info(f"Doc 1: {docs[0].page_content[:80]}...")
                
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Context assembled ({len(context)} chars)")
                
                # Build final prompt with RAG context
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
                response = llm.invoke([
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": construction_prompt}
                ])
                
                generated_prompt = response.content.strip()
                logger.info(f"✓ Generated prompt with RAG ({len(generated_prompt)} chars)")
                
                # Include RAG pipeline details for the frontend
                chunks_payload = [
                    {"content": doc.page_content, "metadata": getattr(doc, "metadata", {}) or {}}
                    for doc in docs
                ]
                return jsonify({
                    'prompt': generated_prompt,
                    'success': True,
                    'search_query': retrieval_query,
                    'selected_doc_names': selected_doc_names,
                    'disable_rag': False,
                    'chunks': chunks_payload,
                })
                
            except Exception as e:
                logger.warning(f"⚠ RAG failed ({str(e)}), falling back to direct generation")
                logger.warning(traceback.format_exc())
                # Fallback to direct generation without RAG
                response = llm.invoke([
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": (
                            "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                            f"Structured Clinical Query: {structured_user_query}\n"
                            f"Original User Question: {user_question}"
                        )
                    }
                ])
                generated_prompt = response.content.strip()
                logger.info(f"✓ Fallback prompt generated ({len(generated_prompt)} chars)")
        elif disable_rag:
            logger.info("NO RAG mode selected; generating prompt from structured query and original user question")
            response = llm.invoke([
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": (
                        "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                        f"Structured Clinical Query: {structured_user_query}\n"
                        f"Original User Question: {user_question}"
                    )
                }
            ])
            generated_prompt = response.content.strip()
            logger.info(f"✓ Direct prompt generated (NO RAG mode) ({len(generated_prompt)} chars)")
        else:
            logger.warning("⚠ RAG system not available, using direct generation without retrieval")
            # Direct generation without RAG as fallback
            response = llm.invoke([
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": (
                        "Create a detailed medical illustration prompt using this structured clinical query and the original request.\n\n"
                        f"Structured Clinical Query: {structured_user_query}\n"
                        f"Original User Question: {user_question}"
                    )
                }
            ])
            generated_prompt = response.content.strip()
            logger.info(f"✓ Direct prompt generated ({len(generated_prompt)} chars)")
        
        request_time = time.time() - request_start
        logger.info(f"[/generate-prompt] Success in {request_time:.2f}s")
        logger.info("="*50)
        
        # Non-RAG path: no search query or chunks
        return jsonify({
            'prompt': generated_prompt,
            'success': True,
            'search_query': structured_user_query,
            'selected_doc_names': selected_doc_names,
            'disable_rag': disable_rag,
            'chunks': [],
        })
    
    except Exception as e:
        request_time = time.time() - request_start
        logger.error(f"[/generate-prompt] Error after {request_time:.2f}s: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("="*50)
        return jsonify({'error': f'Error generating prompt: {str(e)}'}), 500


@app.route('/doc-names', methods=['GET'])
def get_doc_names():
    """Return distinct source document names available in the vector store."""
    global known_doc_names
    latest_names = _fetch_distinct_doc_names()
    if latest_names:
        known_doc_names = latest_names
    names = sorted([name for name in known_doc_names if isinstance(name, str) and name.strip()])
    return jsonify({'doc_names': names, 'count': len(names)}), 200


@app.route('/generate-image', methods=['POST'])
def generate_image():
    """
    Generate an image using Google Gemini based on the provided prompt
    """
    request_start = time.time()
    logger.info("="*50)
    logger.info("[/generate-image] Request received")
    
    try:
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        prompt = data.get('prompt', '')
        
        if not prompt:
            logger.warning("Request missing prompt")
            return jsonify({'error': 'Prompt is required'}), 400
        
        logger.info(f"Prompt length: {len(prompt)}")
        
        if not google_api_key:
            logger.error("Google API key not configured")
            return jsonify({'error': 'Google Generative AI API key not configured. Please set GOOGLE_GENERATIVE_AI_API_KEY environment variable.'}), 500
        
        if not gemini_client:
            logger.error("Gemini client not initialized")
            return jsonify({'error': 'Gemini client not initialized'}), 500
        
        logger.info(f"Generating image with prompt: {prompt[:100]}...")
        
        # Call Gemini API to generate image
        logger.info("Calling Gemini API...")
        api_start = time.time()
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt],
        )
        api_time = time.time() - api_start
        logger.info(f"Gemini API response received in {api_time:.2f}s")
        logger.info("Extracting image...")
        
        # Extract image from response; store in memory (serverless-safe) and optionally on disk
        image_saved = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'image_{timestamp}.png'

        try:
            for part in response.candidates[0].content.parts:
                if part.text:
                    logger.info(f"Text part found: {part.text[:100] if len(part.text) > 100 else part.text}")
                elif part.inline_data:
                    logger.info("Image data found, saving...")
                    image = Image.open(BytesIO(part.inline_data.data))
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    image_bytes = buf.getvalue()
                    IMAGE_STORE[filename] = image_bytes
                    if not IS_SERVERLESS:
                        try:
                            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                            (IMAGES_DIR / filename).write_bytes(image_bytes)
                            logger.info(f"Image saved to disk: {IMAGES_DIR / filename}")
                        except OSError:
                            pass
                    image_saved = True
                    logger.info(f"Image stored (in-memory); filename={filename}")
                    break
        except Exception as part_error:
            logger.error(f"Error extracting image from response: {str(part_error)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing Gemini response: {str(part_error)}'}), 500

        if not image_saved:
            logger.error("No image generated in response")
            return jsonify({'error': 'No image generated in response. Check server logs for details.'}), 500
        
        # Return the URL for serving (works for both local and deployed)
        image_url = f'{request.host_url}images/{filename}'
        
        request_time = time.time() - request_start
        logger.info(f"[/generate-image] Success in {request_time:.2f}s")
        logger.info(f"Image URL: {image_url}")
        logger.info("="*50)
        
        return jsonify({
            'image_url': image_url,
            'filename': filename,
            'success': True
        })
    
    except Exception as e:
        request_time = time.time() - request_start
        logger.error(f"[/generate-image] Error after {request_time:.2f}s: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("="*50)
        return jsonify({'error': f'Error generating image: {str(e)}'}), 500


@app.route('/images/<filename>')
def serve_image(filename):
    """
    Serve generated images from in-memory store or, if not found, from static dir (local).
    """
    logger.info(f"Serving image: {filename}")
    if filename in IMAGE_STORE:
        return send_file(
            BytesIO(IMAGE_STORE[filename]),
            mimetype='image/png',
            as_attachment=False,
            download_name=filename
        )
    if not IS_SERVERLESS and (IMAGES_DIR / filename).exists():
        return send_from_directory(IMAGES_DIR.resolve(), filename)
    return jsonify({'error': 'Image not found'}), 404


@app.route('/edit-image', methods=['POST'])
def edit_image():
    """
    Edit an existing image based on user-requested changes using Google Gemini.
    """
    request_start = time.time()
    logger.info("="*50)
    logger.info("[/edit-image] Request received")

    try:
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        filename = data.get('filename', '')
        changes = data.get('changes', '')

        if not filename:
            logger.warning("Request missing filename")
            return jsonify({'error': 'Filename is required'}), 400

        if not changes:
            logger.warning("Request missing changes")
            return jsonify({'error': 'Changes are required'}), 400

        logger.info(f"Filename: {filename}, Changes: {changes[:100]}...")

        if not google_api_key:
            logger.error("Google API key not configured")
            return jsonify({'error': 'Google Generative AI API key not configured. Please set GOOGLE_GENERATIVE_AI_API_KEY environment variable.'}), 500

        if not gemini_client:
            logger.error("Gemini client not initialized")
            return jsonify({'error': 'Gemini client not initialized'}), 500

        # Load image from in-memory store or from disk (local only)
        image = None
        if filename in IMAGE_STORE:
            image = Image.open(BytesIO(IMAGE_STORE[filename]))
            logger.info(f"Editing image from store: {filename}")
        elif not IS_SERVERLESS and (IMAGES_DIR / filename).exists():
            image = Image.open(IMAGES_DIR / filename)
            logger.info(f"Editing image from disk: {filename}")
        if image is None:
            logger.error(f"Image not found: {filename}")
            return jsonify({'error': f'File not found: {filename}'}), 404

        # Generate a new image based on the existing image and changes
        logger.info("Calling Gemini API for image editing...")
        api_start = time.time()
        prompt = f"Edit the following image based on the requested changes:\n\nChanges: {changes}"
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[prompt, image],
            )
            api_time = time.time() - api_start
            logger.info(f"Gemini API response received in {api_time:.2f}s")
            logger.debug(f"Gemini API response: {response}")
        except Exception as api_error:
            logger.error(f"Error calling Gemini API: {str(api_error)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error calling Gemini API: {str(api_error)}'}), 500

        # Extract the new image from the response; store in memory (serverless-safe)
        image_saved = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f'edited_{timestamp}.png'

        try:
            for part in response.candidates[0].content.parts:
                if part.text:
                    logger.info(f"Text part found: {part.text[:100] if len(part.text) > 100 else part.text}")
                elif part.inline_data:
                    logger.info("Edited image data found, storing...")
                    image = Image.open(BytesIO(part.inline_data.data))
                    buf = BytesIO()
                    image.save(buf, format='PNG')
                    IMAGE_STORE[new_filename] = buf.getvalue()
                    if not IS_SERVERLESS:
                        try:
                            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                            (IMAGES_DIR / new_filename).write_bytes(IMAGE_STORE[new_filename])
                            logger.info(f"Edited image saved to disk: {IMAGES_DIR / new_filename}")
                        except OSError:
                            pass
                    image_saved = True
                    logger.info(f"Edited image stored; filename={new_filename}")
                    break
        except Exception as part_error:
            logger.error(f"Error extracting edited image from response: {str(part_error)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing Gemini response: {str(part_error)}'}), 500

        if not image_saved:
            logger.error("No edited image generated in response")
            return jsonify({'error': 'No edited image generated in response. Check server logs for details.'}), 500

        # Return the URL for the new image
        image_url = f'{request.host_url}images/{new_filename}'

        request_time = time.time() - request_start
        logger.info(f"[/edit-image] Success in {request_time:.2f}s")
        logger.info(f"Edited Image URL: {image_url}")
        logger.info("="*50)

        return jsonify({
            'image_url': image_url,
            'filename': new_filename,
            'success': True
        })

    except Exception as e:
        request_time = time.time() - request_start
        logger.error(f"[/edit-image] Error after {request_time:.2f}s: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("="*50)
        return jsonify({'error': f'Error editing image: {str(e)}'}), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("AI Prompt to Image Generator Server")
    logger.info("="*60)
    logger.info("IMPORTANT: Make sure to create a .env file with your API keys, for example:")
    logger.info("  OPENAI_API_KEY=your-openai-api-key")
    logger.info("  GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-api-key")
    logger.info("The server will automatically load environment variables from .env if present.")
    
    # Use PORT environment variable for deployment, default to 5001 for local
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Server starting on http://localhost:{port}")
    logger.info(f"Open your browser and navigate to http://localhost:{port}")
    logger.info("="*60)
    
    # For production (Render, Railway, etc.), set host to 0.0.0.0
    app.run(debug=False, host='0.0.0.0', port=port)
