from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import logging
import time
from datetime import datetime
import requests
from pathlib import Path
from typing import List
from io import BytesIO
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import certifi

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

# Create directory for generated images
logger.info("Creating images directory...")
IMAGES_DIR = Path('generated_images')
IMAGES_DIR.mkdir(exist_ok=True)
logger.info(f"✓ Images directory ready: {IMAGES_DIR}")

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
DB_NAME = "medical_rag"
COLLECTION_NAME = "medical_documents"
INDEX_NAME = "vector_index"

# Log MongoDB connection details (masked)
if MONGODB_URI:
    masked_uri = MONGODB_URI[:30] + "..." + MONGODB_URI[-20:] if len(MONGODB_URI) > 50 else "URI_SET"
    logger.info(f"MongoDB URI configured: {masked_uri}")
else:
    logger.error("✗ MONGODB_URI environment variable not set!")
    
logger.info(f"Target database: {DB_NAME}, collection: {COLLECTION_NAME}, index: {INDEX_NAME}")

vectorstore = None
retriever = None
mongo_client = None

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
    
    if doc_count == 0:
        logger.warning("⚠ MongoDB collection is empty - run build_mongo_vectorstore.py first")
    else:
        # Create vectorstore
        logger.info("Initializing MongoDB Atlas Vector Search...")
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_model,
            index_name=INDEX_NAME
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        load_time = time.time() - start_time
        logger.info(f"✓ MongoDB vectorstore loaded successfully in {load_time:.2f}s")
        logger.info(f"✓ Retriever configured: similarity search with k=3")
        
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
        'vectorstore_doc_count': doc_count,
        'gemini_client_ready': gemini_client is not None,
        'rag_available': retriever is not None and doc_count > 0
    }
    logger.info(f"Health check: {status}")
    return jsonify(status), 200


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
        
        if not system_instruction:
            logger.warning("Request missing system instruction")
            return jsonify({'error': 'System instruction is required'}), 400
        
        if not openai_api_key:
            logger.error("OpenAI API key not configured")
            return jsonify({'error': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'}), 500
        
        logger.info(f"Question: {user_question[:100]}...")
        
        # Use RAG if vectorstore is available
        if retriever:
            logger.info("Attempting RAG retrieval with timeout protection...")
            try:
                # Extract retrieval query
                extract_system = """Extract:
                    1. Primary medical condition
                    2. Mechanism keywords
                    3. Clinical keywords
                    
                    Return short structured text only.
                """
                
                logger.info("Extracting retrieval query...")
                extract_response = llm.invoke([
                    {"role": "system", "content": extract_system},
                    {"role": "user", "content": user_question}
                ])
                
                retrieval_query = extract_response.content
                logger.info(f"Retrieval query: {retrieval_query[:150]}...")
                
                # Retrieve documents with timeout protection (20 seconds for cold starts)
                logger.info("Starting document retrieval (k=3 for speed)...")
                
                def retrieve_docs():
                    try:
                        return retriever.invoke(retrieval_query)
                    except Exception as e:
                        logger.error(f"Retriever invoke error: {str(e)}")
                        raise
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(retrieve_docs)
                    try:
                        docs = future.result(timeout=20)  # 20 second timeout for cold starts
                        logger.info(f"✓ Retrieved {len(docs)} documents in time")
                        
                        if docs:
                            logger.info(f"Doc 1: {docs[0].page_content[:80]}...")
                        else:
                            logger.warning("⚠ No documents retrieved, using direct generation")
                            raise ValueError("No documents found")
                    except FuturesTimeoutError:
                        logger.error("✗ Document retrieval timed out after 20 seconds")
                        logger.error("This may be due to cold start on free tier - retrying will be faster")
                        raise TimeoutError("Embedding API timeout - please retry")
                
                context = "\n\n".join([doc.page_content for doc in docs])
                logger.info(f"Context assembled ({len(context)} chars)")
                
                # Build final prompt with RAG context
                construction_prompt = f"""
                    Retrieved High-Yield Medical Context:
                    {context}
                    
                    User Request:
                    {user_question}
                    
                    Return a complete structured image generation prompt following the system instruction guidelines.
                """
                
                logger.info("Generating prompt with RAG context...")
                response = llm.invoke([
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": construction_prompt}
                ])
                
                generated_prompt = response.content.strip()
                logger.info(f"✓ Generated prompt with RAG ({len(generated_prompt)} chars)")
                
            except Exception as e:
                logger.warning(f"⚠ RAG failed ({str(e)}), falling back to direct generation")
                logger.warning(traceback.format_exc())
                # Fallback to direct generation without RAG
                response = llm.invoke([
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Create a detailed medical illustration prompt for: {user_question}"}
                ])
                generated_prompt = response.content.strip()
                logger.info(f"✓ Fallback prompt generated ({len(generated_prompt)} chars)")
        else:
            logger.warning("⚠ RAG system not available, using direct generation without retrieval")
            # Direct generation without RAG as fallback
            response = llm.invoke([
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Create a detailed medical illustration prompt for: {user_question}"}
            ])
            generated_prompt = response.content.strip()
            logger.info(f"✓ Direct prompt generated ({len(generated_prompt)} chars)")
        
        request_time = time.time() - request_start
        logger.info(f"[/generate-prompt] Success in {request_time:.2f}s")
        logger.info("="*50)
        
        return jsonify({
            'prompt': generated_prompt,
            'success': True
        })
    
    except Exception as e:
        request_time = time.time() - request_start
        logger.error(f"[/generate-prompt] Error after {request_time:.2f}s: {str(e)}")
        logger.error(traceback.format_exc())
        logger.info("="*50)
        return jsonify({'error': f'Error generating prompt: {str(e)}'}), 500


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
        
        # Extract image from response (matching working notebook code)
        image_saved = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'image_{timestamp}.png'
        filepath = IMAGES_DIR / filename
        
        try:
            # Access via candidates[0].content.parts (correct path based on working code)
            for part in response.candidates[0].content.parts:
                if part.text:
                    logger.info(f"Text part found: {part.text[:100] if len(part.text) > 100 else part.text}")
                
                elif part.inline_data:
                    logger.info("Image data found, saving...")
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(str(filepath))
                    image_saved = True
                    logger.info(f"Image saved successfully to {filepath}")
                    break
        except Exception as part_error:
            logger.error(f"Error extracting image from response: {str(part_error)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing Gemini response: {str(part_error)}'}), 500
        
        if not image_saved:
            logger.error("No image generated in response")
            return jsonify({'error': 'No image generated in response. Check server logs for details.'}), 500
        
        # Return the URL for serving (works for both local and deployed)
        # Use request.host_url to get the current domain
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
    Serve generated images
    """
    logger.info(f"Serving image: {filename}")
    return send_from_directory(IMAGES_DIR.resolve(), filename)


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

        # Check if the image exists
        filepath = IMAGES_DIR / filename
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return jsonify({'error': f'File not found: {filename}'}), 404

        logger.info(f"Editing image: {filepath}")

        # # Open the existing image
        # with open(filepath, 'rb') as image_file:
        #     image_data = image_file.read()

        # # Wrap the image data in the appropriate structure
        # from gemini_sdk.models import File
        # wrapped_image_data = File(data=image_data, filename=filename)

        image = Image.open(filepath)

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

        # Extract the new image from the response
        image_saved = False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f'edited_{timestamp}.png'
        new_filepath = IMAGES_DIR / new_filename

        # Ensure the images directory exists
        if not IMAGES_DIR.exists():
            logger.warning(f"Images directory does not exist. Creating: {IMAGES_DIR}")
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        # Log the full path where the image will be saved
        logger.info(f"Attempting to save image to path: {new_filepath}")

        try:

            for part in response.candidates[0].content.parts:
                if part.text:
                    logger.info(f"Text part found: {part.text[:100] if len(part.text) > 100 else part.text}")
                
                elif part.inline_data:
                    logger.info("Image data found, saving...")
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(str(filepath))
                    image_saved = True
                    logger.info(f"Image saved successfully to {filepath}")
                    break
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    logger.info("Edited image data found, saving...")
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        # image.verify()  # Verify image integrity
                        # image = Image.open(BytesIO(part.inline_data))  # Reopen after verification
                        image.save(str(new_filepath))
                        image_saved = True
                        logger.info(f"Edited image saved successfully to {new_filepath}")
                        break
                    except Exception as save_error:
                        logger.error(f"Error saving image to path {new_filepath}: {str(save_error)}")
                        logger.error(traceback.format_exc())
                        return jsonify({'error': f'Error saving image: {str(save_error)}'}), 500
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
    logger.info("IMPORTANT: Make sure to set your OpenAI API key:")
    logger.info("  export OPENAI_API_KEY='your-api-key-here'")
    
    # Use PORT environment variable for deployment, default to 5001 for local
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Server starting on http://localhost:{port}")
    logger.info(f"Open your browser and navigate to http://localhost:{port}")
    logger.info("="*60)
    
    # For production (Render, Railway, etc.), set host to 0.0.0.0
    app.run(debug=False, host='0.0.0.0', port=port)
