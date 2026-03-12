"""
LLM and Gemini client initialization.
"""
import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from google import genai

import config

logger = logging.getLogger(__name__)


def init_llm() -> Optional[ChatOpenAI]:
    """Initialize OpenAI LLM. Returns None on failure."""
    try:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=config.OPENAI_API_KEY,
            request_timeout=60,
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)
        return None


def init_gemini() -> Any:
    """Initialize Google Gemini client. Returns None if no API key or on failure."""
    try:
        client = (
            genai.Client(api_key=config.GOOGLE_API_KEY)
            if config.GOOGLE_API_KEY
            else None
        )
        if client:
            logger.info("Gemini client initialized successfully")
        else:
            logger.warning("Gemini client not initialized (no API key)")
        return client
    except Exception as e:
        logger.error("Failed to initialize Gemini client: %s", e)
        return None


def init_serper() -> Optional[GoogleSerperAPIWrapper]:
    """Initialize Google Serper wrapper for web search. Returns None if not configured."""
    if not config.SERPER_API_KEY:
        logger.info("Serper API key not configured")
        return None
    try:
        wrapper = GoogleSerperAPIWrapper(serper_api_key=config.SERPER_API_KEY)
        logger.info("GoogleSerperAPIWrapper initialized for web retrieval fallback")
        return wrapper
    except Exception as e:
        logger.warning("Could not initialize GoogleSerperAPIWrapper: %s", e)
        return None
