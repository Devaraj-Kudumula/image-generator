"""
LLM and Gemini client initialization.
"""
import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from google import genai

import config

logger = logging.getLogger(__name__)


def init_conversation_llm() -> Optional[ChatOpenAI]:
    """OpenAI chat model for the AI Chat page (long-form, full-history conversations)."""
    try:
        llm = ChatOpenAI(
            model=config.OPENAI_CONVERSATION_MODEL,
            temperature=config.OPENAI_CONVERSATION_TEMPERATURE,
            max_tokens=config.OPENAI_CONVERSATION_MAX_OUTPUT,
            api_key=config.OPENAI_API_KEY,
            request_timeout=config.OPENAI_CONVERSATION_REQUEST_TIMEOUT,
        )
        logger.info(
            "Conversation LLM initialized (model=%s, max_output=%s)",
            config.OPENAI_CONVERSATION_MODEL,
            config.OPENAI_CONVERSATION_MAX_OUTPUT,
        )
        return llm
    except Exception as e:
        logger.error("Failed to initialize conversation LLM: %s", e)
        return None


def init_rag_llm() -> Optional[ChatOpenAI]:
    """OpenAI chat model for RAG document Q&A."""
    try:
        llm = ChatOpenAI(
            model=config.OPENAI_RAG_MODEL,
            temperature=config.OPENAI_RAG_TEMPERATURE,
            max_tokens=config.OPENAI_RAG_MAX_OUTPUT,
            api_key=config.OPENAI_API_KEY,
        )
        logger.info("RAG LLM initialized (model=%s)", config.OPENAI_RAG_MODEL)
        return llm
    except Exception as e:
        logger.error("Failed to initialize RAG LLM: %s", e)
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
