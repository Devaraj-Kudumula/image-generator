"""
Shared application state populated at startup. Used by routes and services.
"""
# No imports from app packages to avoid circular dependencies.


class AppState:
    """Holds runtime state: clients, vectorstore, retriever, etc."""

    def __init__(self):
        self.openai_api_key = None
        self.google_api_key = None
        self.llm = None
        self.gemini_client = None
        self.mongo_client = None
        self.vectorstore = None
        self.retriever = None
        self.embedding_model = None
        self.known_doc_names = []
        self.google_serper_wrapper = None
        self.ondemand_sessions = {}


state = AppState()
