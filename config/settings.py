from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Data settings
PRODUCT_CATEGORIES = [
    "Credit card", 
    "Personal loan", 
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers"
]

# Text processing settings
TEXT_CLEANING_CONFIG = {
    "lowercase": True,
    "remove_special_chars": True,
    "remove_boilerplate": True
}

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval settings
TOP_K = 5
SIMILARITY_METRIC = "cosine"

# Generation settings
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, 
state that you don't have enough information. 

Context: {context}

Question: {question}

Answer:"""
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7