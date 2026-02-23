import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b-q4_K_M")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bona/bge-m3-korean")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

INDEX_NAME = "content_vector_index"
