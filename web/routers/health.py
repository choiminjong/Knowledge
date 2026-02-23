from fastapi import APIRouter

from web.config import NEO4J_DB, OLLAMA_MODEL
from web.services.rag_service import driver

router = APIRouter()


@router.get("/health")
async def health_check():
    try:
        with driver.session(database=NEO4J_DB) as session:
            session.run("RETURN 1")
        return {"status": "healthy", "neo4j": "connected", "llm": OLLAMA_MODEL}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
