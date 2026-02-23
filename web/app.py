import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.config import OLLAMA_MODEL, EMBEDDING_MODEL, NEO4J_URI, NEO4J_DB
from web.services.rag_service import driver, initialize
from web.routers import graph, query, health


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize()
        print(f"LLM:       {OLLAMA_MODEL}")
        print(f"Embedding: {EMBEDDING_MODEL}")
        print(f"Neo4j:     {NEO4J_URI} (db: {NEO4J_DB})")
        print("Retriever 초기화 완료")
    except Exception as e:
        print(f"Retriever 초기화 실패: {e}")
    yield
    driver.close()


app = FastAPI(title="GraphRAG Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(graph.router)
app.include_router(query.router)
app.include_router(health.router)


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
