import os
import re
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.config import OLLAMA_MODEL, EMBEDDING_MODEL, NEO4J_URI, NEO4J_DB
from web.rag import driver, graphrag_list, graphrag_summary, initialize
from web.parser import cypher_capture, extract_nodes_from_answer


# ---------------------------------------------------------------------------
# Pydantic 모델
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    mode: str = "summary"


class QueryResponse(BaseModel):
    answer: str
    used_nodes: List[str]
    used_edges: List[str]
    retriever_used: str
    context: str = ""
    elapsed_sec: float = 0.0
    cypher_query: str = ""


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


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/graph")
async def get_graph():
    try:
        with driver.session(database=NEO4J_DB) as session:
            nodes_result = session.run("""
                MATCH (n)
                WHERE n:Article OR n:Category OR n:Content OR n:Media OR n:Author
                RETURN
                    elementId(n) as id,
                    labels(n)[0] as label,
                    CASE
                        WHEN n:Article THEN n.title
                        WHEN n:Category THEN n.name
                        WHEN n:Content THEN substring(n.chunk, 0, 50) + '...'
                        WHEN n:Media THEN n.name
                        WHEN n:Author THEN n.name
                        ELSE 'Unknown'
                    END as title,
                    properties(n) as properties
            """)
            nodes = [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "title": r["title"] or "No title",
                    "properties": {
                        k: v for k, v in dict(r["properties"]).items()
                        if k != "embedding"
                    },
                }
                for r in nodes_result
            ]

            edges_result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE (n:Article OR n:Category OR n:Content OR n:Media OR n:Author)
                  AND (m:Article OR m:Category OR m:Content OR m:Media OR m:Author)
                RETURN
                    elementId(r) as id,
                    elementId(n) as source,
                    elementId(m) as target,
                    type(r) as relationship
            """)
            edges = [
                {
                    "id": r["id"],
                    "source": r["source"],
                    "target": r["target"],
                    "relationship": r["relationship"],
                }
                for r in edges_result
            ]

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_graphrag(req: QueryRequest):
    try:
        from web.rag import graphrag_list as gl, graphrag_summary as gs

        if gs is None or gl is None:
            raise HTTPException(status_code=500, detail="GraphRAG not initialized")

        active_rag = gs if req.mode == "summary" else gl
        query_with_k = f"{req.question} (상위 {req.top_k}개만 답변)"
        cypher_capture.pop()

        start = time.time()
        result = active_rag.search(query_text=query_with_k, return_context=True)
        elapsed = round(time.time() - start, 1)
        captured_cypher = cypher_capture.pop()

        print(f"[Query] {req.question}")
        print(f"[Elapsed] {elapsed}s")
        if captured_cypher:
            print(f"[Cypher] {captured_cypher}")

        used_nodes, used_edges = [], []
        retriever_used = "unknown"
        context_str = ""
        cypher_query = ""

        try:
            if hasattr(result, "retriever_result") and result.retriever_result:
                rr = result.retriever_result
                if hasattr(rr, "metadata") and rr.metadata:
                    print(f"[Retriever metadata keys] {list(rr.metadata.keys())}")
                    cypher_query = rr.metadata.get("cypher", "")
                if hasattr(rr, "items") and rr.items:
                    for item in rr.items:
                        if hasattr(item, "metadata") and item.metadata:
                            if "tool" in item.metadata:
                                retriever_used = item.metadata["tool"]
                            if "cypher" in item.metadata and not cypher_query:
                                cypher_query = item.metadata["cypher"]
                        if hasattr(item, "content"):
                            context_str += str(item.content) + "\n\n"

            if not cypher_query and captured_cypher:
                cypher_query = captured_cypher
            if not cypher_query and context_str:
                m = re.search(
                    r"(MATCH\s.*?RETURN\s[^\n]+)", context_str,
                    re.IGNORECASE | re.DOTALL,
                )
                if m:
                    cypher_query = m.group(1).strip()

            answer_text = result.answer if hasattr(result, "answer") else str(result)
            used_nodes, used_edges = extract_nodes_from_answer(answer_text)

            for cat in ["정치", "경제", "사회", "생활/문화", "스포츠", "IT/과학"]:
                if cat in req.question:
                    used_nodes.append(f"Category_{cat}")
                    break

        except Exception as e:
            print(f"파싱 오류: {e}")

        return QueryResponse(
            answer=result.answer if hasattr(result, "answer") else str(result),
            used_nodes=list(set(used_nodes)),
            used_edges=list(set(used_edges)),
            retriever_used=retriever_used,
            context=context_str[:1000] if context_str else "",
            elapsed_sec=elapsed,
            cypher_query=cypher_query,
        )
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


@app.get("/health")
async def health_check():
    try:
        with driver.session(database=NEO4J_DB) as session:
            session.run("RETURN 1")
        return {"status": "healthy", "neo4j": "connected", "llm": OLLAMA_MODEL}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000)
