import os
import re
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    Text2CypherRetriever,
    ToolsRetriever,
)
from neo4j_graphrag.generation import RagTemplate, GraphRAG


class CypherCaptureHandler(logging.Handler):
    """text2cypher 로거에서 생성된 Cypher 쿼리를 캡처합니다."""
    def __init__(self):
        super().__init__()
        self.last_cypher = ""

    def emit(self, record):
        msg = record.getMessage()
        if "Cypher query:" in msg:
            parts = msg.split("Cypher query:", 1)
            if len(parts) > 1:
                self.last_cypher = parts[1].strip()

    def pop(self) -> str:
        q = self.last_cypher
        self.last_cypher = ""
        return q


cypher_capture = CypherCaptureHandler()
t2c_logger = logging.getLogger("neo4j_graphrag.retrievers.text2cypher")
t2c_logger.addHandler(cypher_capture)
t2c_logger.setLevel(logging.DEBUG)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b-q4_K_M")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bona/bge-m3-korean")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j"))
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")
INDEX_NAME = "content_vector_index"

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
llm = OllamaLLM(
    model_name=OLLAMA_MODEL,
    model_params={
        "options": {"temperature": 0.1, "num_predict": 4096, "num_ctx": 8192, "repeat_penalty": 1.2, "repeat_last_n": 128},
        "think": False,
    },
)
embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_retriever = None
vector_cypher_retriever = None
text2cypher_retriever = None
tools_retriever = None
graphrag = None
graphrag_list = None
graphrag_summary = None
prompt_list = None
prompt_summary = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_retrievers()
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


def extract_all_field_values(text: str, field_name: str) -> List[str]:
    values = []
    pattern1 = rf"['\"]?{field_name}['\"]?\s*=\s*['\"]([^'\"]+)['\"]"
    values.extend(re.findall(pattern1, text))
    pattern2 = rf"['\"]?{field_name}['\"]?\s*:\s*['\"]([^'\"]+)['\"]"
    values.extend(re.findall(pattern2, text))
    if field_name == "article_id":
        values.extend(re.findall(r"(ART_\d{3}_\d{10})", text))
    if field_name == "content_id":
        values.extend(re.findall(r"(ART_\d{3}_\d{10}_chunk_\d+)", text))
    return list(set(values))


def is_valid_article_id(article_id: str) -> bool:
    return bool(re.match(r"^ART_\d{3}_\d{10}$", article_id))


def is_valid_category(category: str) -> bool:
    invalid = {"Unknown", "No title", "text2cypher_retriever",
               "vector_retriever", "vectorcypher_retriever"}
    if category in invalid or "retriever" in category.lower():
        return False
    return len(category) > 0


def extract_nodes_from_content(content: str) -> tuple[List[str], List[str]]:
    nodes, edges = [], []
    for aid in extract_all_field_values(content, "article_id"):
        if is_valid_article_id(aid):
            nodes.append(f"Article_{aid}")
    for cat in extract_all_field_values(content, "category_name"):
        if is_valid_category(cat):
            nodes.append(f"Category_{cat}")
    for cid in extract_all_field_values(content, "content_id"):
        if cid:
            nodes.append(f"Content_{cid}")
    if any("Article_" in n for n in nodes) and any("Category_" in n for n in nodes):
        edges.append("BELONGS_TO")
    if any("Article_" in n for n in nodes) and any("Content_" in n for n in nodes):
        edges.append("HAS_CHUNK")
    return list(set(nodes)), list(set(edges))


def extract_vectorcypher_nodes(content: str) -> tuple[List[str], List[str]]:
    nodes, edges = [], []
    related_idx = content.find("related_articles")
    main_content = content[:related_idx] if related_idx > 0 else content

    cid_match = re.search(r"content_id=\\?['\"]?(ART_\d{3}_\d{10}_chunk_\d+)", main_content)
    if cid_match:
        cid = cid_match.group(1)
        nodes.append(f"Content_{cid}")
        aid = re.sub(r"_chunk_\d+$", "", cid)
        if is_valid_article_id(aid):
            nodes.append(f"Article_{aid}")
            edges.append("HAS_CHUNK")

    if not any("Article_" in n for n in nodes):
        aid_match = re.search(r"article_id=\\?['\"]?(ART_\d{3}_\d{10})", main_content)
        if aid_match and is_valid_article_id(aid_match.group(1)):
            nodes.append(f"Article_{aid_match.group(1)}")

    cat_match = re.search(r"category_name=\\?['\"]?([가-힣/]+)", main_content)
    if cat_match and is_valid_category(cat_match.group(1).strip()):
        nodes.append(f"Category_{cat_match.group(1).strip()}")
        edges.append("BELONGS_TO")

    return list(set(nodes)), list(set(edges))


def extract_nodes_from_answer(answer: str) -> tuple[List[str], List[str]]:
    """답변 텍스트에서 URL을 추출하고, URL로부터 article_id를 역추적하여 노드를 결정합니다."""
    nodes, edges = [], []

    urls = re.findall(r"https?://n\.news\.naver\.com/mnews/article/(\d+)/(\d+)", answer)
    for media_code, article_num in urls:
        media_code_padded = media_code.zfill(3)
        article_num_padded = article_num.zfill(10)
        aid = f"ART_{media_code_padded}_{article_num_padded}"
        nodes.append(f"Article_{aid}")

    aid_direct = re.findall(r"(ART_\d{3}_\d{10})(?!_chunk)", answer)
    for aid in aid_direct:
        node = f"Article_{aid}"
        if node not in nodes:
            nodes.append(node)

    if nodes:
        edges.append("HAS_CHUNK")

    return list(set(nodes)), list(set(edges))


def get_neo4j_schema() -> str:
    with driver.session(database=NEO4J_DB) as session:
        node_info = session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName, propertyTypes
            RETURN nodeType, collect(propertyName) as properties
        """).data()
        patterns = session.run("""
            MATCH (n)-[r]->(m)
            RETURN DISTINCT labels(n)[0] as source, type(r) as rel, labels(m)[0] as target
            LIMIT 20
        """).data()
    schema = "=== Neo4j Schema ===\n\n노드 타입:\n"
    for n in node_info:
        schema += f"- {n['nodeType']}: {n['properties']}\n"
    schema += "\n관계 패턴:\n"
    for p in patterns:
        schema += f"- ({p['source']})-[:{p['rel']}]->({p['target']})\n"
    return schema


def initialize_retrievers():
    global vector_retriever, vector_cypher_retriever, text2cypher_retriever
    global tools_retriever, graphrag

    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder,
        neo4j_database=NEO4J_DB,
    )

    retrieval_query = """
    WITH node AS content, score
    MATCH (content)<-[:HAS_CHUNK]-(article:Article)
    OPTIONAL MATCH (article)-[:BELONGS_TO]->(category:Category)
    OPTIONAL MATCH (category)<-[:BELONGS_TO]-(related_article:Article)
    WHERE related_article <> article
    RETURN
        content.content_id AS content_id,
        content.chunk AS chunk,
        article.article_id AS article_id,
        article.title AS article_title,
        article.url AS article_url,
        article.published_date AS article_date,
        category.name AS category_name,
        score AS similarity_score,
        collect(DISTINCT {
            article_id: related_article.article_id,
            title: related_article.title,
            url: related_article.url,
            published_date: related_article.published_date
        })[0..5] AS related_articles
    """
    vector_cypher_retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        retrieval_query=retrieval_query,
        embedder=embedder,
        neo4j_database=NEO4J_DB,
    )

    neo4j_schema = get_neo4j_schema()
    examples = [
        'USER INPUT: 경제 분야의 최신 뉴스\nCYPHER QUERY:\nMATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "경제"})\nRETURN a.article_id, a.title, a.url, a.published_date\nORDER BY a.published_date DESC LIMIT 10',
        'USER INPUT: 카테고리별 기사 개수\nCYPHER QUERY:\nMATCH (a:Article)-[:BELONGS_TO]->(c:Category)\nRETURN c.name as category, count(a) as article_count\nORDER BY article_count DESC',
        'USER INPUT: 매일경제에서 나온 최신 뉴스 3개\nCYPHER QUERY:\nMATCH (m:Media {name: "매일경제"})-[:PUBLISHED]->(a:Article)\nRETURN a.article_id, a.title, a.url, a.published_date\nORDER BY a.published_date DESC LIMIT 3',
    ]
    text2cypher_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
        neo4j_database=NEO4J_DB,
    )

    vector_tool = vector_retriever.convert_to_tool(
        name="vector_retriever",
        description=(
            "뉴스 기사 본문의 의미를 기반으로 유사한 내용을 검색한다. "
            "특정 주제, 인물, 사건에 대한 기사를 찾을 때 사용한다. "
            "카테고리 목록, 기사 수 집계, 언론사별 조회에는 사용하지 않는다."
        ),
    )
    vector_cypher_tool = vector_cypher_retriever.convert_to_tool(
        name="vectorcypher_retriever",
        description=(
            "의미 기반 검색 결과에 기사의 상세정보(제목, URL, 날짜)와 "
            "같은 카테고리의 관련 기사를 함께 반환한다. "
            "특정 주제의 기사를 찾되 관련 기사도 함께 필요할 때 사용한다. "
            "카테고리 목록, 기사 수 집계, 언론사별 조회에는 사용하지 않는다."
        ),
    )
    text2cypher_tool = text2cypher_retriever.convert_to_tool(
        name="text2cypher_retriever",
        description=(
            "카테고리별 기사 목록, 언론사별 기사 수, 최신 기사 N개 등 "
            "구조적 조건으로 검색한다. "
            "'카테고리', '최신', '목록', '개수', '언론사별' 같은 "
            "조건이 포함된 질문에 반드시 이 도구를 사용한다."
        ),
    )

    tools_retriever = ToolsRetriever(
        driver=driver,
        llm=llm,
        tools=[vector_tool, vector_cypher_tool, text2cypher_tool],
    )

    global prompt_list, prompt_summary, graphrag_list, graphrag_summary

    prompt_list = RagTemplate(
        template="""당신은 Neo4j 그래프 데이터베이스에 저장된 뉴스 기사만을 검색하여 답변하는 어시스턴트입니다.

질문: {query_text}

검색된 문서 정보:
{context}

반드시 지켜야 할 규칙:
1. 오직 위의 검색 결과(Context)에 포함된 내용만 사용하여 답변하세요.
2. 검색 결과가 비어있거나 질문과 관련 없는 경우, "해당 내용의 기사가 데이터베이스에 없습니다."라고만 답하세요.
3. 절대로 검색 결과에 없는 내용을 추측하거나 자체 지식으로 답변하지 마세요.
4. 같은 URL의 기사는 중복 제거하여 한 번만 표시하세요.
5. 질문에 "상위 N개만 답변"이라는 지시가 있으면, 중복 제거 후 N개의 기사만 답변하세요.
6. 반드시 아래 형식으로만 답변하세요:

1. 기사제목
   - URL: https://...
   - 발행일: YYYY-MM-DD HH:MM:SS

답변:""",
        expected_inputs=["context", "query_text"],
    )

    prompt_summary = RagTemplate(
        template="""당신은 Neo4j 그래프 데이터베이스에 저장된 뉴스 기사를 분석하여 주제별 요약 브리핑을 제공하는 AI 어시스턴트입니다.

질문: {query_text}

검색된 기사 정보:
{context}

반드시 지켜야 할 규칙:
1. 오직 위의 검색 결과에 포함된 내용만을 기반으로 요약하세요.
2. 검색 결과가 비어있거나 질문과 관련 없는 경우, "관련 기사가 데이터베이스에 없습니다."라고만 답하세요.
3. 절대로 검색 결과에 없는 내용을 추측하거나 자체 지식으로 답변하지 마세요.
4. 반드시 아래 형식으로 답변하세요:

## [주제] 요약

[여러 기사의 핵심 내용을 종합하여 3~5문장으로 요약]

### 주요 포인트
- [핵심 사항 1]
- [핵심 사항 2]
- [핵심 사항 3]

### 참고 기사
1. 기사제목 (발행일)
   URL: https://...

답변:""",
        expected_inputs=["context", "query_text"],
    )

    graphrag_list = GraphRAG(
        llm=llm,
        retriever=tools_retriever,
        prompt_template=prompt_list,
    )
    graphrag_summary = GraphRAG(
        llm=llm,
        retriever=tools_retriever,
        prompt_template=prompt_summary,
    )
    graphrag = graphrag_summary


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


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
                {"id": r["id"], "label": r["label"], "title": r["title"] or "No title",
                 "properties": {k: v for k, v in dict(r["properties"]).items() if k != "embedding"}}
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
                {"id": r["id"], "source": r["source"], "target": r["target"],
                 "relationship": r["relationship"]}
                for r in edges_result
            ]

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_graphrag(req: QueryRequest):
    try:
        import time
        if graphrag_summary is None or graphrag_list is None:
            raise HTTPException(status_code=500, detail="GraphRAG not initialized")

        active_rag = graphrag_summary if req.mode == "summary" else graphrag_list
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
                m = re.search(r"(MATCH\s.*?RETURN\s[^\n]+)", context_str, re.IGNORECASE | re.DOTALL)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
