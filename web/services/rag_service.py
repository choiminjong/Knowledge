import neo4j
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    Text2CypherRetriever,
    ToolsRetriever,
)
from neo4j_graphrag.generation import RagTemplate, GraphRAG

from web.config import (
    OLLAMA_MODEL, EMBEDDING_MODEL,
    NEO4J_URI, NEO4J_AUTH, NEO4J_DB,
    INDEX_NAME,
)


# ---------------------------------------------------------------------------
# Neo4j / LLM / Embedder
# ---------------------------------------------------------------------------

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

llm = OllamaLLM(
    model_name=OLLAMA_MODEL,
    model_params={
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
            "num_ctx": 8192,
            "repeat_penalty": 1.2,
            "repeat_last_n": 128,
        },
        "think": False,
    },
)

embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Retriever / GraphRAG (앱 시작 시 초기화)
# ---------------------------------------------------------------------------

tools_retriever = None
graphrag_list = None
graphrag_summary = None


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


def initialize():
    """Retriever 와 GraphRAG 인스턴스를 생성합니다. 앱 시작(lifespan)에서 호출됩니다."""
    global tools_retriever, graphrag_list, graphrag_summary

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
        'USER INPUT: 경제 분야의 최신 뉴스\nCYPHER QUERY:\n'
        'MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "경제"})\n'
        'RETURN a.article_id, a.title, a.url, a.published_date\n'
        'ORDER BY a.published_date DESC LIMIT 10',
        'USER INPUT: 카테고리별 기사 개수\nCYPHER QUERY:\n'
        'MATCH (a:Article)-[:BELONGS_TO]->(c:Category)\n'
        'RETURN c.name as category, count(a) as article_count\n'
        'ORDER BY article_count DESC',
        'USER INPUT: 매일경제에서 나온 최신 뉴스 3개\nCYPHER QUERY:\n'
        'MATCH (m:Media {name: "매일경제"})-[:PUBLISHED]->(a:Article)\n'
        'RETURN a.article_id, a.title, a.url, a.published_date\n'
        'ORDER BY a.published_date DESC LIMIT 3',
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
