# API 레퍼런스

`web/app.py` FastAPI 서버의 엔드포인트 상세 명세입니다.

## 서버 실행

```bash
uv run python web/app.py
# http://localhost:8000
```

## 엔드포인트

### GET /

프론트엔드 HTML 페이지를 반환합니다.

### GET /health

서버 상태를 확인합니다.

**Response:**

```json
{
  "status": "healthy",
  "neo4j": "connected",
  "llm": "qwen3:8b-q4_K_M"
}
```

**curl 예시:**

```bash
curl http://localhost:8000/health
```

---

### GET /graph

Neo4j의 전체 노드와 엣지를 JSON으로 반환합니다. 프론트엔드 그래프 시각화에 사용됩니다.

**Response:**

```json
{
  "nodes": [
    {
      "id": "4:abc:123",
      "label": "Article",
      "title": "[속보] 민주당 44.8%...",
      "properties": {
        "article_id": "ART_655_0000029796",
        "title": "[속보] 민주당 44.8%...",
        "url": "https://n.news.naver.com/...",
        "published_date": "2026-02-16 12:00:10"
      }
    }
  ],
  "edges": [
    {
      "id": "5:abc:456",
      "source": "4:abc:123",
      "target": "4:abc:789",
      "relationship": "BELONGS_TO"
    }
  ]
}
```

**curl 예시:**

```bash
curl http://localhost:8000/graph
```

---

### POST /query

GraphRAG 검색을 수행합니다.

**Request Body:**

```json
{
  "question": "정치 카테고리의 최신 뉴스 알려줘",
  "top_k": 3,
  "mode": "summary"
}
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| question | string | (필수) | 자연어 질문 |
| top_k | int | 3 | 답변에 포함할 기사 수 |
| mode | string | "summary" | `"summary"` (요약 브리핑) 또는 `"list"` (기사 목록) |

**Response:**

```json
{
  "answer": "## 정치 카테고리 요약\n\n최근 정치 분야에서...",
  "used_nodes": ["Article_ART_655_0000029796", "Category_정치"],
  "used_edges": ["HAS_CHUNK"],
  "retriever_used": "text2cypher_retriever",
  "context": "검색된 원본 텍스트...",
  "elapsed_sec": 45.2,
  "cypher_query": "MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: '정치'}) RETURN ..."
}
```

| 필드 | 설명 |
|------|------|
| answer | LLM이 생성한 답변 (Markdown) |
| used_nodes | 하이라이트할 노드 ID 목록 |
| used_edges | 하이라이트할 관계 타입 목록 |
| retriever_used | 사용된 Retriever 이름 |
| context | 검색된 원본 컨텍스트 (최대 1000자) |
| elapsed_sec | 소요 시간 (초) |
| cypher_query | Text2Cypher 사용 시 실행된 Cypher 쿼리 |

**curl 예시:**

```bash
# 요약 모드
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "트럼프 관련 뉴스", "top_k": 3, "mode": "summary"}'

# 목록 모드
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "카테고리별 기사 개수", "top_k": 10, "mode": "list"}'
```

## 에러 응답

모든 엔드포인트는 에러 시 HTTP 500을 반환합니다:

```json
{
  "detail": "에러 메시지\n스택 트레이스..."
}
```
