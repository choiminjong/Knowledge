# GraphBuilder_RecursiveChunk 가이드 — GraphBuilder_RecursiveChunk.ipynb

전체 내용을 Notion 페이지에 복사해 넣으세요.

---

## 개요

**목적**: DataScrapping 결과 Excel을 Neo4j 그래프 DB에 적재. GraphBuilder와 동일하되, **Recursive 구분자 기반 청킹** 사용.

| 항목 | GraphBuilder | 이 노트북 |
|------|--------------|----------|
| 청킹 | 고정 500자 | recursive_chunk_text |
| 구분자 | 없음 | `\n\n` → `\n` → `. ` → ` ` 순 |
| LangChain | 사용 가능 | 사용 안 함 (순수 Python) |

---

## 사전 준비

- **입력**: `Articles_*.xlsx` (DataScrapping 결과물, 프로젝트 루트)
- **Neo4j**: Desktop 또는 Aura, `bolt://127.0.0.1:7687`
- **.env**: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DB
- **패키지**: pandas, neo4j, python-dotenv, openpyxl

---

## 청킹 방식

**구분자 우선순위**: `\n\n`(문단) → `\n`(줄) → `. `(문장) → `。` → `, ` → ` `(공백)

고정 청킹은 500자로 단순 분할, Recursive는 위 구분자로 문맥 경계를 존중하며 분할.

---

## 실행 순서 (위→아래)

1. **데이터 로드** — glob으로 최신 `Articles_*.xlsx` 선택 → `pd.read_excel()`
2. **Neo4j 연결** — dotenv, driver, verify_connectivity
3. **청킹 함수 정의** — chunk_text(비교용), recursive_chunk_text(사용)
4. **청킹 비교** (선택) — 고정 vs Recursive 결과 비교
5. **DB 초기화** — DETACH DELETE, Unique 제약조건
6. **노드/관계 함수** — Article, Content, Media, Category, Author, HAS_CHUNK, PUBLISHED, BELONGS_TO, WROTE
7. **그래프 빌드** — build_graph_from_dataframe (recursive_chunk_text 사용)
8. **검증** — 노드/관계 수, 카테고리별 기사 수, 샘플 조회
9. **정리** — driver.close()

---

## 그래프 스키마

**노드**: Article, Content, Media, Category, Author  
**관계**: HAS_CHUNK(Article→Content), PUBLISHED(Media→Article), BELONGS_TO(Article→Category), WROTE(Author→Article)

```
Media -PUBLISHED-> Article -HAS_CHUNK-> Content
         Article -BELONGS_TO-> Category
Author -WROTE-> Article
```

**Cypher 예시**

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
MATCH (a:Article)-[:BELONGS_TO]->(cat:Category)
RETURN cat.name, count(a) ORDER BY count(a) DESC
```

---

## 검증

- 노드 수: Content ~239, Article 60, Author ~58, Media ~38, Category 6 (60건 기준)
- 관계: HAS_CHUNK ~239, BELONGS_TO 60, PUBLISHED 60, WROTE 60

---

## 트러블슈팅

**Articles_*.xlsx 없음** → DataScrapping 먼저 실행

**Neo4j 연결 실패** → 서버 실행, .env 확인, 방화벽 확인

**clear_database** → 해당 DB 전체 삭제됨, 다른 프로젝트와 DB 분리 권장

**Content 0개** → df['content'] 값 확인, recursive_chunk_text 결과 print

**느림** → 소량으로 테스트 후 확대, Neo4j 메모리/설정 점검
