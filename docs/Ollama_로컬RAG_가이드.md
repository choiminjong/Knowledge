# Ollama 로컬 GraphRAG 가이드

Ollama를 사용하여 **완전 로컬 환경**에서 뉴스 기사 GraphRAG 검색 시스템을 실행하는 가이드입니다.

## 구성 요소

| 구분 | 모델 | 용도 |
|------|------|------|
| **LLM** | `qwen3:8b-q4_K_M` | 한국어 답변 생성, Tool Calling, Text2Cypher |
| **Embedding** | `bona/bge-m3-korean` | 한국어 텍스트 → 1024차원 벡터 변환 |
| **Graph DB** | Neo4j | 뉴스 기사 그래프 저장/검색 |

---

## 1. Ollama 설치

### Windows

1. https://ollama.ai 에서 Windows용 설치 파일 다운로드
2. 설치 파일 실행 (관리 정책으로 차단 시 파일 속성 → 차단 해제)
3. 설치 후 시스템 트레이에 Ollama 아이콘 확인

### 설치 확인

```powershell
ollama --version
```

---

## 2. 모델 다운로드

```powershell
# LLM (Qwen3 8B - 한국어 지원 + Tool Calling 안정, Q4 양자화)
ollama pull qwen3:8b-q4_K_M

# Embedding (BGE-M3 한국어)
ollama pull bona/bge-m3-korean
```

다운로드 확인:

```powershell
ollama list
```

출력 예시:

```
NAME                                     SIZE
qwen3:8b-q4_K_M                         ~5.2GB
bona/bge-m3-korean                       ~1.2GB
```

---

## 3. GPU 성능 최적화 (선택)

RTX 4050 (6GB VRAM) 등 NVIDIA GPU가 있으면 환경변수를 설정합니다.

### 환경변수 설정 (PowerShell)

```powershell
# GPU 레이어 전부 사용
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", "-1", "User")

# CPU 스레드 수 (본인 코어 수로 변경)
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_THREADS", "8", "User")
```

CPU 코어 수 확인:

```powershell
(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
```

설정 후 **Ollama를 재시작**해야 적용됩니다 (트레이 아이콘 우클릭 → Quit → 다시 실행).

---

## 4. 동작 테스트

### 4-1. CLI 테스트

```powershell
# LLM 테스트 (한국어 응답 확인)
ollama run qwen3:8b-q4_K_M "안녕하세요, 테스트입니다. 한 줄로 답해주세요."
```

한국어로 자연스러운 응답이 오면 성공입니다. `/bye`로 종료합니다.

### 4-2. Python 테스트

```powershell
# LLM 테스트
python -c "from neo4j_graphrag.llm import OllamaLLM; llm = OllamaLLM(model_name='qwen3:8b-q4_K_M', model_params={'options':{'temperature':0}}); print(llm.invoke('안녕').content)"

# Embedding 테스트 (1024차원 확인)
python -c "from neo4j_graphrag.embeddings import OllamaEmbeddings; e = OllamaEmbeddings(model='bona/bge-m3-korean'); print(f'차원: {len(e.embed_query(\"테스트\"))}')"
```

---

## 5. 프로젝트 환경 설정

### 5-1. Python 패키지 설치

```powershell
cd c:\project\Knowledge
uv sync
```

### 5-2. .env 파일 확인

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4jpass
NEO4J_DB=neo4j

OLLAMA_MODEL=qwen3:8b-q4_K_M
OLLAMA_EMBEDDING_MODEL=bona/bge-m3-korean
```

### 5-3. Neo4j 데이터 확인 (Cypher 쿼리)

Neo4j Browser (`http://localhost:7474`)에 접속하여 아래 쿼리로 그래프 데이터를 확인합니다.

#### 전체 그래프 시각화

```cypher
-- 노드와 관계를 그래프로 표시 (Neo4j Browser에서 시각적 확인)
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

#### 노드/관계 수 확인

```cypher
-- 전체 노드 수
MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count
ORDER BY count DESC

-- 전체 관계 수
MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count
ORDER BY count DESC
```

#### 카테고리별 기사 수

```cypher
MATCH (a:Article)-[:BELONGS_TO]->(c:Category)
RETURN c.name AS category, count(a) AS articles
ORDER BY articles DESC
```

#### 언론사별 기사 수

```cypher
MATCH (m:Media)-[:PUBLISHED]->(a:Article)
RETURN m.name AS media, count(a) AS articles
ORDER BY articles DESC
```

#### 특정 기사의 전체 관계 탐색

```cypher
-- 기사 1개를 기준으로 연결된 언론사, 카테고리, 기자, 본문 청크 확인
MATCH (a:Article)
WITH a LIMIT 1
OPTIONAL MATCH (m:Media)-[:PUBLISHED]->(a)
OPTIONAL MATCH (a)-[:BELONGS_TO]->(cat:Category)
OPTIONAL MATCH (auth:Author)-[:WROTE]->(a)
OPTIONAL MATCH (a)-[:HAS_CHUNK]->(c:Content)
RETURN a.title, m.name AS media, cat.name AS category,
       auth.name AS author, count(c) AS chunks
```

#### 임베딩 상태 확인

```cypher
-- 임베딩이 생성된 Content 노드 수
MATCH (c:Content) WHERE c.embedding IS NOT NULL
RETURN count(c) AS embedded_count

-- 임베딩 차원 확인 (1024이면 정상)
MATCH (c:Content) WHERE c.embedding IS NOT NULL
RETURN size(c.embedding) AS dimension LIMIT 1
```

#### 벡터 인덱스 확인

```cypher
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties
WHERE type = "VECTOR"
RETURN name, labelsOrTypes, properties
```

---

## 6. notebooks/03_ToolsRetriever.ipynb 실행 순서

Neo4j가 실행 중이어야 합니다.

### 실행 순서

1. **Cell 2**: 모듈 로드
2. **Cell 3**: Ollama LLM + Embedding + Neo4j 초기화 → `All OK` 확인
3. **Cell 5**: 기존 임베딩 삭제 + bge-m3-korean으로 재임베딩 (198개, ~30초)
4. **Cell 7**: 벡터 인덱스 생성 (1024차원)
5. **Cell 9**: Vector 검색 테스트
6. **Cell 11**: VectorCypher Retriever 생성
7. **Cell 13**: VectorCypher 검색 테스트
8. **Cell 15~17**: Text2Cypher Retriever (스키마 추출 + Few-shot 설정)
9. **Cell 19**: Text2Cypher 검색 테스트
10. **Cell 21**: ToolsRetriever 생성 (3개 도구 통합)
11. **Cell 23**: ToolsRetriever 검색 테스트
12. **Cell 25**: GraphRAG 파이프라인 생성
13. **Cell 27~31**: GraphRAG 질의 테스트

### VRAM 사용 패턴

Ollama는 한 번에 하나의 모델만 GPU에 올립니다:

```
Cell 5 (임베딩):  bge-m3-korean GPU 로드 (~1.2GB)
Cell 9+ (검색):   bge-m3-korean (검색용 임베딩)
Cell 17+ (LLM):   Qwen3 GPU 로드 (~5.2GB), bge-m3-korean 자동 언로드
```

6GB VRAM에서 모델 교체는 자동으로 이루어지며, 첫 호출 시 몇 초의 로딩 시간이 있습니다.

---

## 7. 양자화 가이드

VRAM에 따라 적절한 양자화 버전을 선택합니다.

| VRAM | LLM 추천 | 태그 |
|------|----------|------|
| 4GB | Qwen3 4B | `qwen3:4b-q4_K_M` |
| 6GB | Qwen3 8B Q4 (기본) | `qwen3:8b-q4_K_M` |
| 8GB | Qwen3 8B Q5 | `qwen3:8b-q5_K_M` |
| 12GB+ | Qwen3 8B Q8 | `qwen3:8b-q8_0` |

양자화 단계 설명:

| 양자화 | 비트 | 특징 |
|--------|------|------|
| Q4_K_M | 4bit | 속도·품질 균형, 6GB에 적합 |
| Q5_K_M | 5bit | 품질 우선, 8GB 이상 권장 |
| Q8_0 | 8bit | 최고 품질, 12GB 이상 필요 |

모델 변경 시 `.env`의 `OLLAMA_MODEL`을 수정합니다.

---

## 8. 트러블슈팅

### `connection refused` 오류

Ollama가 실행되지 않고 있습니다.

```powershell
# Ollama 실행 상태 확인
ollama list

# 실행 안 되면 시작 메뉴에서 Ollama 실행
```

### `model not found` 오류

모델을 아직 다운로드하지 않았습니다.

```powershell
ollama pull qwen3:8b-q4_K_M
ollama pull bona/bge-m3-korean
```

### `import ollama` 실패

Python ollama 패키지가 설치되지 않았습니다.

```powershell
pip install ollama
# 또는
pip install "neo4j-graphrag[ollama]"
```

### 응답이 너무 느림

GPU가 사용되지 않고 CPU로 동작 중일 수 있습니다.

```powershell
# NVIDIA GPU 확인
nvidia-smi

# GPU 환경변수 확인
[Environment]::GetEnvironmentVariable("OLLAMA_NUM_GPU", "User")
```

`OLLAMA_NUM_GPU`가 설정되지 않았으면 `-1`로 설정 후 Ollama 재시작합니다.

### Text2Cypher가 잘못된 Cypher를 생성함

8B 모델에서는 복잡한 Cypher 생성이 부정확할 수 있습니다. `notebooks/03_ToolsRetriever.ipynb`의 Cell 17에서 `CYPHER_EXAMPLES`에 Few-shot 예제를 추가하면 정확도가 향상됩니다.

### Tool Calling이 잘못된 도구를 선택함

간단한 질문부터 테스트하세요:

- "북한 관련 뉴스" → vector_retriever 선택 예상
- "카테고리별 기사 수" → text2cypher_retriever 선택 예상
- "삼성전자 관련 기사와 같은 분야 뉴스" → vectorcypher_retriever 선택 예상

---

## 9. 모델 비교 참고

### LLM

| 모델 | 한국어 | Tool Calling | 용량 |
|------|--------|-------------|------|
| EXAONE 3.5 7.8B | 최상 | 불안정 | ~4.5GB |
| **Qwen3 8B** (기본) | 상 | 상 | ~5.2GB |
| Llama 3.2 3B | 중 | 중 | ~2GB |

### Embedding

| 모델 | 한국어 | 차원 | 용량 |
|------|--------|------|------|
| **bona/bge-m3-korean** (기본) | 상 | 1024 | ~1.2GB |
| bge-m3 | 중상 | 1024 | ~1.2GB |
| nomic-embed-text | 중 | 768 | ~274MB |
