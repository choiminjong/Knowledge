# Knowledge GraphRAG

Neo4j 그래프 데이터베이스와 Ollama 로컬 LLM을 활용한 **뉴스 기사 검색 및 요약 시스템**입니다.

네이버 뉴스를 크롤링하고, 그래프 DB에 적재한 후, GraphRAG 파이프라인으로 자연어 질의응답과 시각화를 제공합니다.

## 아키텍처

```
[네이버 뉴스] ──크롤링──> [Excel] ──그래프 구축──> [Neo4j]
                                                    │
                                              임베딩 생성
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │               │               │
                              VectorRetriever  VectorCypher  Text2Cypher
                                    │               │               │
                                    └───────┬───────┘               │
                                            │                       │
                                      ToolsRetriever ◄──────────────┘
                                            │
                                        GraphRAG
                                            │
                                    ┌───────┴───────┐
                                    │               │
                              요약 브리핑      기사 목록
                                    │               │
                                    └───────┬───────┘
                                            │
                                      웹 시각화 (FastAPI + vis.js)
```

## 실행 순서

| 단계 | 노트북 | 설명 |
|------|--------|------|
| STEP 1 | `notebooks/01_DataScrapping.ipynb` | 네이버 뉴스 6개 카테고리 크롤링 |
| STEP 2 | `notebooks/02_GraphBuilder.ipynb` | Neo4j 그래프 DB 구축 (Recursive Chunking) |
| STEP 3 | `notebooks/03_ToolsRetriever.ipynb` | 임베딩 생성 + GraphRAG 검색 파이프라인 |
| Web | `uv run python -m web.app` | 그래프 시각화 + AI 검색 웹 서버 |

## 빠른 시작

### 사전 준비

- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Neo4j Desktop](https://neo4j.com/download/) 또는 Neo4j Aura
- [Ollama](https://ollama.ai)

### 설치

```bash
git clone https://github.com/choiminjong/Knowledge.git
cd Knowledge

# 환경 구성 (Python + 모든 의존성)
uv sync

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어 Neo4j 비밀번호 등 설정

# Ollama 모델 다운로드
ollama pull qwen3:8b-q4_K_M
ollama pull bona/bge-m3-korean
```

### 실행

```bash
# STEP 1~3: Jupyter 노트북을 순서대로 실행
# (notebooks/ 폴더의 01 → 02 → 03 순서)

# 웹 시각화 서버 실행
uv run python -m web.app
# http://localhost:8000 접속
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| LLM | Ollama `qwen3:8b-q4_K_M` (로컬) |
| Embedding | Ollama `bona/bge-m3-korean` (1024차원) |
| Graph DB | Neo4j |
| GraphRAG | `neo4j-graphrag[ollama]` |
| 웹 서버 | FastAPI + Uvicorn |
| 프론트엔드 | vis-network.js + marked.js |
| 크롤링 | Selenium + undetected-chromedriver |

## 프로젝트 구조

```
Knowledge/
├── notebooks/                    # Jupyter 노트북 (STEP 1~3)
│   ├── 01_DataScrapping.ipynb    # 네이버 뉴스 크롤링
│   ├── 02_GraphBuilder.ipynb     # Neo4j 그래프 구축
│   └── 03_ToolsRetriever.ipynb   # GraphRAG 검색 시스템
├── web/                          # 웹 시각화 서버 (FastAPI)
│   ├── app.py                    # FastAPI 앱 생성, 미들웨어, 라우터 등록
│   ├── config.py                 # 환경변수, 설정값
│   ├── routers/                  # API 라우트 핸들러
│   │   ├── graph.py              # GET /graph - 그래프 데이터 조회
│   │   ├── query.py              # POST /query - GraphRAG 검색
│   │   └── health.py             # GET /health - 서버 상태 확인
│   ├── services/                 # 비즈니스 로직
│   │   ├── rag_service.py        # LLM, Retriever, 프롬프트 초기화
│   │   └── parser.py             # 응답 파싱, Cypher 캡처 유틸리티
│   └── static/
│       └── index.html            # 프론트엔드 UI
├── docs/                         # 문서
│   ├── guides/                   # 상세 가이드
│   │   ├── 01_환경설정.md
│   │   ├── 02_데이터수집.md
│   │   ├── 03_그래프구축.md
│   │   ├── 04_검색시스템.md
│   │   ├── 05_웹시각화.md
│   │   └── 06_API레퍼런스.md
│   ├── releases/                 # 릴리즈 노트
│   │   └── v0.1.0.md
│   └── Ollama_로컬RAG_가이드.md
├── pyproject.toml                # 프로젝트 설정 + 의존성
├── uv.lock                       # 의존성 락 파일
├── .python-version               # Python 3.12
├── .env.example                  # 환경변수 템플릿
└── README.md
```

## 그래프 스키마

```
(Media)─[:PUBLISHED]─>(Article)─[:HAS_CHUNK]─>(Content)
                          │
                     [:BELONGS_TO]
                          │
                      (Category)
                          
(Author)─[:WROTE]─>(Article)
```

| 노드 | 설명 |
|------|------|
| Article | 뉴스 기사 (제목, URL, 발행일) |
| Content | 기사 본문 청크 (임베딩 벡터 포함) |
| Category | 카테고리 (정치, 경제, 사회, 생활/문화, IT/과학, 세계) |
| Media | 언론사 |
| Author | 기자 |

## 문서

상세한 사용 가이드는 [docs/guides/](docs/guides/) 폴더를 참고하세요.

- [환경설정 가이드](docs/guides/01_환경설정.md)
- [데이터 수집 가이드](docs/guides/02_데이터수집.md)
- [그래프 구축 가이드](docs/guides/03_그래프구축.md)
- [검색 시스템 가이드](docs/guides/04_검색시스템.md)
- [웹 시각화 가이드](docs/guides/05_웹시각화.md)
- [API 레퍼런스](docs/guides/06_API레퍼런스.md)
- [Ollama 로컬 RAG 가이드](docs/Ollama_로컬RAG_가이드.md)

## 라이선스

MIT License
