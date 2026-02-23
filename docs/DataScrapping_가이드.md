# DataScrapping 가이드 — DataScrapping.ipynb / data_scrapping.py

전체 내용을 Notion 페이지에 복사해 넣으세요.

---

## 개요

**목적**: 네이버 뉴스 분야별 헤드라인 기사를 수집하여 Excel로 저장. 이후 GraphBuilder에서 사용.

**수집 URL**  
정치 100 | 경제 101 | 사회 102 | 생활/문화 103 | IT/과학 105 | 세계 104  
→ `https://news.naver.com/section/{코드}`

**특징**: Selenium + undetected-chromedriver(headless), 노트북/스크립트 2가지 실행, 기본 60건(카테고리당 10건)

---

## 사전 준비

- **패키지**: `uv pip install -r requirements.txt` (또는 `pip install -r requirements.txt`)
- **Chrome**: 최신 버전 설치 (`chrome://version`으로 확인)
- **체크**: 가상환경 활성화, 패키지 설치, Chrome 설치, 인터넷 연결

---

## 실행 (노트북)

`DataScrapping.ipynb`를 열고 **위에서부터 순서대로** 셀 실행.

1. 패키지 임포트 (selenium, undetected_chromedriver, pandas 등)
2. 카테고리 설정 + `NUM_ARTICLES_PER_CATEGORY = 10`
3. Selenium 드라이버 초기화
4. `get_article_links`, `parse_article_detail` 함수 정의
5. 전체 카테고리 크롤링 실행
6. `df_articles = pd.DataFrame(all_articles)` 후 Excel 저장

**예상 소요**: 60건 기준 약 5~10분

---

## 실행 (스크립트)

```bash
python data_scrapping.py
```

**옵션**: `--num 5` / `--categories 정치 경제` / `--output my.xlsx`

```bash
python data_scrapping.py --num 3 --categories 정치 경제 --output politics.xlsx
```

- Ctrl+C로 중단 가능 (진행분은 저장되지 않음)

---

## 수집 데이터

**출력**: `Articles_YYYYMMDD_HHMMSS.xlsx`

| 컬럼 | 설명 |
|------|------|
| article_id | `ART_매체코드_기사번호` 또는 `ART_YYYYMMDDHHMMSS` |
| title | 기사 제목 |
| content | 기사 본문 |
| url | 기사 URL |
| published_date | 발행일 |
| source | 언론사 |
| author | 기자명 |
| category | 정치, 경제, 사회 등 |

**제외**: URL에 `/comment/` 포함 시, 제목 없는 경우

---

## 트러블슈팅

**numpy.dtype size changed** → `pip install --upgrade numpy pandas` 또는 가상환경 재생성 후 재설치

**ChromeDriver 버전 오류** → Chrome 최신 버전 설치

**headless** → `--headless=new` 사용 시 창 미표시. 디버깅 시 해당 옵션 제거

**링크 0개** → `time.sleep(3)` 증가, CSS 셀렉터 확인 (`a.sa_text_lede`, `#dic_area` 등)

**제목/본문 비어 있음** → 제목: `#title_area span`, 본문: `#dic_area` 등 셀렉터 확인

**Kernel crashed** → numpy/pandas 재설치, 다른 커널 선택
