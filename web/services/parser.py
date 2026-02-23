import re
import logging
from typing import List


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


# ---------------------------------------------------------------------------
# 유효성 검사
# ---------------------------------------------------------------------------

def is_valid_article_id(article_id: str) -> bool:
    return bool(re.match(r"^ART_\d{3}_\d{10}$", article_id))


def is_valid_category(category: str) -> bool:
    invalid = {
        "Unknown", "No title",
        "text2cypher_retriever", "vector_retriever", "vectorcypher_retriever",
    }
    if category in invalid or "retriever" in category.lower():
        return False
    return len(category) > 0


# ---------------------------------------------------------------------------
# 필드 추출
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 노드/엣지 추출
# ---------------------------------------------------------------------------

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
        aid = f"ART_{media_code.zfill(3)}_{article_num.zfill(10)}"
        nodes.append(f"Article_{aid}")

    aid_direct = re.findall(r"(ART_\d{3}_\d{10})(?!_chunk)", answer)
    for aid in aid_direct:
        node = f"Article_{aid}"
        if node not in nodes:
            nodes.append(node)

    if nodes:
        edges.append("HAS_CHUNK")

    return list(set(nodes)), list(set(edges))
