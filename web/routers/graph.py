from fastapi import APIRouter, HTTPException

from web.config import NEO4J_DB
from web.services.rag_service import driver

router = APIRouter()


@router.get("/graph")
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
