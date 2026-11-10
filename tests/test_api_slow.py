"""Smoke test of the real app on the real `kb` corpus with the lexical retriever.

Needs the built corpus (`primum data build kb`); builds or loads the BM25 index under
`data/indexes/kb/bm25/`. Loads no neural model and makes no API call.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from primum.api.app import create_app
from primum.config import get_settings

pytestmark = pytest.mark.slow

KB = get_settings().processed_dir / "kb" / "manifest.json"


@pytest.mark.skipif(not KB.is_file(), reason="kb corpus not built")
def test_retrieve_bm25_on_the_real_kb() -> None:
    client = TestClient(create_app())
    query = "What are the symptoms of asthma?"
    response = client.post(
        "/api/retrieve", json={"query": query, "k": 5, "retrievers": ["bm25", "bm25+concepts"]}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    for name in ("bm25", "bm25+concepts"):
        hits = body["results"][name]
        assert len(hits) == 5
        assert [hit["rank"] for hit in hits] == [1, 2, 3, 4, 5]
        scores = [hit["score"] for hit in hits]
        assert scores == sorted(scores, reverse=True) and scores[-1] > 0
        assert all(hit["components"]["bm25"] > 0 for hit in hits)
    # MedlinePlus has an asthma topic, so the concept linker must find it and expand the query.
    assert any(c["preferred"].lower() == "asthma" for c in body["concepts"])
    assert body["expanded_query"].startswith(query)
    assert any("asthma" in hit["passage"]["text"].lower() for hit in body["results"]["bm25"])

    health = client.get("/api/health").json()
    assert "bm25" in health["indexes"]["kb"] and health["corpora"]["kb"] is True
    assert client.get("/api/meta").json()["corpora"]["kb"] > 0
