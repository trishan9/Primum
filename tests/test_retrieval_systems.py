"""Dense index, hybrid fusion, reranking, query expansion and the retriever factory.

No model is loaded here: the dense index takes a deterministic bag-of-words embedder whose
cosines can be computed by hand, and the reranker is a stub.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

import primum.retrieval.bm25 as bm25_module
import primum.retrieval.concepts as concepts_module
import primum.retrieval.dense as dense_module
import primum.retrieval.factory as factory
from primum.config import Settings
from primum.retrieval import (
    RETRIEVERS,
    BM25Index,
    ConceptLinker,
    DenseIndex,
    EncoderNotFound,
    HybridRetriever,
    RerankedRetriever,
    build_retriever,
)
from primum.types import Passage

# The toy "encoder": one axis per concept, with clinical synonyms sharing the lay term's
# axis. This is the property a real dense encoder has and BM25 lacks.
AXES = {"heart": 0, "myocardial": 0, "attack": 1, "infarction": 1, "kidney": 2, "stone": 3}


def embed(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), 4))
    for row, text in enumerate(texts):
        for word in text.lower().split():
            if word in AXES:
                vectors[row, AXES[word]] += 1
    return vectors


def passage(pid: str, text: str) -> Passage:
    return Passage(passage_id=pid, doc_id=pid, text=text, source="pubmed")


PASSAGES = [
    passage("p0", "heart attack kidney"),  # (1, 1, 1, 0)
    passage("p1", "kidney stone"),  # (0, 0, 1, 1)
    passage("p2", "heart heart kidney"),  # (2, 0, 1, 0)
    passage("p3", "unrelated words"),  # zero vector
    passage("p4", "myocardial infarction"),  # (1, 1, 0, 0): a paraphrase of "heart attack"
]
QUERY = "heart attack"  # (1, 1, 0, 0)


def ids(hits) -> list[str]:
    return [hit.passage.passage_id for hit in hits]


# --- dense -------------------------------------------------------------------------------


def test_embeddings_are_unit_norm_and_zero_vectors_stay_finite():
    index = DenseIndex(embed, PASSAGES)
    norms = np.linalg.norm(index.embeddings, axis=1)
    np.testing.assert_allclose(norms, [1, 1, 1, 0, 1], atol=1e-6)
    assert index.embeddings.dtype == np.float32


def test_dense_search_scores_are_cosines():
    hits = DenseIndex(embed, PASSAGES, name="dense-toy").search(QUERY, k=5)
    # cos(q, p4) = 1; cos(q, p0) = 2 / (sqrt 2 * sqrt 3); cos(q, p2) = 2 / (sqrt 2 * sqrt 5);
    # p1 and p3 are orthogonal and tie at 0, broken by row order.
    assert ids(hits) == ["p4", "p0", "p2", "p1", "p3"]
    np.testing.assert_allclose(
        [h.score for h in hits],
        [1.0, 2 / (sqrt(2) * sqrt(3)), 2 / (sqrt(2) * sqrt(5)), 0.0, 0.0],
        atol=1e-6,
    )
    assert hits[0].retriever == "dense-toy"
    assert hits[1].components == {"dense": hits[1].score, "dense_rank": 2.0}


def test_dense_search_batch_matches_search():
    index = DenseIndex(embed, PASSAGES)
    queries = [QUERY, "kidney stone", "nothing"]
    for query, hits in zip(queries, index.search_batch(queries, k=2), strict=True):
        assert hits == index.search(query, k=2)
        assert len(hits) == 2


def test_dense_rejects_an_encoder_with_the_wrong_output_shape():
    with pytest.raises(ValueError, match="encoder returned shape"):
        DenseIndex(lambda texts: np.zeros((1, 4)), PASSAGES)


def test_missing_local_encoder_directory_is_a_typed_error(tmp_path):
    with pytest.raises(EncoderNotFound, match="primum train encoder"):
        DenseIndex(tmp_path / "encoder-ft", PASSAGES)
    assert issubclass(EncoderNotFound, FileNotFoundError)


def test_named_corpus_embeddings_are_cached_per_encoder_and_corpus(tmp_path, monkeypatch):
    corpus = list(PASSAGES)
    encoded: list[int] = []

    def fake_loader(model_name: str, batch_size: int):
        def counting_embed(texts: list[str]) -> np.ndarray:
            encoded.append(len(texts))
            return embed(texts)

        return counting_embed

    monkeypatch.setattr(dense_module, "resolve_corpus", lambda name: (corpus, tmp_path / name))
    monkeypatch.setattr(dense_module, "_sentence_transformer_embedder", fake_loader)

    first = DenseIndex("org/Toy-Encoder", "toy")
    assert first.name == "dense-org-toy-encoder"
    assert (tmp_path / "toy" / "dense-org-toy-encoder" / "embeddings.npy").exists()
    assert encoded == [5]

    second = DenseIndex("org/Toy-Encoder", "toy")  # loaded from disk: passages not re-encoded
    assert encoded == [5]
    np.testing.assert_array_equal(second.embeddings, first.embeddings)
    assert ids(second.search(QUERY, k=1)) == ["p4"]  # only now is the encoder needed
    assert encoded == [5, 1]

    corpus.append(passage("p5", "stone"))  # corpus changed: the cache is rebuilt
    assert DenseIndex("org/Toy-Encoder", "toy").embeddings.shape == (6, 4)

    DenseIndex("org/other", "toy")  # another encoder gets its own directory
    assert (tmp_path / "toy" / "dense-org-other" / "meta.json").exists()

    # A named retriever caches under its name; swapping the encoder behind that name (as
    # retraining does for "dense-ft") invalidates the cache instead of serving stale vectors.
    DenseIndex("org/Toy-Encoder", "toy", name="dense-ft")
    assert (tmp_path / "toy" / "dense-ft" / "embeddings.npy").exists()
    n_encodes = len(encoded)
    DenseIndex("org/Toy-Encoder", "toy", name="dense-ft")
    assert len(encoded) == n_encodes
    DenseIndex("org/retrained", "toy", name="dense-ft")
    assert len(encoded) == n_encodes + 1


def test_local_encoder_identity_tracks_its_weights(tmp_path):
    model_dir = tmp_path / "encoder-ft"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights after run 1")
    before = dense_module._encoder_identity(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"weights after run 2")
    after = dense_module._encoder_identity(model_dir)
    # Retraining overwrites the directory; embeddings cached from the old weights must miss.
    assert before != after
    assert before.startswith("encoder-ft@") and after.startswith("encoder-ft@")


# --- hybrid ------------------------------------------------------------------------------


@pytest.fixture
def hybrid() -> HybridRetriever:
    return HybridRetriever(
        BM25Index(PASSAGES), DenseIndex(embed, PASSAGES), name="hybrid-toy", rrf_k=60
    )


def test_hybrid_fuses_ranks_by_rrf(hybrid):
    hits = hybrid.search(QUERY, k=5)
    # BM25 ranking (lexical matches only):  p0, p2      -- it cannot see the paraphrase p4
    # dense ranking:                        p4, p0, p2, p1, p3
    assert ids(hits) == ["p0", "p2", "p4", "p1", "p3"]
    expected = [1 / 61 + 1 / 62, 1 / 62 + 1 / 63, 1 / 61, 1 / 64, 1 / 65]
    np.testing.assert_allclose([h.score for h in hits], expected)
    np.testing.assert_allclose([h.components["rrf"] for h in hits], expected)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.retriever == "hybrid-toy" for h in hits)


def test_hybrid_components_carry_both_stage_scores_and_only_real_ranks(hybrid):
    by_id = {hit.passage.passage_id: hit.components for hit in hybrid.search(QUERY, k=5)}
    bm25_scores = BM25Index(PASSAGES).score_all([QUERY])[0]

    assert by_id["p0"]["bm25_rank"] == 1 and by_id["p0"]["dense_rank"] == 2
    assert by_id["p0"]["bm25"] == pytest.approx(float(bm25_scores[0]))
    assert by_id["p0"]["dense"] == pytest.approx(2 / (sqrt(2) * sqrt(3)), abs=1e-6)

    # The paraphrase: found by the dense index alone. BM25 gave it score 0 and no rank.
    assert by_id["p4"]["dense"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["p4"]["dense_rank"] == 1
    assert by_id["p4"]["bm25"] == 0.0
    assert "bm25_rank" not in by_id["p4"]


def test_hybrid_beyond_the_head_continues_with_full_depth_rrf():
    # With candidate_k = 1 each retriever nominates one passage: the head is p0, p4 (tied at
    # 1/61; BM25 listed first). Asking for more appends the rest by RRF over the complete
    # rankings (BM25: p0, p2; dense: p4, p0, p2, p1, p3): p2 = 1/62 + 1/63, p1 = 1/64, p3 = 1/65.
    narrow = HybridRetriever(
        BM25Index(PASSAGES), DenseIndex(embed, PASSAGES), name="h", candidate_k=1
    )
    assert ids(narrow.search(QUERY, k=1)) == ["p0"]
    hits = narrow.search(QUERY, k=5)
    assert ids(hits) == ["p0", "p4", "p2", "p1", "p3"]
    np.testing.assert_allclose(
        [h.score for h in hits], [1 / 61, 1 / 61, 1 / 62 + 1 / 63, 1 / 64, 1 / 65]
    )
    # Tail passages were in neither candidate list, so they carry scores but no ranks.
    assert "bm25_rank" not in hits[2].components and "dense_rank" not in hits[2].components
    assert len(narrow.search(QUERY, k=50)) == len(PASSAGES)


def _crafted_indexes():
    """Fake first-stage scores over 200 passages where fusion depth changes the head.

    BM25 ranks rows 0..149 in index order; dense ranks rows in reverse index order. Row 30 is
    BM25 rank 31 and dense rank 70, so its RRF score 1/91 + 1/130 beats every head passage,
    but only if both candidate lists reach depth 70.
    """
    from types import SimpleNamespace

    n = 200
    passages = [passage(f"p{i}", f"doc {i}") for i in range(n)]
    bm25 = np.zeros(n)
    bm25[:150] = np.linspace(100, 1, 150)
    dense = np.linspace(-0.5, 0.9, n)
    dense[30] = np.sort(dense)[::-1][69] + 1e-9

    def index(scores):
        return SimpleNamespace(
            passages=passages, score_all=lambda qs: np.tile(scores, (len(qs), 1))
        )

    return index(bm25), index(dense)


def test_hybrid_ranking_does_not_depend_on_k():
    # Regression: fusion depth used to grow with k, so an evaluation asking for 100 hits
    # scored a different top 10 from the one served at k = 5.
    bm25, dense = _crafted_indexes()
    hybrid = HybridRetriever(bm25, dense, name="h", candidate_k=50)
    deepest = hybrid.search("q", k=150)
    assert "p30" in ids(deepest)[:100] and "p30" not in ids(deepest)[:10]
    for k in (1, 5, 10, 50, 99, 100, 101):
        assert hybrid.search("q", k=k) == deepest[:k]


def test_reranker_pool_does_not_depend_on_k():
    bm25, dense = _crafted_indexes()
    hybrid = HybridRetriever(bm25, dense, name="h", candidate_k=50)

    class PrefersRow30:
        def score(self, pairs):
            return np.array([float(text == "doc 30") for _, text in pairs], dtype=np.float32)

    reranked = RerankedRetriever(hybrid, PrefersRow30(), name="r", candidate_k=50)
    deepest = reranked.search("q", k=150)
    # p30 is outside the deployed pool, so the cross-encoder must never lift it to the top.
    assert ids(deepest)[0] != "p30"
    for k in (1, 5, 10, 50, 100):
        assert reranked.search("q", k=k) == deepest[:k]


def test_hybrid_search_batch_matches_search(hybrid):
    queries = [QUERY, "kidney stone"]
    for query, hits in zip(queries, hybrid.search_batch(queries, k=3), strict=True):
        assert hits == hybrid.search(query, k=3)


def test_hybrid_rejects_indexes_over_different_passages():
    with pytest.raises(ValueError, match="different passage lists"):
        HybridRetriever(BM25Index(PASSAGES), DenseIndex(embed, PASSAGES[::-1]), name="h")


# --- reranking ---------------------------------------------------------------------------


class StubReranker:
    """Scores a passage by how many kidney-related words it contains."""

    def __init__(self) -> None:
        self.n_pairs = 0

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.n_pairs += len(pairs)
        return np.array(
            [sum(word in ("kidney", "stone") for word in text.split()) for _, text in pairs],
            dtype=np.float32,
        )


def test_reranker_reorders_candidates_and_extends_components(hybrid):
    stub = StubReranker()
    reranked = RerankedRetriever(hybrid, stub, name="hybrid-toy+rerank", candidate_k=4)
    hits = reranked.search(QUERY, k=3)
    # Fused top-4 is p0, p2, p4, p1 with stub scores 1, 1, 0, 2. p0 and p2 tie and keep
    # their fused order.
    assert stub.n_pairs == 4
    assert ids(hits) == ["p1", "p0", "p2"]
    assert [h.score for h in hits] == [2.0, 1.0, 1.0]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert all(h.retriever == "hybrid-toy+rerank" for h in hits)
    assert set(hits[1].components) == {"bm25", "dense", "rrf", "bm25_rank", "dense_rank", "rerank"}
    assert hits[1].components["rerank"] == 1.0


def test_reranker_scores_exactly_candidate_k_and_appends_the_rest_unscored(hybrid):
    stub = StubReranker()
    hits = RerankedRetriever(hybrid, stub, name="r", candidate_k=2).search(QUERY, k=5)
    # Fused order p0, p2, p4, p1, p3. Only p0 and p2 are reranked (stub scores tie at 1, so
    # they keep their order); p4, p1, p3 follow in fused order with their RRF scores.
    assert stub.n_pairs == 2
    assert ids(hits) == ["p0", "p2", "p4", "p1", "p3"]
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.retriever == "r" for h in hits)
    assert ["rerank" in h.components for h in hits] == [True, True, False, False, False]
    assert hits[2].score == hits[2].components["rrf"]


# --- factory -----------------------------------------------------------------------------


@pytest.fixture
def toy_corpus(tmp_path, monkeypatch):
    """A named corpus "toy" with on-disk caches under tmp_path and no real encoder."""
    resolve = lambda name: (list(PASSAGES), tmp_path / "indexes" / name)  # noqa: E731
    monkeypatch.setattr(bm25_module, "resolve_corpus", resolve)
    monkeypatch.setattr(dense_module, "resolve_corpus", resolve)
    monkeypatch.setattr(dense_module, "_sentence_transformer_embedder", lambda *_: embed)
    settings = Settings(finetuned_encoder_dir=tmp_path / "encoder-ft", candidate_k=4)
    monkeypatch.setattr(factory, "get_settings", lambda: settings)
    monkeypatch.setattr(factory, "_cache", {})
    return settings


def test_retriever_names_match_the_spec():
    assert RETRIEVERS == [
        "bm25",
        "dense-base",
        "dense-biomed",
        "dense-ft",
        "hybrid-base",
        "hybrid-ft",
        "hybrid-ft+rerank",
    ]


def test_factory_builds_and_caches_each_system(toy_corpus):
    bm25 = build_retriever("bm25", "toy")
    assert bm25.name == "bm25" and bm25 is build_retriever("bm25", "toy")
    assert ids(bm25.search(QUERY, k=5)) == ["p0", "p2"]

    dense = build_retriever("dense-base", "toy")
    assert dense.name == "dense-base"
    assert ids(dense.search(QUERY, k=1)) == ["p4"]

    hybrid = build_retriever("hybrid-base", "toy")
    assert hybrid.name == "hybrid-base"
    assert ids(hybrid.search(QUERY, k=3)) == ["p0", "p2", "p4"]
    # Composite systems reuse the cached components, not second copies of the indexes.
    assert hybrid._bm25 is bm25 and hybrid._dense is dense


def test_factory_never_substitutes_an_untrained_encoder(toy_corpus):
    for name in ("dense-ft", "hybrid-ft", "hybrid-ft+rerank"):
        with pytest.raises(EncoderNotFound):
            build_retriever(name, "toy")


def test_factory_builds_fine_tuned_and_reranked_systems_once_the_encoder_exists(
    toy_corpus, monkeypatch
):
    toy_corpus.finetuned_encoder_dir.mkdir()
    stub = StubReranker()
    monkeypatch.setattr(factory, "_reranker", lambda: stub)

    reranked = build_retriever("hybrid-ft+rerank", "toy")
    hits = reranked.search(QUERY, k=2)
    assert reranked.name == "hybrid-ft+rerank"
    assert ids(hits) == ["p1", "p0"]
    assert stub.n_pairs == toy_corpus.candidate_k
    assert build_retriever("hybrid-ft", "toy").name == "hybrid-ft"
    assert build_retriever("dense-ft", "toy").name == "dense-ft"


def test_factory_concept_expansion_stage(toy_corpus, monkeypatch):
    linker = ConceptLinker(
        [{"preferred": "Heart Attack", "synonyms": ["Myocardial Infarction"], "mesh_ids": []}]
    )
    monkeypatch.setattr(concepts_module, "_default_linker", lambda: linker)

    expanded = build_retriever("bm25+concepts", "toy")
    hits = expanded.search(QUERY, k=5)
    # Plain BM25 finds p0 and p2; with "Myocardial Infarction" appended it also finds p4.
    assert set(ids(hits)) == {"p0", "p2", "p4"}
    assert all(h.retriever == "bm25+concepts" for h in hits)


@pytest.mark.parametrize("name", ["", "tfidf", "dense-large", "hybrid-", "bm25+magic"])
def test_factory_rejects_unknown_names(toy_corpus, name):
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever(name, "toy")


def test_factory_slow_build_does_not_block_other_systems_and_runs_once(monkeypatch):
    """A first build that takes minutes (embedding a corpus) must not stall lookups of other
    systems, and concurrent first requests for the same system must share one build."""
    import threading

    release = threading.Event()
    calls: list[tuple[str, str]] = []

    def fake_build(name: str, corpus: str) -> object:
        calls.append((name, corpus))
        if name == "slow":
            assert release.wait(timeout=10), "test hung: the slow build was never released"
        return object()

    monkeypatch.setattr(factory, "_build", fake_build)
    monkeypatch.setattr(factory, "_cache", {})
    monkeypatch.setattr(factory, "_key_locks", {})
    cached = build_retriever("fast", "toy")

    results: list[object] = []
    slow = [
        threading.Thread(target=lambda: results.append(build_retriever("slow", "toy")))
        for _ in range(2)
    ]
    for thread in slow:
        thread.start()
    other = threading.Thread(target=lambda: results.append(build_retriever("fast", "toy")))
    other.start()
    other.join(timeout=2)
    assert not other.is_alive(), "a cache hit waited on an unrelated build"
    assert results == [cached]

    release.set()
    for thread in slow:
        thread.join(timeout=10)
    assert results[1] is results[2]
    assert calls.count(("slow", "toy")) == 1
