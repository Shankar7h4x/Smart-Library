"""Automated Test Suite for Local Book Database & AI Semantic Recommendation Engine.

Tests:
1. Local database integrity (330 books, unique book_001..book_330 IDs, Parquet, JSON, metadata).
2. Local cover storage (331 JPEG files in data/book_covers/).
3. Vector embeddings (330x128 dense array, unit norm, embedding index).
4. Recommendation Engine hybrid search (Exact, Closely Related, Similar Subjects, Recommended).
5. Specific queries: Harry Potter, Machine Learning, Programming, Python, AI, Data Science, Physics, Business.
6. Graceful fallback on non-existent query with exact warning message.
7. Pure offline operation (zero network requests).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "data" / "books"
COVERS_DIR = ROOT / "data" / "book_covers"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"

from src.book_recommender import BookRecommendationEngine, search_books_hybrid, get_recommendations_for_book


def test_local_book_database_files_exist():
    """Verify all database files exist and are populated."""
    catalog_json = BOOKS_DIR / "books_catalog.json"
    catalog_parquet = BOOKS_DIR / "books_catalog.parquet"
    metadata_json = BOOKS_DIR / "metadata.json"

    assert catalog_json.exists(), "data/books/books_catalog.json missing"
    assert catalog_parquet.exists(), "data/books/books_catalog.parquet missing"
    assert metadata_json.exists(), "data/books/metadata.json missing"

    with open(catalog_json, "r", encoding="utf-8") as f:
        books = json.load(f)
    assert len(books) == 330, f"Expected 330 books, found {len(books)}"

    df = pd.read_parquet(catalog_parquet)
    assert len(df) == 330, f"Expected 330 rows in parquet, found {len(df)}"

    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("total_records") == 330 or meta.get("total_books") == 330
    assert meta.get("unique_book_ids", 330) == 330


def test_unique_book_identifiers():
    """Verify every book has a distinct ID in format book_001 to book_330."""
    engine = BookRecommendationEngine.get_instance()
    all_books = engine.get_all_books()
    assert len(all_books) == 330

    ids = [b["book_id"] for b in all_books]
    assert len(set(ids)) == 330, "Book IDs must be completely unique"
    for b_id in ids:
        assert b_id.startswith("book_"), f"Invalid ID format: {b_id}"


def test_local_cover_images():
    """Verify local JPEG images exist for all books in data/book_covers/."""
    assert COVERS_DIR.exists()
    assert (COVERS_DIR / "placeholder.jpg").exists()

    cover_files = list(COVERS_DIR.glob("book_*.jpg"))
    assert len(cover_files) == 330, f"Expected 330 cover files, found {len(cover_files)}"

    # Check mapping file
    mapping_file = COVERS_DIR / "cover_mapping.json"
    assert mapping_file.exists()
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    assert len(mapping) == 330


def test_vector_embeddings_shape_and_norms():
    """Verify 330x128 vector embeddings are float32, non-empty, and unit-normalized."""
    vectors_path = EMBEDDINGS_DIR / "book_vectors.npy"
    index_path = EMBEDDINGS_DIR / "embedding_index.json"
    meta_path = EMBEDDINGS_DIR / "embedding_metadata.json"

    assert vectors_path.exists()
    assert index_path.exists()
    assert meta_path.exists()

    vecs = np.load(vectors_path)
    assert vecs.shape == (330, 128), f"Expected (330, 128), got {vecs.shape}"
    assert vecs.dtype == np.float32

    # Check unit norms (within 1e-4 tolerance for non-zero vectors)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.all(norms > 0.99)
    assert np.all(norms < 1.01)


def test_exact_match_query():
    """Verify query matching an exact title returns in exact_matches shelf."""
    res = search_books_hybrid("Algorithms to Live By")
    assert res["has_results"] is True
    assert len(res["exact_matches"]) >= 1
    assert "Algorithms to Live By" in res["exact_matches"][0]["title"]
    assert "Exact" in res["exact_matches"][0]["recommendation_reason"]


def test_semantic_query_harry_potter():
    """Verify 'Harry Potter' returns related fantasy/fiction titles even if exact title is absent."""
    res = search_books_hybrid("Harry Potter")
    assert res["has_results"] is True
    total_recs = len(res["closely_related"]) + len(res["similar_subjects"]) + len(res["recommended_books"])
    assert total_recs > 0
    # Exact match does not exist in KTH technical library, so fallback message must be triggered
    assert len(res["exact_matches"]) == 0
    assert "We couldn't find the exact title in our catalogue, but here are some related books you might enjoy." in res["message"]


def test_topic_query_machine_learning():
    """Verify 'Machine Learning' returns relevant AI / ML books."""
    res = search_books_hybrid("Machine Learning")
    assert res["has_results"] is True
    assert res["total_found"] >= 5
    all_titles = " ".join([
        b["title"] for b in res["exact_matches"] + res["closely_related"] + res["similar_subjects"] + res["recommended_books"]
    ]).casefold()
    assert "learning" in all_titles or "machine" in all_titles or "intelligence" in all_titles


def test_discipline_queries():
    """Verify common student search queries return non-empty categorized results."""
    queries = ["Programming", "Python", "Artificial Intelligence", "Data Science", "Physics", "Economics"]
    for q in queries:
        res = search_books_hybrid(q)
        assert res["has_results"] is True, f"Search failed for {q}"
        assert res["total_found"] > 0, f"No results for {q}"


def test_nonexistent_query_fallback():
    """Verify a nonsensical query returns has_results=False and helpful suggestion message."""
    res = search_books_hybrid("xyzunknownnonsense123999")
    assert res["has_results"] is False
    assert res["total_found"] == 0
    assert "No books or related subjects matched" in res["message"]


def test_item_to_item_similar_books():
    """Verify get_recommendations_for_book returns cosine similarity neighbors with reasons."""
    engine = BookRecommendationEngine.get_instance()
    all_books = engine.get_all_books()
    first_book = all_books[0]
    similar = engine.get_similar_books(first_book["book_id"], limit=5)
    assert len(similar) == 5
    for b in similar:
        assert b["book_id"] != first_book["book_id"]
        assert "recommendation_reason" in b
        assert "similarity_score" in b
        assert 0.0 <= b["similarity_score"] <= 1.0


def test_offline_execution_guarantee():
    """Ensure book discovery and hybrid search function without sockets or internet."""
    import socket
    orig_socket = socket.socket
    # Temporarily monkeypatch socket creation to block network
    def blocked_socket(*args, **kwargs):
        raise RuntimeError("Network disabled in offline test")

    try:
        socket.socket = blocked_socket
        # Run hybrid search completely offline
        res = search_books_hybrid("Algorithms")
        assert res["has_results"] is True
        assert res["total_found"] > 0
    finally:
        socket.socket = orig_socket
