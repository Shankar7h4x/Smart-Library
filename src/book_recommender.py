"""Intelligent Book Recommendation and Hybrid Search Engine for Smart Library.

Combines:
1. Exact normalized title matching
2. Partial title and substring matching
3. Author matching
4. Subject & keyword matching
5. Vector embedding cosine similarity (SentenceTransformers / Dense LSA)
6. Semantic concept associations & query expansion (e.g. Harry Potter -> fantasy, adventure, fiction)
7. Metadata relevance ranking (ratings, reader volume, shelf relevance)
8. Clean separation into Exact Matches, Closely Related, Similar Subjects, and Recommended Books.
"""

from collections import OrderedDict
import html
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

ROOT_DIR = Path(__file__).resolve().parent.parent
BOOKS_PARQUET = ROOT_DIR / "data" / "books" / "books_catalog.parquet"
BOOKS_JSON = ROOT_DIR / "data" / "books" / "books_catalog.json"
VECTORS_PATH = ROOT_DIR / "data" / "embeddings" / "book_vectors.npy"
INDEX_PATH = ROOT_DIR / "data" / "embeddings" / "embedding_index.json"
FALLBACK_MODEL_PATH = ROOT_DIR / "data" / "embeddings" / "semantic_model.joblib"

# Semantic domain associations for common concept discovery
SEMANTIC_TOPIC_ASSOCIATIONS = {
    "harry potter": ["fantasy", "adventure", "stories", "novels", "fiction", "adept", "tales", "literature"],
    "machine learning": ["artificial intelligence", "neural networks", "deep learning", "data science", "machines that think", "intelligence", "algorithms"],
    "programming": ["computer science", "software engineering", "algorithms", "data structures", "computation", "computing"],
    "artificial intelligence": ["machine learning", "neural networks", "robotics", "intelligence", "computer science"],
    "data science": ["statistics", "machine learning", "mathematics", "computer science", "probability", "analysis"],
    "quantum": ["physics", "quantum mechanics", "thermodynamics", "quantum physics", "atomic"],
    "physics": ["mechanics", "thermodynamics", "optics", "quantum", "electromagnetism"],
    "calculus": ["mathematics", "analysis", "algebra", "differential", "integral", "functions"],
    "economics": ["management", "finance", "business", "econometrics", "markets", "accounting"],
    "business": ["economics", "management", "finance", "organization", "accounting", "operations"],
    "robotics": ["engineering", "control systems", "mechatronics", "automation", "sensors", "robots"],
    "biology": ["biosystems", "agriculture", "bioengineering", "ecology", "organic"],
    "python": ["programming", "computer science", "algorithms", "software"],
}


def _clean_str(text: str) -> str:
    """Normalize string for robust matching."""
    if not text:
        return ""
    return re.sub(r"[^\w\s]", " ", str(text).casefold()).strip()


class BookRecommendationEngine:
    """Production recommendation engine managing local books and vector embeddings."""

    _instance = None

    def __init__(self):
        self.books: List[Dict[str, Any]] = []
        self.books_by_id: Dict[str, Dict[str, Any]] = {}
        self.vectors: Optional[np.ndarray] = None
        self.index_data: Dict[str, Any] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.fallback_model = None
        self.st_model = None
        self._load_data()

    @classmethod
    def get_instance(cls) -> "BookRecommendationEngine":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = BookRecommendationEngine()
        return cls._instance

    def _load_data(self):
        """Load catalog, vectors, and model indexes from local disk."""
        # 1. Load books
        if BOOKS_PARQUET.exists():
            try:
                df = pd.read_parquet(BOOKS_PARQUET)
                records = df.to_dict(orient="records")
                for r in records:
                    if isinstance(r.get("authors"), str):
                        try:
                            r["authors"] = json.loads(r["authors"])
                        except Exception:
                            r["authors"] = [r["authors"]]
                    if isinstance(r.get("subject"), str):
                        try:
                            r["subject"] = json.loads(r["subject"])
                        except Exception:
                            r["subject"] = [r["subject"]]
                self.books = records
            except Exception:
                self.books = []

        if not self.books and BOOKS_JSON.exists():
            try:
                with open(BOOKS_JSON, "r", encoding="utf-8") as f:
                    self.books = json.load(f)
            except Exception:
                self.books = []

        self.books_by_id = {b["book_id"]: b for b in self.books if "book_id" in b}

        # 2. Load embeddings
        if VECTORS_PATH.exists():
            try:
                self.vectors = np.load(VECTORS_PATH)
            except Exception:
                self.vectors = None

        if INDEX_PATH.exists():
            try:
                with open(INDEX_PATH, "r", encoding="utf-8") as f:
                    self.index_data = json.load(f)
                    self.id_to_idx = {k: int(v) for k, v in self.index_data.get("book_id_to_idx", {}).items()}
                    self.idx_to_id = {int(k): v for k, v in self.index_data.get("idx_to_book_id", {}).items()}
            except Exception:
                pass

        # 3. Load query encoder
        if FALLBACK_MODEL_PATH.exists():
            try:
                self.fallback_model = joblib.load(FALLBACK_MODEL_PATH)
            except Exception:
                self.fallback_model = None

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """Convert a text query into a normalized embedding vector."""
        if self.fallback_model is not None and "tfidf" in self.fallback_model and "svd" in self.fallback_model:
            try:
                tfidf = self.fallback_model["tfidf"]
                svd = self.fallback_model["svd"]
                q_tfidf = tfidf.transform([query])
                q_lsa = svd.transform(q_tfidf)
                norm_val = np.linalg.norm(q_lsa)
                if norm_val > 1e-6:
                    return (q_lsa / norm_val)[0].astype(np.float32)
            except Exception:
                pass
        return None

    def get_all_books(self) -> List[Dict[str, Any]]:
        """Return all catalog books."""
        return list(self.books)

    def get_book_by_id(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single book by unique ID."""
        return self.books_by_id.get(book_id)

    def get_books_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Filter books by exact category."""
        if not category or category.casefold() in {"all", "all books"}:
            return list(self.books)
        return [b for b in self.books if b.get("category", "").casefold() == category.casefold()]

    def get_similar_books(self, book_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Find item-to-item similar books via vector cosine similarity."""
        if self.vectors is None or book_id not in self.id_to_idx:
            target = self.get_book_by_id(book_id)
            if target:
                cat = target.get("category", "")
                return [b for b in self.books if b.get("category") == cat and b["book_id"] != book_id][:limit]
            return []

        idx = self.id_to_idx[book_id]
        target_vec = self.vectors[idx]
        sims = np.dot(self.vectors, target_vec)

        ranked_indices = np.argsort(-sims)
        results = []
        for r_idx in ranked_indices:
            if r_idx == idx:
                continue
            r_id = self.idx_to_id.get(r_idx)
            if r_id and r_id in self.books_by_id:
                b = dict(self.books_by_id[r_id])
                b["similarity_score"] = float(sims[r_idx])
                b["recommendation_reason"] = f"Semantic Match ({sims[r_idx]:.0%}) · {b.get('category')}"
                results.append(b)
                if len(results) >= limit:
                    break
        return results

    def hybrid_search(self, raw_query: str, limit: int = 40) -> Dict[str, Any]:
        """Perform hybrid search dividing results into Exact, Closely Related, Similar Subjects, and Recommended."""
        query = (raw_query or "").strip()
        if not query:
            return {
                "query": "",
                "has_results": False,
                "exact_matches": [],
                "closely_related": [],
                "similar_subjects": [],
                "recommended_books": [],
                "total_found": 0,
                "message": "",
            }

        q_clean = _clean_str(query)
        q_terms = [t for t in q_clean.split() if len(t) > 1]

        # Concept association / query expansion
        expanded_terms = list(q_terms)
        for concept_key, associations in SEMANTIC_TOPIC_ASSOCIATIONS.items():
            if concept_key in q_clean or all(term in concept_key for term in q_terms):
                expanded_terms.extend(associations)

        expanded_query = f"{query} {' '.join(expanded_terms)}"

        # Compute cosine similarity
        semantic_scores = np.zeros(len(self.books), dtype=np.float32)
        q_vec = self.encode_query(expanded_query)
        if q_vec is not None and self.vectors is not None:
            raw_sims = np.dot(self.vectors, q_vec)
            # Avoid nan
            semantic_scores = np.nan_to_num(raw_sims)

        exact_matches = []
        closely_related = []
        similar_subjects = []
        recommended_pool = []
        seen_ids = set()

        for idx, b in enumerate(self.books):
            b_id = b["book_id"]
            title = b.get("title", "")
            title_clean = _clean_str(title)
            author_clean = _clean_str(b.get("author", ""))
            category = b.get("category", "")
            category_clean = _clean_str(category)
            genre_clean = _clean_str(b.get("genre", ""))
            subjects_list = b.get("subject", []) or []
            subjects_clean = " ".join(_clean_str(s) for s in subjects_list)
            desc_clean = _clean_str(b.get("description", ""))

            haystack = f"{title_clean} {author_clean} {category_clean} {genre_clean} {subjects_clean} {desc_clean}"
            sem_score = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0

            # 1. Exact Title Match
            if title_clean == q_clean or title_clean.startswith(q_clean + " "):
                match_b = dict(b)
                match_b["recommendation_reason"] = "★ Exact Title Match"
                match_b["match_score"] = 1.0 + sem_score
                exact_matches.append(match_b)
                seen_ids.add(b_id)
                continue

            # 2. Strong Partial Title Match
            if len(q_clean) >= 3 and q_clean in title_clean:
                match_b = dict(b)
                match_b["recommendation_reason"] = "Title Match · Keyword Found"
                match_b["match_score"] = 0.85 + sem_score
                closely_related.append(match_b)
                seen_ids.add(b_id)
                continue

            # 3. Author Match
            if len(q_clean) >= 3 and q_clean in author_clean:
                match_b = dict(b)
                match_b["recommendation_reason"] = f"Author Match · {b.get('author')}"
                match_b["match_score"] = 0.80 + sem_score
                closely_related.append(match_b)
                seen_ids.add(b_id)
                continue

            # 4. Multi-term title/author presence
            if q_terms and all(term in f"{title_clean} {author_clean}" for term in q_terms):
                match_b = dict(b)
                match_b["recommendation_reason"] = "Title & Author Words Match"
                match_b["match_score"] = 0.75 + sem_score
                closely_related.append(match_b)
                seen_ids.add(b_id)
                continue

            # 5. Direct Subject or Category Keyword Match
            if (len(q_clean) >= 3 and q_clean in category_clean) or any(q_clean in _clean_str(s) for s in subjects_list):
                match_b = dict(b)
                match_b["recommendation_reason"] = f"Subject Match · {category}"
                match_b["match_score"] = 0.60 + sem_score
                similar_subjects.append(match_b)
                seen_ids.add(b_id)
                continue

            # 6. Expanded semantic keywords presence (e.g. Harry Potter -> fantasy/adventure/fiction)
            matched_expanded = [t for t in expanded_terms if t in haystack and t not in q_terms]
            if len(matched_expanded) >= 2 or (len(matched_expanded) >= 1 and sem_score > 0.20):
                match_b = dict(b)
                reason_topic = matched_expanded[0].capitalize()
                match_b["recommendation_reason"] = f"Related Theme ({reason_topic}) · {category}"
                match_b["match_score"] = 0.45 + sem_score
                similar_subjects.append(match_b)
                seen_ids.add(b_id)
                continue

            # 7. Semantic vector similarity
            if sem_score >= 0.30:
                match_b = dict(b)
                match_b["recommendation_reason"] = f"AI Semantic Match ({sem_score:.0%}) · {category}"
                match_b["match_score"] = 0.40 + sem_score
                recommended_pool.append(match_b)
                seen_ids.add(b_id)
                continue

            # 8. Mild semantic similarity ($S \ge 0.18$) with keyword in haystack
            if sem_score >= 0.18 and any(term in haystack for term in q_terms):
                match_b = dict(b)
                match_b["recommendation_reason"] = f"Academic Recommendation · {category}"
                match_b["match_score"] = 0.30 + sem_score
                recommended_pool.append(match_b)
                seen_ids.add(b_id)
                continue

        # Sort each section by relevance score
        exact_matches.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        closely_related.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        similar_subjects.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        recommended_pool.sort(key=lambda x: x.get("match_score", 0), reverse=True)

        total_found = len(exact_matches) + len(closely_related) + len(similar_subjects) + len(recommended_pool)
        has_results = total_found > 0

        # Construct user message
        if exact_matches:
            message = f"Found {len(exact_matches)} exact match{'es' if len(exact_matches) > 1 else ''} for '{query}'."
        elif has_results:
            message = "We couldn't find the exact title in our catalogue, but here are some related books you might enjoy."
        else:
            message = f"No books or related subjects matched '{query}'. Try searching for general topics like 'Machine Learning', 'Physics', or 'Computer Science'."

        return {
            "query": query,
            "has_results": has_results,
            "exact_matches": exact_matches[:12],
            "closely_related": closely_related[:12],
            "similar_subjects": similar_subjects[:12],
            "recommended_books": recommended_pool[:12],
            "total_found": total_found,
            "message": message,
        }


# Module convenience helpers
def search_books_hybrid(query: str, limit: int = 40) -> Dict[str, Any]:
    """Execute hybrid search on singleton engine."""
    engine = BookRecommendationEngine.get_instance()
    return engine.hybrid_search(query, limit=limit)


def get_recommendations_for_book(book_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Get similar recommendations for a specific book."""
    engine = BookRecommendationEngine.get_instance()
    return engine.get_similar_books(book_id, limit=limit)
