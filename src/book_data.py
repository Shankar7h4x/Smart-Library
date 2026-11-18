"""Book discovery using local academic database, vector embeddings and Open Library metadata.

Bundled records are not proof of current KTH holdings or physical-copy availability.
Normal searches execute locally without requiring network connectivity.
"""

import json
from pathlib import Path
import re
from urllib.parse import urlencode

import requests
import streamlit as st

from src.book_recommender import BookRecommendationEngine

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
BOOK_FIELDS = (
    "key,title,author_name,first_publish_year,edition_count,cover_i,"
    "want_to_read_count,currently_reading_count,availability,language,subject"
)

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_books_catalog.json"
LOCAL_BOOKS_PARQUET = Path(__file__).resolve().parent.parent / "data" / "books" / "books_catalog.parquet"
LOCAL_BOOKS_JSON = Path(__file__).resolve().parent.parent / "data" / "books" / "books_catalog.json"

CURATED_LIBRARY_BOOKS = []

# Load local curated records into CURATED_LIBRARY_BOOKS for full backward compatibility
if LOCAL_BOOKS_JSON.exists():
    try:
        with open(LOCAL_BOOKS_JSON, "r", encoding="utf-8") as f:
            _local_records = json.load(f)
        for _b in _local_records:
            _avail_str = _b.get("availability_status", "Available in Stacks")
            _status_str = (
                "borrow_available" if _avail_str == "Available in Stacks"
                else "open" if _avail_str == "Digital E-Book"
                else "borrow_unavailable" if _avail_str in {"On Reserve", "Checked Out"}
                else "unknown"
            )
            CURATED_LIBRARY_BOOKS.append({
                "book_id": _b.get("book_id"),
                "key": _b.get("work_key", _b.get("key")),
                "title": _b.get("title"),
                "author_name": _b.get("authors", [_b.get("author", "Academic Contributor")]),
                "authors": _b.get("authors", [_b.get("author", "Academic Contributor")]),
                "first_publish_year": _b.get("publication_year"),
                "year": _b.get("publication_year"),
                "edition_count": 1,
                "cover_i": _b.get("cover_i"),
                "cover_local_path": _b.get("cover_local_path"),
                "want_to_read_count": _b.get("want_to_read"),
                "currently_reading_count": _b.get("reading"),
                "availability": {"status": _status_str},
                "catalogue_status": _avail_str,
                "availability_live": False,
                "source_type": "local_catalog_dataset",
                "category": _b.get("category"),
                "genre": _b.get("genre"),
                "subject": _b.get("subject", []),
                "rating": _b.get("rating"),
                "shelf": _b.get("shelf"),
                "description": _b.get("description"),
            })
    except Exception:
        pass
elif CATALOG_PATH.exists():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            _catalog_data = json.load(f)
        for _cname, _b_list in _catalog_data.items():
            for _b in _b_list:
                _avail_str = _b.get("availability", "Available in Stacks")
                _status_str = (
                    "borrow_available" if _avail_str == "Available in Stacks"
                    else "open" if _avail_str == "Digital E-Book"
                    else "borrow_unavailable" if _avail_str in {"On Reserve", "Checked Out"}
                    else "unknown"
                )
                CURATED_LIBRARY_BOOKS.append({
                    "key": _b["key"],
                    "title": _b["title"],
                    "author_name": _b.get("authors", ["Academic Contributor"]),
                    "first_publish_year": _b.get("year"),
                    "edition_count": 1,
                    "cover_i": _b.get("cover_i"),
                    "want_to_read_count": _b.get("want_to_read"),
                    "currently_reading_count": _b.get("reading"),
                    "availability": {"status": _status_str},
                    "catalogue_status": _avail_str,
                    "availability_live": False,
                    "source_type": "local_catalog_dataset",
                    "category": _cname,
                    "rating": _b.get("rating"),
                    "shelf": _b.get("shelf"),
                    "description": f"{_b['title']} — local catalogue dataset record; verify current holdings in KTH Primo.",
                })
    except Exception:
        pass


def get_curated_books(category: str = None) -> list:
    """Return local curated academic library books, optionally filtered by category."""
    if not category or category.casefold() in {"all", "all books"}:
        return list(CURATED_LIBRARY_BOOKS)
    return [b for b in CURATED_LIBRARY_BOOKS if b.get("category", "").casefold() == category.casefold()]


def search_local_books(query: str, limit: int = 40) -> list:
    """Search the local database using the hybrid recommendation engine."""
    engine = BookRecommendationEngine.get_instance()
    res = engine.hybrid_search(query, limit=limit)
    combined = res["exact_matches"] + res["closely_related"] + res["similar_subjects"] + res["recommended_books"]
    if combined:
        return combined[:limit]

    # Substring fallback
    q = query.strip().casefold()
    if not q:
        return list(CURATED_LIBRARY_BOOKS[:limit])
    matches = [b for b in CURATED_LIBRARY_BOOKS if q in str(b.get("title", "")).casefold() or q in " ".join(str(a) for a in b.get("author_name", [])).casefold()]
    return matches[:limit]


@st.cache_data(ttl=900, show_spinner=False)
def search_book_title_prefix(prefix: str, limit: int = 50):
    """Search local catalog first; forgiving fallback to Open Library if online."""
    query = (prefix or "").strip()
    if not query:
        return list(CURATED_LIBRARY_BOOKS[:limit]), len(CURATED_LIBRARY_BOOKS), None

    safe_query = re.sub(r"[^\w\s'-]", " ", query, flags=re.UNICODE).strip()
    if not safe_query:
        return [], 0, None

    # Step 1: Query local recommendation engine
    local_matches = search_local_books(safe_query, limit=limit)

    # Step 2: If local matches exist or offline, return immediately
    online_matches, error = [], None
    try:
        params = {"q": safe_query, "limit": min(max(int(limit) * 2, 20), 100), "fields": BOOK_FIELDS}
        response = requests.get(
            OPEN_LIBRARY_SEARCH, params=params,
            headers={"User-Agent": "SmartLibraryIntelligence/2.0 (academic discovery)"}, timeout=2,
        )
        if response.status_code == 200:
            docs = response.json().get("docs", [])
            terms = [t.casefold() for t in safe_query.split() if len(t) > 1]
            for doc in docs:
                title = str(doc.get("title", ""))
                authors = " ".join(doc.get("author_name", []) or [])
                subjects = " ".join(str(s) for s in (doc.get("subject", []) or []))
                haystack = f"{title} {authors} {subjects}".casefold()
                title_norm = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
                if terms and not all(term in haystack for term in terms):
                    continue
                doc["_search_title"] = title_norm
                online_matches.append(doc)
    except Exception:
        # Offline mode active; zero error shown when local matches exist
        if not local_matches:
            error = "Live Open Library search is temporarily unavailable. Showing matching local reserve titles where available."

    seen, merged = set(), []
    for book in local_matches + online_matches:
        key = book.get("book_id") or book.get("key") or (str(book.get("title", "")).casefold(), tuple(book.get("author_name", [])))
        if key not in seen:
            seen.add(key)
            enriched = dict(book)
            if book in local_matches:
                enriched.setdefault("source_type", "local_catalog_dataset")
                enriched.setdefault("availability_live", False)
            else:
                enriched["source_type"] = "open_library"
                enriched["availability_live"] = False
                enriched.setdefault("catalogue_status", None)
            merged.append(enriched)

    return merged[:limit], len(merged), (error if not merged else None)


@st.cache_data(ttl=1800, show_spinner=False)
def search_books(query: str):
    """Search local catalog and Open Library for a human-entered query."""
    docs, count, err = search_book_title_prefix(query, limit=40)
    return docs, err


def kth_primo_search_url(query: str) -> str:
    """Build a public KTH Primo catalog search link for physical-copy availability."""
    params = urlencode({
        "query": f"any,contains,{query.strip()}",
        "tab": "Everything",
        "search_scope": "MyInst_and_CI",
        "vid": "46KTH_INST:46KTH_VU1_L",
        "lang": "en",
    })
    return f"https://kth-ch.primo.exlibrisgroup.com/discovery/search?{params}"
