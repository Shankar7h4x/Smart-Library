"""Book discovery using bundled records and Open Library bibliographic metadata.

Bundled records are not proof of current KTH holdings or physical-copy availability.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlencode

import requests
import streamlit as st

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
BOOK_FIELDS = (
    "key,title,author_name,first_publish_year,edition_count,cover_i,"
    "want_to_read_count,currently_reading_count,availability,language,subject"
)

# Load the bundled academic catalogue dataset (330 records across 6 disciplines)
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_books_catalog.json"
CURATED_LIBRARY_BOOKS = []

if CATALOG_PATH.exists():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            _catalog_data = json.load(f)
        for _cname, _b_list in _catalog_data.items():
            for _b in _b_list:
                _avail_str = _b.get("availability", "Available")
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

# Missing catalogue files produce an empty result; do not fabricate fallback titles.\n\n
def get_curated_books(category: str = None) -> list:
    """Return local curated academic library books, optionally filtered by category."""
    if not category or category == "All":
        return list(CURATED_LIBRARY_BOOKS)
    return [b for b in CURATED_LIBRARY_BOOKS if b.get("category", "").lower() == category.lower()]


def _match_local_books(query: str) -> list:
    """Fuzzy/substring search in local curated campus reserves."""
    q = query.strip().casefold()
    if not q:
        return list(CURATED_LIBRARY_BOOKS)

    matches = []
    for b in CURATED_LIBRARY_BOOKS:
        title = b.get("title", "").casefold()
        authors_raw = b.get("author_name", []) or []
        if not isinstance(authors_raw, (list, tuple)):
            authors_raw = [authors_raw]
        authors = " ".join(str(a) for a in authors_raw).casefold()
        category = b.get("category", "").casefold()
        subjects_raw = b.get("subject", []) or []
        if not isinstance(subjects_raw, (list, tuple)):
            subjects_raw = [subjects_raw]
        subjects = " ".join(str(s) for s in subjects_raw).casefold()
        desc = b.get("description", "").casefold()

        # Score relevance
        if title.startswith(q):
            score = 100
        elif q in title:
            score = 80
        elif any(a.startswith(q) or q in a for a in b.get("author_name", [])):
            score = 60
        elif q in category or q in subjects or q in desc:
            score = 40
        elif all(term in f"{title} {authors} {category} {subjects} {desc}" for term in q.split() if len(term) > 1):
            score = 25
        else:
            score = 0

        if score > 0:
            matches.append((score, b))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in matches]


def search_local_books(query: str, limit: int = 40) -> list:
    """Search the bundled catalogue without making a network request."""
    return _match_local_books(query)[:max(0, int(limit))]


@st.cache_data(ttl=900, show_spinner=False)
def search_book_title_prefix(prefix: str, limit: int = 50):
    """Search local reserves and Open Library with forgiving title/author matching."""
    query = (prefix or "").strip()
    if not query:
        return list(CURATED_LIBRARY_BOOKS[:limit]), len(CURATED_LIBRARY_BOOKS), None

    safe_query = re.sub(r"[^\w\s'-]", " ", query, flags=re.UNICODE).strip()
    if not safe_query:
        return [], 0, None

    local_matches = _match_local_books(safe_query)
    online_matches, error = [], None
    try:
        params = {"q": safe_query, "limit": min(max(int(limit) * 2, 20), 100), "fields": BOOK_FIELDS}
        response = requests.get(
            OPEN_LIBRARY_SEARCH, params=params,
            headers={"User-Agent": "SmartLibraryIntelligence/2.0 (academic discovery)"}, timeout=4,
        )
        response.raise_for_status()
        docs = response.json().get("docs", [])
        terms = [t.casefold() for t in safe_query.split() if len(t) > 1]
        for doc in docs:
            title = str(doc.get("title", ""))
            authors = " ".join(doc.get("author_name", []) or [])
            subjects = " ".join(str(s) for s in (doc.get("subject", []) or []))
            haystack = f"{title} {authors} {subjects}".casefold()
            title_norm = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
            # Require the meaningful query words in title or author, but allow punctuation
            # and token spacing differences (e.g. Harry-Potter vs Harry Potter).
            if terms and not all(term in haystack for term in terms):
                continue
            doc["_search_title"] = title_norm
            online_matches.append(doc)
    except Exception as exc:
        error = "Live Open Library search is temporarily unavailable. Showing matching local reserve titles where available."

    seen, merged = set(), []
    for book in local_matches + online_matches:
        key = book.get("key") or (str(book.get("title", "")).casefold(), tuple(book.get("author_name", [])))
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
    query_terms = [term.casefold() for term in safe_query.split() if len(term) > 1]
    def relevance(book):
        title = str(book.get("title", "")).casefold()
        authors = book.get("author_name", []) or []
        if not isinstance(authors, (list, tuple)):
            authors = [authors]
        author_text = " ".join(str(author) for author in authors).casefold()
        title_words = set(re.findall(r"[a-z0-9]+", title))
        term_hits = sum(term in title_words for term in query_terms)
        author_hits = sum(term in author_text for term in query_terms)
        exact = int(title == safe_query.casefold())
        prefix = int(title.startswith(safe_query.casefold()))
        overlap = term_hits / max(1, len(query_terms))
        return (exact, prefix, overlap, author_hits, int(book.get("source_type") == "local_catalog_dataset"))
    merged.sort(key=relevance, reverse=True)
    return merged[:limit], len(merged), (error if not merged else None)


@st.cache_data(ttl=1800, show_spinner=False)
def search_books(query: str):
    """Search Open Library and local catalog for a human-entered query."""
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
