"""Resources page: academic book discovery and catalogue verification.
Layout:
- Top 50% / 50% Split:
    - Left side top 50%: Bordered Book Search & Requirements Panel
    - Right side top 50%: AI Chatbot (Book Specialist Mode)
- Bottom Full-Width Section:
    - Responsive book-cover shelves with source labels and Primo verification.
"""

import base64
import html
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st
import streamlit.components.v1 as components

from config import (
    COLOR_BORDER,
    COLOR_CARD,
    COLOR_ELEVATED_CARD,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
)
from src.books import kth_primo_search_url, search_books
from src.book_data import get_curated_books
from src.chatbot import render_chatbot_panel

LOCAL_PARQUET = Path(__file__).resolve().parent.parent / "data" / "books" / "books_catalog.parquet"
LOCAL_JSON = Path(__file__).resolve().parent.parent / "data" / "books" / "books_catalog.json"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_books_catalog.json"
COVERS_DIR = Path(__file__).resolve().parent.parent / "data" / "book_covers"


@lru_cache(maxsize=512)
def get_cover_data_uri(book_id: str) -> str:
    """Read local cover JPEG and return as cached base64 data URI."""
    if not book_id:
        return ""
    cover_file = COVERS_DIR / f"{book_id}.jpg"
    if not cover_file.exists():
        cover_file = COVERS_DIR / "placeholder.jpg"
    if cover_file.exists():
        try:
            with open(cover_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return ""
    return ""


def load_category_books() -> Dict[str, List[Dict[str, Any]]]:
    """Load categorized records from the local book database or bundled discovery dataset."""
    if LOCAL_PARQUET.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(LOCAL_PARQUET)
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for _, row in df.iterrows():
                item = row.to_dict()
                cat = item.get("category", "General Reading")
                item["catalogue_status"] = item.get("availability")
                item["source_type"] = "local_catalog_dataset"
                item["availability_live"] = False
                grouped.setdefault(cat, []).append(item)
            return grouped
        except Exception:
            pass

    if LOCAL_JSON.exists():
        try:
            with open(LOCAL_JSON, "r", encoding="utf-8") as f:
                books = json.load(f)
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for b in books:
                item = dict(b)
                cat = item.get("category", "General Reading")
                item["catalogue_status"] = item.get("availability")
                item["source_type"] = "local_catalog_dataset"
                item["availability_live"] = False
                grouped.setdefault(cat, []).append(item)
            return grouped
        except Exception:
            pass

    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                raw_catalog = json.load(f)
            normalized = {}
            for category, books in raw_catalog.items():
                normalized[category] = []
                for book in books:
                    item = dict(book)
                    item["catalogue_status"] = item.get("availability")
                    item["source_type"] = "local_catalog_dataset"
                    item["availability_live"] = False
                    normalized[category].append(item)
            return normalized
        except Exception:
            pass

    curated = get_curated_books()
    grouped = {}
    for b in curated:
        cat = b.get("category", "General Reading")
        grouped.setdefault(cat, []).append(b)
    return grouped


def _render_movie_card_html(book: Dict[str, Any]) -> str:
    # Render explicit catalogue provenance and avoid implying live circulation data.
    b_title = html.escape(str(book.get("title", "Academic Volume")))
    authors_list = book.get("authors", book.get("author_name", [])) or []
    if not authors_list and book.get("author"):
        authors_list = [book["author"]]
    if not isinstance(authors_list, (list, tuple)):
        authors_list = [str(authors_list)]
    b_author = html.escape(", ".join(str(author) for author in authors_list[:1])) or "Author not listed"
    local_record = book.get("source_type") == "local_catalog_dataset" or bool(book.get("catalogue_status")) or bool(book.get("book_id"))
    raw_status = str(book.get("catalogue_status") or book.get("availability") or "")
    status_labels = {
        "Available in Stacks": "Dataset tag · Available in stacks (unverified)",
        "On Reserve": "Dataset tag · On reserve (unverified)",
        "Checked Out": "Dataset tag · Checked out (unverified)",
        "Digital E-Book": "Dataset tag · Digital e-book (unverified)",
    }
    availability_label = status_labels.get(raw_status, "Availability not checked")
    status_class = {
        "Available in Stacks": "available",
        "On Reserve": "reserve",
        "Checked Out": "checked",
        "Digital E-Book": "digital",
    }.get(raw_status, "unknown")
    shelf_value = book.get("shelf") if local_record else None
    shelf_label = (
        f"Dataset shelf label: {html.escape(str(shelf_value))} · verify in KTH Primo"
        if shelf_value else "Shelf location: verify in KTH Primo"
    )
    source_badge = "LOCAL DATASET" if local_record else "OPEN LIBRARY"

    # Recommendation reason badge
    reason = book.get("recommendation_reason")
    reason_badge = ""
    if reason:
        reason_badge = f'<div class="movie-card-reason" title="{html.escape(str(reason))}">{html.escape(str(reason))}</div>'

    # Local cover image resolution
    book_id = book.get("book_id")
    cover_data_uri = get_cover_data_uri(book_id) if book_id else ""
    c_id = book.get("cover_i")
    primo_url = html.escape(kth_primo_search_url(book.get("title", "")))

    if cover_data_uri:
        cover_markup = (
            f'<img src="{cover_data_uri}" class="movie-cover" alt="{b_title}" loading="lazy"/>'
        )
    elif c_id:
        cover_markup = (
            f'<img src="https://covers.openlibrary.org/b/id/{c_id}-M.jpg" '
            f'class="movie-cover" alt="{b_title}" loading="lazy"/>'
        )
    else:
        cover_markup = (
            '<div class="movie-cover-placeholder">'
            '<span style="font-size:2.2rem;">📖</span>'
            '<span style="font-size:.68rem;font-weight:800;margin-top:4px;">BOOK COVER</span>'
            '</div>'
        )

    return f'''<div class="movie-book-card">
      <div class="movie-cover-wrap">
        {cover_markup}
        <div class="movie-badge-top-left">BOOK</div>
        <div class="movie-badge-top-right">{source_badge}</div>
      </div>
      <div class="movie-card-info">
        {reason_badge}
        <div class="movie-card-title" title="{b_title}">{b_title}</div>
        <div class="movie-card-author" title="{b_author}">{b_author}</div>
        <div class="movie-card-status">
          <span class="movie-status-pill status-{status_class}">{availability_label}</span>
        </div>
        <div class="movie-card-shelf" title="{shelf_label}">📍 {shelf_label}</div>
        <a href="{primo_url}" target="_blank" rel="noopener noreferrer" class="movie-primo-btn">Verify live holdings ↗</a>
      </div>
    </div>'''






def _render_resources_top_section(bundle, all_books_flat):
    # =========================================================================
    # TOP SECTION: BALANCED TWO-COLUMN DESKTOP (SEARCH & COMPOSER + AI SPECIALIST)
    # =========================================================================
    top_left_col, top_right_col = st.columns([0.55, 0.45], gap="small")

    # LEFT SIDE TOP: BORDERED BOOK SEARCH & REQUIREMENTS
    with top_left_col:
        with st.container(key="resources_search_panel", border=True):
            st.markdown(
                """
                <div style="margin-bottom: 1.15rem;">
                  <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                      <span style="background: rgba(0,208,132,0.18); color: var(--green); font-size: 0.78rem; font-weight: 800; padding: 3px 9px; border-radius: 9999px; text-transform: uppercase; border: 1px solid var(--green);">
                        Book Discovery
                      </span>
                      <span style="font-size: 0.85rem; color: var(--green);">KTH Primo verification · Open Library discovery</span>
                    </div>
                    <span style="background: var(--surface); border: 1px solid var(--line); color: var(--green); font-size: 0.76rem; padding: 3px 8px; border-radius: 6px;">
                      📚 330+ Curated Catalogue Records
                    </span>
                  </div>
                  <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.25;">
                    Discover Library Stacks & Digital Reserves
                  </h1>
                  <p style="font-size: 1rem; color: var(--muted); margin: 0; line-height: 1.55;">
                    Explore academic titles, filter catalogue metadata, review recorded shelf labels, and verify current copy availability in KTH Primo.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 1. Title or keyword
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--green); margin-bottom: 0.25rem;'>1. Book Title or Keyword</div>", unsafe_allow_html=True)
            book_query = st.text_input(
                "Book title or keyword",
                placeholder="e.g. Deep Learning, Algorithms, Quantum Physics, Macroeconomics...",
                label_visibility="collapsed",
                key="res_book_query",
            )

            # 2. Author name
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--green); margin-top: 0.65rem; margin-bottom: 0.25rem;'>2. Author Name (Optional)</div>", unsafe_allow_html=True)
            author_query = st.text_input(
                "Author name",
                placeholder="e.g. Knuth, Russell, Feynman, Goodfellow, Sutton...",
                label_visibility="collapsed",
                key="res_author_query",
            )

            # 3. Subject / Category
            st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--green); margin-top: 0.65rem; margin-bottom: 0.25rem;'>3. Subject / Discipline</div>", unsafe_allow_html=True)
            cat_options = [
                "Computer Science & AI",
                "Mathematics & Statistics",
                "Physics & Quantum",
                "Robotics & Engineering",
                "Economics & Management",
                "Agriculture & Biosystems",
            ]
            selected_cats = st.multiselect(
                "Subject category",
                options=cat_options,
                default=[],
                label_visibility="collapsed",
                key="res_cat_multiselect",
            )

            # 4. Language & Book Type
            lang_col, type_col = st.columns(2, gap="small")
            with lang_col:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: var(--green); margin-top: 0.6rem; margin-bottom: 0.2rem;'>4. Language</div>", unsafe_allow_html=True)
                pref_lang = st.selectbox(
                    "Language",
                    options=["All Languages", "English", "Swedish", "German", "Unknown language"],
                    index=0,
                    label_visibility="collapsed",
                    key="res_lang_select",
                )
            with type_col:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: var(--green); margin-top: 0.6rem; margin-bottom: 0.2rem;'>5. Format</div>", unsafe_allow_html=True)
                st.caption("Format metadata is not supplied consistently by the current sources.")
                pref_format = "All Formats"

            with st.expander("Advanced filters", expanded=False):
                year_col, availability_col = st.columns(2, gap="small")
                with year_col:
                    publication_year = st.slider(
                        "Publication year", min_value=1450, max_value=2026,
                        value=(1450, 2026), key="res_publication_year",
                    )
                    sort_by = st.selectbox(
                        "Sort results", ["Best match", "Newest first", "Title A–Z"],
                        key="res_sort_by",
                    )
                with availability_col:
                    availability_filter = st.selectbox(
                        "Dataset-listed status (not live availability)",
                        options=[
                            "Any status", "Listed as available in stacks", "On reserve",
                            "Checked out", "Digital e-book", "Availability not verified",
                        ],
                        key="res_availability_filter",
                    )
                    source_filter = st.selectbox(
                        "Record source", ["All sources", "Local catalogue dataset", "Open Library"],
                        key="res_source_filter",
                    )
                st.caption(
                    "Filters use available catalogue metadata. Status labels are from the bundled "
                    "dataset, not a live circulation feed."
                )

            # 6. Additional Requirements
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: var(--green); margin-top: 0.6rem; margin-bottom: 0.2rem;'>6. Additional Requirements</div>", unsafe_allow_html=True)
            extra_reqs = st.text_area(
                "Additional requirements",
                placeholder="e.g. latest edition, includes coding exercises, gentle intro for beginners...",
                label_visibility="collapsed",
                key="res_extra_notes",
                height=65,
            )

            # Prominent Action Button
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Discover & Search Stacks", key="btn_discover_books_top", type="primary", use_container_width=True):
                search_term = book_query.strip() or author_query.strip()
                if not search_term and selected_cats:
                    search_term = selected_cats[0]
                if not search_term:
                    st.warning("Enter a book title, keyword, or author to search the catalogue.")
                    for state_key in (
                        "resource_search_results", "resource_search_error", "resource_search_term", "resource_search_hybrid"
                    ):
                        st.session_state.pop(state_key, None)
                    st.rerun()
                else:
                    with st.spinner(f"Searching catalogue and semantic recommendations for {search_term}..."):
                        from src.book_recommender import search_books_hybrid
                        hybrid_res = search_books_hybrid(search_term)
                        # Flatten results for flat search results compatibility
                        all_found = []
                        seen_ids = set()
                        for group in [
                            hybrid_res.get("exact_matches", []),
                            hybrid_res.get("closely_related", []),
                            hybrid_res.get("similar_subjects", []),
                            hybrid_res.get("recommended_books", []),
                        ]:
                            for b in group:
                                b_id = b.get("book_id", b.get("title"))
                                if b_id not in seen_ids:
                                    seen_ids.add(b_id)
                                    all_found.append(b)
                        found = all_found
                        search_error = None
                    if author_query.strip():
                        author_term = author_query.casefold().strip()
                        author_matches = []
                        for item in found:
                            names = item.get("author_name", item.get("authors", [])) or []
                            if isinstance(names, str):
                                names = [names]
                            if author_term in " ".join(str(name) for name in names).casefold():
                                author_matches.append(item)
                        found = author_matches
                    if selected_cats:
                        found = [
                            item for item in found
                            if item.get("category") in selected_cats
                        ]
                    if publication_year != (1450, 2026):
                        found = [
                            item for item in found
                            if item.get("first_publish_year") is not None
                            and publication_year[0] <= int(item["first_publish_year"]) <= publication_year[1]
                        ]
                    if pref_lang != "All Languages":
                        lang_codes = {
                            "English": {"eng"}, "Swedish": {"swe", "sv"},
                            "German": {"ger", "deu", "de"},
                        }
                        language_filtered = []
                        for item in found:
                            languages = item.get("language", item.get("languages", [])) or []
                            if isinstance(languages, str):
                                languages = [languages]
                            normalized_languages = {str(code).lower() for code in languages}
                            if pref_lang == "Unknown language":
                                if not normalized_languages:
                                    language_filtered.append(item)
                            elif lang_codes[pref_lang].intersection(normalized_languages):
                                language_filtered.append(item)
                        found = language_filtered
                    if pref_format != "All Formats":
                        allowed_status = {
                            "Printed / Physical": {"Available in Stacks", "On Reserve", "Checked Out"},
                            "Digital E-Book": {"Digital E-Book"},
                            "Course Reserve": {"On Reserve"},
                        }[pref_format]
                        found = [item for item in found if item.get("catalogue_status") in allowed_status]
                    if source_filter == "Local catalogue dataset":
                        found = [item for item in found if item.get("source_type") == "local_catalog_dataset"]
                    elif source_filter == "Open Library":
                        found = [item for item in found if item.get("source_type") == "open_library"]

                    if availability_filter != "Any status":
                        status_match = {
                            "Listed as available in stacks": "Available in Stacks",
                            "On reserve": "On Reserve", "Checked out": "Checked Out",
                            "Digital e-book": "Digital E-Book",
                        }.get(availability_filter)
                        if status_match:
                            found = [item for item in found if item.get("catalogue_status") == status_match]
                        else:
                            # Neither source verifies live physical-copy status.
                            found = [item for item in found if not item.get("availability_live", False)]
                    if extra_reqs.strip():
                        requirement_terms = [
                            term.casefold() for term in re.findall(r"[A-Za-z0-9]+", extra_reqs)
                            if len(term) > 2
                        ]
                        def requirement_score(item):
                            authors = item.get("author_name", item.get("authors", [])) or []
                            if isinstance(authors, str):
                                authors = [authors]
                            subjects = item.get("subject", []) or []
                            if isinstance(subjects, str):
                                subjects = [subjects]
                            haystack = " ".join([
                                str(item.get("title", "")),
                                " ".join(str(x) for x in authors),
                                " ".join(str(x) for x in subjects),
                                str(item.get("description", "")),
                                str(item.get("catalogue_status", "")),
                            ]).casefold()
                            return sum(term in haystack for term in requirement_terms)
                        found.sort(key=requirement_score, reverse=True)
                    if sort_by == "Newest first" or "latest" in extra_reqs.casefold():
                        found.sort(key=lambda item: item.get("first_publish_year") or 0, reverse=True)
                    elif sort_by == "Title A–Z":
                        found.sort(key=lambda item: str(item.get("title", "")).casefold())
                    st.session_state.resource_search_results = found
                    st.session_state.resource_search_error = search_error
                    st.session_state.resource_search_term = search_term
                    st.session_state.resource_search_hybrid = hybrid_res
                    if found:
                        st.session_state.scroll_to_results_token = f"search_{time.time()}"
                    else:
                        st.session_state.pop("scroll_to_results_token", None)
                    st.rerun()

            # Availability guide explains the source and limits of the catalogue status labels.
            st.markdown(
                """
                <div class="availability-guide">
                  <div class="availability-guide-title">How to check library availability</div>
                  <div class="availability-guide-copy">
                    Status labels and shelf codes below come from the bundled catalogue dataset,
                    not a live circulation feed. Confirm today's copy status and borrowing options in KTH Primo.
                  </div>
                  <div class="availability-guide-grid">
                    <div><span class="availability-dot available"></span><b>Available in Stacks</b><small>Dataset lists a stack copy</small></div>
                    <div><span class="availability-dot reserve"></span><b>On Reserve</b><small>Dataset marks a reserve copy</small></div>
                    <div><span class="availability-dot checked"></span><b>Checked Out</b><small>Dataset marks it as loaned</small></div>
                    <div><span class="availability-dot digital"></span><b>Digital E-Book</b><small>Dataset labels a digital edition</small></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            primo_query = book_query.strip() or author_query.strip() or (
                selected_cats[0] if selected_cats else ""
            )
            if primo_query:
                st.link_button(
                    "↗ Verify live holdings in KTH Primo",
                    kth_primo_search_url(primo_query),
                    use_container_width=True,
                )

        # RIGHT SIDE TOP 50%: AI CHATBOT (BOOK SPECIALIST MODE)
    with top_right_col:
        render_chatbot_panel(bundle, "Resources")



def render_resources_page(bundle=None):
    """
    Render Book Discovery Platform:
    - Top Left (55%): Bordered Book Search & Requirements Form.
    - Top Right (45%): AI Chatbot in Book Specialist Mode.
    - Bottom: book-cover shelves across 6 disciplines (50+ books each).
    """
    cat_data = load_category_books()
    all_books_flat = get_curated_books()

    _render_resources_top_section(bundle, all_books_flat)

    # =========================================================================
    # BOTTOM FULL-WIDTH SECTION: ACADEMIC BOOK COVER SHELVES (BORDERED BOX)
    # =========================================================================
    with st.container(key="resources_books_panel", border=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid var(--line); padding-bottom: 0.75rem; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.5rem;">
              <div>
                <div style="font-size: 1.6rem; font-weight: 800; color: var(--ink);">🎬 Academic Stacks & Digital Course Reserves</div>
                <div style="font-size: 0.92rem; color: var(--muted);">Book covers and curated catalogue records · Verify current borrowing status in KTH Primo</div>
              </div>
              <span style="background: var(--surface-2); color: var(--green); border: 1px solid var(--green); font-size: 0.78rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                330 Catalogue Records
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Inject book-cover shelf CSS. st.html content is isolated from
        # the app stylesheet, so apply the active palette inside this style block too.
        dark_shelf = bool(st.session_state.get("dark_mode", False))
        shelf_theme_css = (
            """
            <style>
              .movie-book-card { background:#1d2a22 !important; border-color:#2b3b31 !important; box-shadow:0 4px 14px rgba(0,0,0,.22) !important; }
              .movie-cover-wrap, .movie-cover-placeholder { background:#17211b !important; }
              .movie-card-title { color:#edf4ee !important; }
              .movie-card-author, .movie-card-shelf { color:#a0afa5 !important; }
              .movie-primo-btn { background:#17211b !important; border-color:#2b3b31 !important; color:#55c58a !important; }
              .movie-carousel-row { scrollbar-color:#55c58a #1d2a22 !important; }
              .movie-carousel-row::-webkit-scrollbar-track { background:#17211b !important; }
              .movie-carousel-row::-webkit-scrollbar-thumb { background:#2b3b31 !important; }
              .movie-cover-placeholder { background:linear-gradient(180deg,#17211b 0%,#101713 100%) !important; }
            </style>
            """
            if dark_shelf else
            """
            <style>
              .movie-book-card { background:#ffffff !important; border-color:#dfe8e1 !important; box-shadow:0 4px 14px rgba(24,50,35,.06) !important; }
              .movie-cover-wrap, .movie-cover-placeholder { background:#edf3ee !important; }
              .movie-card-title { color:#17251e !important; }
              .movie-card-author, .movie-card-shelf { color:#718078 !important; }
              .movie-primo-btn { background:#edf3ee !important; border-color:#dfe8e1 !important; color:#1d6949 !important; }
              .movie-carousel-row { scrollbar-color:#1d6949 #edf3ee !important; }
              .movie-carousel-row::-webkit-scrollbar-track { background:#edf3ee !important; }
              .movie-carousel-row::-webkit-scrollbar-thumb { background:#c8d8cd !important; }
              .movie-cover-placeholder { background:linear-gradient(180deg,#ffffff 0%,#edf3ee 100%) !important; }
            </style>
            """
        )
        st.html(
            """
            <style>
            .movie-carousel-row {
                display: flex;
                flex-wrap: nowrap;
                overflow-x: auto;
                gap: 16px;
                padding: 8px 4px 18px 4px;
                scrollbar-width: thin;
                scrollbar-color: var(--green) var(--paper);
                -webkit-overflow-scrolling: touch;
            }
            .movie-carousel-row::-webkit-scrollbar {
                height: 8px;
            }
            .movie-carousel-row::-webkit-scrollbar-track {
                background: var(--paper);
                border-radius: 4px;
            }
            .movie-carousel-row::-webkit-scrollbar-thumb {
                background: var(--line);
                border-radius: 4px;
            }
            .movie-carousel-row::-webkit-scrollbar-thumb:hover {
                background: var(--green);
            }
            .movie-book-card {
                flex: 0 0 170px;
                width: 170px;
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 12px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
            }
            .movie-book-card:hover {
                transform: translateY(-6px) scale(1.02);
                border-color: var(--green);
                box-shadow: 0 12px 28px rgba(0, 208, 132, 0.25);
            }
            .movie-cover-wrap {
                position: relative;
                width: 100%;
                height: 220px;
                background: var(--paper);
                overflow: hidden;
            }
            .movie-cover {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
                transition: transform 0.3s ease;
            }
            .movie-book-card:hover .movie-cover {
                transform: scale(1.05);
            }
            .movie-cover-placeholder {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: linear-gradient(180deg, var(--surface) 0%, var(--paper) 100%);
                border-bottom: 1px solid var(--line);
            }
            .movie-badge-top-left {
                position: absolute;
                top: 8px;
                left: 8px;
                background: rgba(8, 11, 9, 0.85);
                backdrop-filter: blur(4px);
                color: #FFD25F;
                font-size: 0.72rem;
                font-weight: 800;
                padding: 2px 7px;
                border-radius: 6px;
                border: 1px solid rgba(255, 210, 95, 0.4);
            }
            .movie-badge-top-right {
                position: absolute;
                top: 8px;
                right: 8px;
                background: rgba(0, 208, 132, 0.85);
                color: var(--paper);
                font-size: 0.68rem;
                font-weight: 900;
                padding: 2px 6px;
                border-radius: 4px;
                letter-spacing: 0.05em;
            }
            .movie-card-info {
                padding: 10px;
                display: flex;
                flex-direction: column;
                flex-grow: 1;
                justify-content: space-between;
            }
            .movie-card-title {
                font-size: 0.85rem;
                font-weight: 700;
                color: var(--ink);
                line-height: 1.3;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
                margin-bottom: 3px;
            }
            .movie-card-author {
                font-size: 0.74rem;
                color: var(--green);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                margin-bottom: 6px;
            }
            .movie-card-status {
                margin-bottom: 4px;
            }
            .movie-status-pill {
                display:inline-block;
                max-width:100%;
                white-space:normal;
                font-size: 0.61rem;
                line-height:1.35;
                font-weight: 800;
                padding: 3px 6px;
                border-radius: 5px;
                letter-spacing: 0.01em;
            }
            .status-available { background:rgba(34,151,101,.12); color:#229765; border:1px solid rgba(34,151,101,.45); }
            .status-reserve { background:rgba(212,154,54,.13); color:#a66e12; border:1px solid rgba(212,154,54,.5); }
            .status-checked { background:rgba(198,93,93,.12); color:#bd4e4e; border:1px solid rgba(198,93,93,.45); }
            .status-digital { background:rgba(77,145,199,.12); color:#397fb5; border:1px solid rgba(77,145,199,.45); }
            .status-unknown { background:rgba(120,128,120,.1); color:#718078; border:1px solid rgba(120,128,120,.35); }
            .movie-card-shelf {
                font-size: 0.7rem;
                color: var(--muted);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                margin-bottom: 8px;
            }
            .movie-card-reason {
                font-size: 0.65rem;
                font-weight: 750;
                color: var(--green);
                background: rgba(0, 208, 132, 0.12);
                border: 1px solid rgba(0, 208, 132, 0.35);
                padding: 2px 6px;
                border-radius: 4px;
                margin-bottom: 5px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                letter-spacing: 0.02em;
            }
            .movie-primo-btn {
                display: block;
                text-align: center;
                background: var(--surface-2);
                border: 1px solid var(--line);
                color: var(--green) !important;
                font-size: 0.72rem;
                font-weight: 700;
                padding: 5px 8px;
                border-radius: 6px;
                text-decoration: none !important;
                transition: all 0.2s ease;
            }
            .movie-primo-btn:hover {
                background: var(--green);
                color: var(--paper) !important;
                border-color: var(--green);
            }
            </style>
            """ + shelf_theme_css
        )

        # Search results belong in this carousel area (the blue-arrow location),
        # never between the search form and the shelves.
        search_active = "resource_search_term" in st.session_state
        if search_active:
            # Search results are rendered in the first full-width shelf below.
            st.markdown('<div id="book-search-results-section"></div><div id="book-search-results-anchor"></div>', unsafe_allow_html=True)
            search_term = str(st.session_state.get("resource_search_term", ""))
            search_books_found = st.session_state.get("resource_search_results", []) or []
            hybrid_res = st.session_state.get("resource_search_hybrid") or {}

            exact_matches = hybrid_res.get("exact_matches", [])
            closely_related = hybrid_res.get("closely_related", [])
            similar_subjects = hybrid_res.get("similar_subjects", [])
            recommended_books = hybrid_res.get("recommended_books", [])

            st.markdown(f'<div class="book-search-results-heading"><h3>Search results for: {html.escape(search_term)}</h3></div>', unsafe_allow_html=True)
            info_col, clear_col = st.columns([5, 1], vertical_alignment="center")
            with info_col:
                if search_books_found:
                    st.caption(
                        f"{len(search_books_found)} matching titles · Scroll the shelves below to browse covers. "
                        "Confirm physical availability through KTH Primo."
                    )
                else:
                    st.info("No matching titles were found. Try a broader keyword or use the Primo catalogue link.")
                    st.link_button("Search KTH Primo", kth_primo_search_url(search_term))
            with clear_col:
                if st.button("Clear search", key="clear_resource_search"):
                    for state_key in (
                        "resource_search_results", "resource_search_error", "resource_search_term", "resource_search_hybrid", "scroll_to_results_token"
                    ):
                        st.session_state.pop(state_key, None)
                    st.rerun()

            # If exact matches are missing but related books exist:
            if not exact_matches and (closely_related or similar_subjects or recommended_books):
                st.info("We couldn't find the exact title in our catalogue, but here are some related books you might enjoy.")

            first_shelf = (f"Search results · {search_term}", search_books_found[:55])

            # Build 4 categorized shelves under active search
            movie_rows = []
            if exact_matches:
                movie_rows.append(("🎯 1. Exact Matches", exact_matches))
            if closely_related:
                movie_rows.append(("✨ 2. Closely Related Books", closely_related))
            if similar_subjects:
                movie_rows.append(("📚 3. Similar Subjects & Disciplines", similar_subjects))
            if recommended_books:
                movie_rows.append(("💡 4. Recommended Books (AI Semantic Discovery)", recommended_books))

            # Fallback if no specific shelf matched but search_books_found has items
            if not movie_rows and search_books_found:
                movie_rows.append(first_shelf)
        else:
            first_shelf = ("Recommended Academic Stacks & Course Reserves", all_books_flat[:55])
            movie_rows = [
                first_shelf,
                ("Computer Science, Algorithms & AI", cat_data.get("Computer Science & AI", all_books_flat[:55])),
                ("Mathematics, Statistics & Discrete Logic", cat_data.get("Mathematics & Statistics", all_books_flat[55:110])),
                ("Physics, Quantum Mechanics & Materials", cat_data.get("Physics & Quantum", all_books_flat[110:165])),
                ("Robotics, Autonomous Systems & Control Engineering", cat_data.get("Robotics & Engineering", all_books_flat[165:220])),
                ("Economics, Financial Engineering & Decision Science", cat_data.get("Economics & Management", all_books_flat[220:275])),
                ("Agriculture & Biosystems", cat_data.get("Agriculture & Biosystems", [])),
            ]

        for idx, (row_title, books_list) in enumerate(movie_rows):
            count = len(books_list)
            st.markdown(
                f"""
                <div style="display: flex; align-items: baseline; justify-content: space-between; margin-top: 1.25rem; margin-bottom: 0.45rem;">
                  <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span style="font-size: 1.25rem; font-weight: 800; color: var(--ink);">{row_title}</span>
                    <span style="background: var(--surface-2); color: var(--muted); border: 1px solid var(--line); font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;">
                      {count} Volumes
                    </span>
                  </div>
                  <span style="font-size: 0.8rem; color: var(--green); font-weight: 700;">Swipe & Explore ➔</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cards_html_list = [_render_movie_card_html(b) for b in books_list]
            st.html(f'<div class="movie-carousel-row">{"".join(cards_html_list)}</div>')

            # Once the search results shelf has rendered, smooth-scroll into view if triggered by a new search
            if idx == 0 and search_active:
                scroll_token = st.session_state.pop("scroll_to_results_token", None)
                if scroll_token:
                    components.html(
                        """
                        <script>
                        (() => {
                          const scrollToSection = () => {
                            try {
                              const doc = window.parent.document;
                              const target = doc.getElementById("book-search-results-section")
                                || doc.getElementById("book-search-results-anchor")
                                || doc.querySelector('[class*="st-key-resources_books_panel"]');
                              if (target) {
                                target.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" });
                              }
                            } catch (e) {
                              try {
                                const target = document.getElementById("book-search-results-section");
                                if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
                              } catch (err) {}
                            }
                          };
                          if (window.parent && window.parent.requestAnimationFrame) {
                            window.parent.requestAnimationFrame(() => {
                              setTimeout(scrollToSection, 80);
                              setTimeout(scrollToSection, 300);
                            });
                          } else {
                            setTimeout(scrollToSection, 120);
                          }
                        })();
                        </script>
                        """,
                        height=0,
                        scrolling=False,
                    )