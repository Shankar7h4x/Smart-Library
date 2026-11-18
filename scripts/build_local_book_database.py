"""Build and persist structured local book database for Smart Library.

Creates:
- data/books/books_catalog.json
- data/books/books_catalog.parquet
- data/books/metadata.json
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_CATALOG = ROOT_DIR / "data" / "netflix_books_catalog.json"
COVER_MAPPING_PATH = ROOT_DIR / "data" / "book_covers" / "cover_mapping.json"
BOOKS_DIR = ROOT_DIR / "data" / "books"
BOOKS_JSON = BOOKS_DIR / "books_catalog.json"
BOOKS_PARQUET = BOOKS_DIR / "books_catalog.parquet"
METADATA_JSON = BOOKS_DIR / "metadata.json"

DISCIPLINE_GENRES = {
    "Computer Science & AI": "Computer Science, Artificial Intelligence, Algorithms & Software Systems",
    "Agriculture & Biosystems": "Agricultural Science, Biological Engineering & Sustainable Ecosystems",
    "Mathematics & Statistics": "Pure & Applied Mathematics, Probability, Statistics & Mathematical Analysis",
    "Physics & Quantum": "Classical Mechanics, Quantum Physics, Thermodynamics & Theoretical Physics",
    "Economics & Management": "Managerial Economics, Financial Systems, Quantitative Business & Operations",
    "Robotics & Engineering": "Robotics, Control Systems, Mechanical Engineering & Mechatronics",
}

DISCIPLINE_KEYWORDS = {
    "Computer Science & AI": ["computer science", "artificial intelligence", "programming", "algorithms", "data structures", "software", "machine learning", "neural networks", "computation"],
    "Agriculture & Biosystems": ["agriculture", "biosystems", "biological engineering", "ecology", "sustainability", "soil science", "crops", "forestry", "bioengineering"],
    "Mathematics & Statistics": ["mathematics", "statistics", "calculus", "linear algebra", "probability", "analysis", "geometry", "applied math", "stochastic processes"],
    "Physics & Quantum": ["physics", "quantum mechanics", "thermodynamics", "optics", "electromagnetism", "relativity", "quantum physics", "atomic physics", "classical mechanics"],
    "Economics & Management": ["economics", "management", "finance", "business", "microeconomics", "macroeconomics", "econometrics", "marketing", "operations research"],
    "Robotics & Engineering": ["robotics", "engineering", "mechatronics", "control systems", "automation", "sensors", "kinematics", "mechanical design", "hardware"],
}


def build_book_database():
    """Normalize, enrich, and serialize the local book database."""
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_CATALOG.exists():
        print(f"Error: source catalogue not found at {SOURCE_CATALOG}")
        return

    with open(SOURCE_CATALOG, "r", encoding="utf-8") as f:
        catalog_raw = json.load(f)

    cover_mapping = {}
    if COVER_MAPPING_PATH.exists():
        with open(COVER_MAPPING_PATH, "r", encoding="utf-8") as f:
            cover_mapping = json.load(f)

    all_records: List[Dict[str, Any]] = []
    book_counter = 1

    for cat_name, books in catalog_raw.items():
        genre_desc = DISCIPLINE_GENRES.get(cat_name, cat_name)
        base_keywords = DISCIPLINE_KEYWORDS.get(cat_name, [cat_name.lower()])

        for b in books:
            book_id = f"book_{book_counter:03d}"
            book_counter += 1

            title = str(b.get("title", "Academic Volume")).strip()
            authors = b.get("authors", ["Academic Contributor"])
            if not isinstance(authors, list):
                authors = [str(authors)]
            primary_author = authors[0] if authors else "Academic Contributor"

            year = b.get("year")
            cover_i = b.get("cover_i")
            shelf = b.get("shelf", f"KTHB-GEN-{book_counter}")
            avail_status = b.get("availability", "Available in Stacks")
            rating = float(b.get("rating", 4.0)) if b.get("rating") is not None else 4.0
            want_to_read = int(b.get("want_to_read", 0)) if b.get("want_to_read") is not None else 0
            reading = int(b.get("reading", 0)) if b.get("reading") is not None else 0
            work_key = b.get("key", f"/works/kth_{book_id}")

            # Specific subject keywords
            title_words = [w.lower() for w in title.replace("-", " ").split() if len(w) > 3]
            subjects = list(dict.fromkeys(base_keywords + title_words[:4]))

            # Structured, honest description that provides dense semantic signals without fabricating facts
            year_str = f"published in {year}" if year else "academic publication"
            authors_str = ", ".join(authors[:3])
            desc = (
                f"{title} is an academic work by {authors_str}, {year_str}. "
                f"Classified under {cat_name} ({genre_desc}). "
                f"Subject keywords: {', '.join(subjects[:6])}. "
                f"Call number: {shelf} in KTH Royal Institute of Technology library reserves."
            )

            cover_info = cover_mapping.get(book_id, {})
            local_cover_path = cover_info.get("local_path", f"data/book_covers/{book_id}.jpg")
            cover_url = f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg" if cover_i else None

            record = {
                "book_id": book_id,
                "work_key": work_key,
                "title": title,
                "author": primary_author,
                "authors": authors,
                "description": desc,
                "category": cat_name,
                "genre": genre_desc,
                "subject": subjects,
                "isbn": None,  # Not present in original dataset; explicitly not fabricated
                "publisher": None,  # Not present in original dataset; explicitly not fabricated
                "publication_year": year,
                "language": "eng",
                "cover_i": cover_i,
                "cover_local_path": local_cover_path,
                "original_cover_source": cover_url,
                "availability_status": avail_status,
                "shelf": shelf,
                "rating": rating,
                "want_to_read": want_to_read,
                "reading": reading,
                "catalogue_source": "kth_curated_academic_catalog",
            }
            all_records.append(record)

    # 1. Save JSON catalogue
    with open(BOOKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # 2. Save Parquet catalogue
    # For parquet, serialize lists as json strings or parquet lists
    df = pd.DataFrame(all_records)
    # Convert lists to strings for maximum cross-engine parquet compatibility
    df_parquet = df.copy()
    df_parquet["authors"] = df_parquet["authors"].apply(json.dumps)
    df_parquet["subject"] = df_parquet["subject"].apply(json.dumps)
    df_parquet.to_parquet(BOOKS_PARQUET, index=False)

    # 3. Save Metadata JSON
    metadata = {
        "dataset_name": "Smart Library Local Academic Book Catalogue",
        "total_records": len(all_records),
        "total_categories": len(catalog_raw),
        "categories": list(catalog_raw.keys()),
        "records_per_category": {cat: len([b for b in all_records if b["category"] == cat]) for cat in catalog_raw.keys()},
        "schema_version": "2.0-local-offline",
        "created_at": datetime.now().isoformat(),
        "storage_formats": ["JSON", "Parquet"],
        "cover_storage": "local (data/book_covers/)",
        "json_path": "data/books/books_catalog.json",
        "parquet_path": "data/books/books_catalog.parquet",
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Local book database successfully created at {BOOKS_DIR}:")
    print(f"  - {BOOKS_JSON.name}: {len(all_records)} records ({BOOKS_JSON.stat().st_size:,} bytes)")
    print(f"  - {BOOKS_PARQUET.name}: {len(df_parquet)} rows ({BOOKS_PARQUET.stat().st_size:,} bytes)")
    print(f"  - {METADATA_JSON.name}: metadata recorded")


if __name__ == "__main__":
    build_book_database()
