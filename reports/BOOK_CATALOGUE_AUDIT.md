# Smart Library: Book Catalogue Audit Report

**Audit Date:** November 18, 2026  
**Auditor:** Senior Data Engineer & Recommendation Systems Architect  
**Source Dataset:** `data/netflix_books_catalog.json`  
**Target Destination:** `data/books/books_catalog.json` & `data/books/books_catalog.parquet`

---

## 1. Executive Summary

A comprehensive inventory and audit of the existing Smart Library book catalogue was performed prior to database migration and local recommendation engine deployment.

The catalogue contains **330 bundled academic records** curated across 6 scientific disciplines at KTH Royal Institute of Technology:
1. `Computer Science & AI` (55 titles)
2. `Agriculture & Biosystems` (55 titles)
3. `Mathematics & Statistics` (55 titles)
4. `Physics & Quantum` (55 titles)
5. `Economics & Management` (55 titles)
6. `Robotics & Engineering` (55 titles)

---

## 2. Quantitative Inventory Metrics

| Audit Metric | Observed Count | Percentage of Total | Notes |
|---|---|---|---|
| **Total Catalogue Records** | **330** | 100.0% | 55 records per category |
| **Unique Open Library Keys** | **327** | 99.1% | 3 books appear across 2 interdisciplinary categories |
| **Unique Book Titles** | **275** | 83.3% | Multiple editions / cross-category entries |
| **Records with Author Information** | **330** | 100.0% | Zero missing author fields |
| **Records with Available Year** | **330** | 100.0% | Publication years span 1876 to 2024 |
| **Records with Cover Image ID (`cover_i`)** | **300** | 90.9% | Verified Open Library cover IDs |
| **Records Missing Cover Image (`cover_i=None`)** | **30** | 9.1% | Require local procedural fallback placeholder |
| **Broken Image URLs in Offline Mode** | **330** | 100.0% | Currently relies on external `covers.openlibrary.org` CDN |
| **Original Descriptions in Raw JSON** | **0** | 0.0% | Missing in raw json; synthesized in runtime code |
| **Duplicate Keys across Categories** | **3** | 0.9% | Interdisciplinary volumes (`OL1922611W`, `OL1826387W`, `OL4301885W`) |
| **Records with Physical Shelf Location** | **330** | 100.0% | Formatted as `KTHB-{DISCIPLINE}-{NUM}` |
| **Records with Availability Tags** | **330** | 100.0% | `Available in Stacks`, `On Reserve`, `Checked Out`, `Digital E-Book` |

---

## 3. Disciplinary Distribution

```
Discipline                     Count   Cover Available   Missing Cover
----------------------------------------------------------------------
Computer Science & AI             55                52               3
Agriculture & Biosystems          55                47               8
Mathematics & Statistics          55                52               3
Physics & Quantum                 55                51               4
Economics & Management            55                49               6
Robotics & Engineering            55                49               6
----------------------------------------------------------------------
TOTAL                            330               300              30
```

---

## 4. Duplicate Records & Interdisciplinary Overlap

Three Open Library work identifiers appear in two different categories due to interdisciplinary subject classification:
1. **`/works/OL1922611W`**: *Mathematical structures for computer science*
   - Appears in: `Computer Science & AI` and `Mathematics & Statistics`
2. **`/works/OL1826387W`**: *Introductory Mathematical Analysis*
   - Appears in: `Mathematics & Statistics` and `Economics & Management`
3. **`/works/OL4301885W`**: *Applied mathematics for business, economics, and the social sciences*
   - Appears in: `Mathematics & Statistics` and `Economics & Management`

In the new local database, each record receives a stable unique primary identifier (`book_001` through `book_330`) while preserving the shared work key and cross-category associations.

---

## 5. Metadata Gap Analysis & Migration Strategy

1. **Cover Image CDN Reliance:**
   - *Current issue:* Existing frontend attempts to load `https://covers.openlibrary.org/b/id/{cover_i}-M.jpg`. In an offline environment or behind firewalls, these fail to render.
   - *Migration solution:* Download all 300 available images into `data/book_covers/` using stable IDs, and generate a standardized local SVG/PNG placeholder for the 30 books lacking cover IDs.
2. **Description Gaps:**
   - *Current issue:* The raw JSON catalogue omitted explicit descriptions, generating a generic runtime stub: `"{title} — local catalogue dataset record; verify current holdings in KTH Primo."`
   - *Migration solution:* Generate meaningful, structured local descriptions incorporating title, author, discipline, publication era, and subject keywords to provide dense semantic signals for vector embeddings without fabricating biographical or physical facts.
3. **Offline Embedding Storage:**
   - Convert all 330 records into local high-dimensional numerical embeddings stored in `data/embeddings/book_vectors.npy` with an accompanying JSON index.
