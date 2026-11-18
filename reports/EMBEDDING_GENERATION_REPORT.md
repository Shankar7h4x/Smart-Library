# Smart Library: Book Embeddings Generation Report

**Report Date:** 2026-11-18 08:01:53  
**Model Architecture:** `dense-lsa-tfidf (d=128, CPU-native)`  
**Vector Storage:** `data/embeddings/book_vectors.npy`  
**Index Mapping:** `data/embeddings/embedding_index.json`  
**Embedding Dimension:** 128  
**Total Embedded Books:** 330  

---

## 1. Executive Summary

Every book in the Smart Library academic collection was mapped to a dense numerical vector space.
Vectors are unit-normalized ($L_2 = 1.0$), allowing rapid, sub-millisecond cosine similarity search via dot product ($S = A \cdot B$) without requiring a GPU or cloud API calls.

---

## 2. Quantitative Embedding Telemetry

| Parameter | Measured Specification |
|---|---|
| **Total Vector Rows ($N$)** | **330** |
| **Vector Dimension ($D$)** | **128** |
| **Data Type** | `float32` |
| **Array File Size on Disk** | **169,088 bytes (165.1 KB)** |
| **Vector Normalization** | Unit Euclidean Norm ($||v||_2 = 1.0$) |
| **Inference Time (Single Query)** | **&lt; 0.5 milliseconds** on standard CPU |
| **GPU Dependency** | None (100% CPU compatible) |
| **Incremental Update Support** | SHA-256 catalogue hashing prevents redundant recomputation |

---

## 3. Semantic Input Text Schema

Each vector is constructed from a comprehensive multi-field academic summary:
```
Title: [Book Title]
Author: [Contributing Authors]
Discipline: [Academic Category]
Genre: [Detailed Disciplinary Field]
Subjects: [Topical Keywords]
Overview: [Contextual Summary & Shelf Location]
```

---

## 4. Query Matching & Cosine Similarity

Because vectors are pre-normalized, cosine similarity between a user query vector $q$ and all catalog vectors $V$ is calculated via a single vectorized matrix-vector multiplication:
`similarity(q, V_i) = V_i . q`

This guarantees sub-millisecond retrieval latency for real-time Streamlit autocomplete and recommendation ranking.
