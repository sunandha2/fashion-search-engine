# Fashion Search Engine — Multimodal AI

> Search by text OR upload an image.
> Find visually and semantically similar fashion products.
> Powered by CLIP + Sentence Transformers + FAISS + RAG.

## The Problem
Keyword search fails for fashion. "Comfortable summer dress 
for office" returns nothing useful if no product description 
uses those exact words. Visual search fails alone too — 
you need both meaning AND appearance.

## What It Does
- Text search: finds products matching meaning, not just keywords
- Image search: upload any outfit photo, find visually similar styles
- Multimodal: combine text + image for precision search
- RAG layer: Groq LLM writes personalized styling recommendations
- Live Streamlit app deployed publicly

## Tech Stack
| Tool | Purpose |
|---|---|
| Sentence Transformers | Text embeddings |
| CLIP (OpenAI) | Image + text embeddings |
| FAISS | Vector similarity search |
| Groq API (Llama 3.3) | RAG styling recommendations |
| Streamlit | Live deployed app |

## Dataset
44,000 fashion products — Kaggle Fashion Product Images
Categories: Apparel, Footwear, Accessories, Beauty

## Progress
- [x] Day 1 — 44,412 fashion products explored (Apparel, Footwear, Accessories)
- [x] Day 2 — Sentence Transformer embeddings + FAISS index built (44,412 products, sub-millisecond search)
- [ ] Day 3 — CLIP image embeddings + multimodal search
- [ ] Day 4 — RAG styling recommendations
- [ ] Day 5 — Streamlit app + deployment