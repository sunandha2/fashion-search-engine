# Fashion Search Engine — Multimodal AI Search & Recommendation System

Search 44,412 fashion products using natural language or images. Built using Sentence Transformers, OpenAI CLIP, FAISS vector search, RAG, and Groq Llama 3.3 70B.

## Live Demo
[Streamlit App](YOUR_STREAMLIT_LINK)

## Tech Stack
Python • FAISS • Sentence Transformers • OpenAI CLIP • Groq Llama 3.3 70B • RAG • Streamlit
# Demo
![### Semantic Search]](<Screenshot 2026-06-05 163252.png>)
![### CLIP Visual Search]](<Screenshot 2026-06-06 135311.png>)
![### Upload Image Similarity Search](<Screenshot 2026-06-06 135428.png>)
## Problem
Traditional keyword search fails when users describe products naturally.

Example:

comfortable kurta for office

may not appear in product descriptions even when relevant products exist.

This project combines semantic understanding and visual similarity search to improve fashion product discovery.

## Features
1. Semantic search across 44,412 products

2. Text-to-image retrieval using CLIP

3. Image-to-image similarity search

4. FAISS vector database for fast retrieval

5. RAG-powered styling recommendations

6. Streamlit deployment

## System Architecture
User Query
    │
    ▼
Sentence Transformer / CLIP
    │
    ▼
FAISS Vector Search
    │
    ▼
Top-K Products
    │
    ▼
Groq Llama 3.3 70B
    │
    ▼
Styling Recommendations

## Dataset
Kaggle Fashion Product Images Dataset
44,412 products
41,883 product images

Categories:

Apparel
Footwear
Accessories
Beauty

## Technical Implementation

Semantic Search

all-MiniLM-L6-v2
384-dimensional embeddings
FAISS IndexFlatIP
Cosine similarity retrieval

Visual Search

CLIP ViT-B/32
512-dimensional embeddings
Text-to-image retrieval
Image-to-image retrieval

Recommendation Engine

Retrieval-Augmented Generation
Groq Llama 3.3 70B
Context-aware fashion recommendations

## Results
| Metric           | Value            |
| ---------------- | ---------------- |
| Products Indexed | 44,412           |
| Product Images   | 41,883           |
| Embedding Model  | all-MiniLM-L6-v2 |
| Visual Model     | CLIP ViT-B/32    |
| Search Modes     | 3                |
| Deployment       | Streamlit        |

## Project Structure
fashion-search-engine/
│
├── app/
│   ├── app.py
│   
│
├── notebooks/
│   ├── build_text_index.py
|   │   
|   └── rag_styling.py
│
├── embeddings/
│
├── data/
│
├── requirements.txt
│
└── README.md

## Run Locally
git clone https://github.com/sunandha2/fashion-search-engine.git

cd fashion-search-engine

pip install -r requirements.txt

streamlit run app/app.py

## Key Learnings
Multimodal retrieval systems
Vector databases using FAISS
CLIP embeddings
RAG pipelines
LLM integration with Groq
Production deployment using Streamlit
