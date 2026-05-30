import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import time

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\fashion-search-engine')
os.makedirs('embeddings', exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv('data/products_clean.csv')
print(f"Products loaded: {len(df)}")

# Build rich text description for each product
# More context = better search results
df['search_text'] = (
    df['productDisplayName'].fillna('') + ' ' +
    df['articleType'].fillna('') + ' ' +
    df['baseColour'].fillna('') + ' ' +
    df['gender'].fillna('') + ' ' +
    df['masterCategory'].fillna('') + ' ' +
    df['subCategory'].fillna('') + ' ' +
    df['usage'].fillna('') + ' ' +
    df['season'].fillna('')
).str.strip()

# Remove any remaining NaN or empty rows
df = df[df['search_text'].notna() & (df['search_text'] != '')]
print(f"Products after cleaning: {len(df)}")

print(f"\nSample search text:")
for i in range(3):
    print(f"  {i+1}. {df['search_text'].iloc[i]}")

print("=" * 60)
print("STEP 2 — Loading Sentence Transformer model")
print("=" * 60)

# all-MiniLM-L6-v2 is fast, lightweight, excellent for fashion
# Downloads ~90MB on first run — cached after that
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Model loaded: all-MiniLM-L6-v2")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

print("=" * 60)
print("STEP 3 — Generating embeddings for all 44,412 products")
print("=" * 60)
print("This takes 3-5 minutes. Do not close the terminal.")

start = time.time()

# Batch encode for speed
embeddings = model.encode(
    df['search_text'].tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # L2 normalize for cosine similarity
)

elapsed = time.time() - start
print(f"\nEmbeddings generated in {elapsed:.1f} seconds")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Each product is a {embeddings.shape[1]}-dimensional vector")

print("=" * 60)
print("STEP 4 — Building FAISS index")
print("=" * 60)

# IndexFlatIP = Inner Product (cosine similarity with normalized vectors)
# Most accurate — searches all 44K vectors
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Add all embeddings to index
index.add(embeddings.astype(np.float32))
print(f"FAISS index built")
print(f"Total vectors in index: {index.ntotal}")

print("=" * 60)
print("STEP 5 — Testing the search")
print("=" * 60)

def search_products(query, top_k=10):
    """Search for similar products given a text query"""
    # Encode query to vector
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Search FAISS index
    scores, indices = index.search(query_vector, top_k)

    # Get product details
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    return results[['productDisplayName', 'articleType',
                     'baseColour', 'gender', 'masterCategory',
                     'similarity_score', 'id']]

# Test with 5 different queries
test_queries = [
    "comfortable blue kurta for office women",
    "casual white tshirt men summer",
    "black leather shoes formal men",
    "pink floral dress women party",
    "running shoes men sports lightweight"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = search_products(query, top_k=5)
    for _, row in results.iterrows():
        print(f"  [{row['similarity_score']:.3f}] {row['productDisplayName']} ({row['baseColour']}, {row['gender']})")

print("=" * 60)
print("STEP 6 — Saving index and metadata")
print("=" * 60)

# Save FAISS index
faiss.write_index(index, 'embeddings/text_index.faiss')
print("Saved: embeddings/text_index.faiss")

# Save product metadata aligned with index
product_metadata = df[[
    'id', 'productDisplayName', 'articleType',
    'baseColour', 'gender', 'masterCategory',
    'subCategory', 'usage', 'season', 'image_path'
]].reset_index(drop=True)

product_metadata.to_pickle('embeddings/product_metadata.pkl')
print("Saved: embeddings/product_metadata.pkl")

# Save embeddings for Day 3 (image search)
np.save('embeddings/text_embeddings.npy', embeddings)
print("Saved: embeddings/text_embeddings.npy")

print(f"\nIndex size: {index.ntotal} products")
print(f"Search ready — sub-millisecond queries on 44K products")
