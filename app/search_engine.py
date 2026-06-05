import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from groq import Groq
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\fashion-search-engine')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("=" * 60)
print("STEP 1 — Loading search engine components")
print("=" * 60)

# Load sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence transformer loaded")

# Load FAISS index
index = faiss.read_index('embeddings/text_index.faiss')
print(f"FAISS index loaded: {index.ntotal} products")

# Load product metadata
products = pd.read_pickle('embeddings/product_metadata.pkl')
print(f"Product metadata loaded: {len(products)} products")

print("\n" + "=" * 60)
print("STEP 2 — Building search function")
print("=" * 60)

def search_products(query, top_k=8):
    """Semantic search — finds products by meaning not keywords"""
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(query_vector, top_k)

    results = products.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    results['image_path'] = results['id'].apply(
        lambda x: f"data/images/images/{x}.jpg"
    )
    return results.reset_index(drop=True)

print("Search function ready")

print("\n" + "=" * 60)
print("STEP 3 — Building RAG styling recommendation")
print("=" * 60)

# RAG Context — fashion knowledge base
RAG_CONTEXT = """
You are a personal fashion stylist with expertise in Indian fashion.

INDIAN FASHION KNOWLEDGE:
- Kurtas: Versatile Indian top. Cotton kurtas for office/casual,
  silk/embroidered for festivals/weddings
- Sarees: Traditional wear. Cotton for daily, silk for occasions
- Salwar suits: 3-piece set. Professional and comfortable
- Lehenga: Festive/wedding wear, highly embellished
- Indo-western: Mix of Indian and western — trendy for young professionals

STYLING RULES:
- Office wear: Solid colors, subtle prints, cotton/linen fabrics
- Casual: Bright colors, bold prints, comfortable fabrics
- Party/evening: Rich fabrics (silk, chiffon), embellishments
- Festive/wedding: Traditional silhouettes, rich colors, embroidery

COLOR COMBINATIONS THAT WORK:
- Blue + white: Classic, professional
- Red + gold: Festive, traditional
- Pastels (mint, lavender, peach): Soft, feminine
- Black + any color: Always works
- Navy + beige: Sophisticated office look

POPULAR INDIAN ACCESSORIES:
- Kolhapuri flats: Casual/office with kurtas
- Block heels: Semi-formal, pairs with salwar suits
- Jhumkas (earrings): Traditional, goes with kurtas/sarees
- Potli bag: Festive occasions
- Tote bag: Office/daily use

OCCASION GUIDE:
- Office: Kurta + palazzo/straight pants + flats/block heels
- Casual outing: Kurti + jeans + sneakers/flats
- Festival: Salwar suit or saree + traditional jewelry
- Wedding: Lehenga or heavy saree + heels + statement jewelry
"""

def generate_styling_recommendation(query, search_results):
    """Use Groq RAG to generate personalized styling recommendation"""

    # Format top 5 results
    results_text = "\n".join([
        f"  {i+1}. {row['productDisplayName']} "
        f"({row['baseColour']}, {row['gender']}, "
        f"{row['articleType']}) — "
        f"similarity: {row['similarity_score']:.3f}"
        for i, (_, row) in enumerate(
            search_results.head(5).iterrows()
        )
    ])

    # Get unique categories in results
    categories = search_results['articleType'].value_counts().head(3)
    colors = search_results['baseColour'].value_counts().head(3)

    prompt = f"""
{RAG_CONTEXT}

CUSTOMER SEARCH QUERY: "{query}"

TOP SEARCH RESULTS:
{results_text}

RESULT SUMMARY:
- Top article types: {', '.join(categories.index.tolist())}
- Top colors found: {', '.join(colors.index.tolist())}

Write a personalized styling recommendation with 3 sections:
1. TOP PICK (2 sentences): Recommend the best product from results and why it matches the query
2. STYLING TIP (2 sentences): How to style this item — what to pair it with (bottoms, shoes, accessories)
3. OCCASION FIT (1 sentence): What occasion or setting this works best for

Be specific about Indian fashion context.
Use the actual product names from results.
Keep total under 120 words.
Sound like a knowledgeable personal stylist.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4
    )

    return response.choices[0].message.content.strip()

print("RAG styling function ready")

print("\n" + "=" * 60)
print("STEP 4 — Testing search + RAG recommendations")
print("=" * 60)

test_queries = [
    "blue kurta for office women",
    "casual white tshirt men summer",
    "black formal shoes men",
    "pink party dress women evening",
    "comfortable running shoes men"
]

recommendations = []

for query in test_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    # Search
    results = search_products(query, top_k=8)

    print(f"Top 5 results:")
    for i, (_, row) in enumerate(results.head(5).iterrows()):
        print(f"  [{row['similarity_score']:.3f}] "
              f"{row['productDisplayName']} "
              f"({row['baseColour']}, {row['gender']})")

    # Generate RAG recommendation
    recommendation = generate_styling_recommendation(query, results)
    print(f"\nStylist recommendation:")
    print(recommendation)

    recommendations.append({
        'query': query,
        'top_result': results.iloc[0]['productDisplayName'],
        'top_result_color': results.iloc[0]['baseColour'],
        'recommendation': recommendation
    })

    time.sleep(0.5)

print("\n" + "=" * 60)
print("STEP 5 — Saving recommendations")
print("=" * 60)

rec_df = pd.DataFrame(recommendations)
rec_df.to_csv('outputs/styling_recommendations.csv', index=False)
print(f"Saved: outputs/styling_recommendations.csv")
print(f"Queries tested: {len(rec_df)}")

print("\n" + "=" * 60)
print("STEP 6 — Summary")
print("=" * 60)

print(f"\nSearch engine stats:")
print(f"Products indexed: {index.ntotal:,}")
print(f"Embedding model: all-MiniLM-L6-v2 (384 dims)")
print(f"Search type: Semantic (meaning-based, not keyword)")
print(f"RAG context: Indian fashion knowledge base")
print(f"LLM: Groq Llama 3.3 70B")
print(f"\nQueries tested: {len(rec_df)}")
print("Outputs:")
print("  outputs/styling_recommendations.csv")