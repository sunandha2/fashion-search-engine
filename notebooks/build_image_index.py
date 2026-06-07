import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\fashion-search-engine')
BASE_DIR = r'C:\Users\sunandha\Downloads\gitdemo\fashion-search-engine'
os.makedirs('embeddings', exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading clean product data")
print("=" * 60)

df = pd.read_csv('data/products_clean.csv')
print(f"Products: {len(df)}")

visual_categories = ['Apparel', 'Footwear', 'Accessories']
df_visual = df[df['masterCategory'].isin(visual_categories)].copy()
df_visual = df_visual.reset_index(drop=True)
print(f"Visual products (Apparel+Footwear+Accessories): {len(df_visual)}")

df_visual['image_path'] = df_visual['id'].apply(
    lambda x: f"data/images/images/{x}.jpg"
)
df_visual['image_exists'] = df_visual['image_path'].apply(
    lambda p: os.path.exists(os.path.join(BASE_DIR, p))
)
df_visual = df_visual[df_visual['image_exists']].reset_index(drop=True)
print(f"Products with confirmed images: {len(df_visual)}")

print("\n" + "=" * 60)
print("STEP 2 — Loading CLIP model")
print("=" * 60)

print("Loading CLIP (openai/clip-vit-base-patch32)...")
print("First run downloads ~600MB — cached after that...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"CLIP loaded on: {device}")
print(f"Image embedding dimension: 512")

print("\n" + "=" * 60)
print("STEP 3 — Generating image embeddings")
print("=" * 60)
print("Processing images in batches of 32...")
print("This takes 15-30 minutes for 40K images. Do not close terminal.")

def get_image_embedding(image_path):
    try:
        full_path = os.path.join(BASE_DIR, image_path)
        image = Image.open(full_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            # use vision_model directly instead of get_image_features
            outputs = model.vision_model(**inputs)
            features = outputs.pooler_output
            features = model.visual_projection(features)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0]
    except Exception as e:
        print(f"ERROR: {full_path} → {e}")
        return None

BATCH_SIZE = 32
all_embeddings = []
valid_indices = []

start = time.time()

for i in tqdm(range(0, len(df_visual), BATCH_SIZE),
              desc="Generating image embeddings"):
    batch = df_visual.iloc[i:i+BATCH_SIZE]

    for idx, row in batch.iterrows():
        emb = get_image_embedding(row['image_path'])
        if emb is not None:
            all_embeddings.append(emb)
            valid_indices.append(idx)

elapsed = time.time() - start
print(f"\nEmbeddings generated in {elapsed/60:.1f} minutes")
print(f"Successfully processed: {len(all_embeddings)} images")
print(f"Failed / skipped: {len(df_visual) - len(all_embeddings)} images")

print("\n" + "=" * 60)
print("STEP 4 — Building CLIP FAISS index")
print("=" * 60)

embeddings_array = np.array(all_embeddings).astype(np.float32)
print(f"Embeddings array shape: {embeddings_array.shape}")

dimension = embeddings_array.shape[1]
image_index = faiss.IndexFlatIP(dimension)
image_index.add(embeddings_array)

print(f"CLIP FAISS index built")
print(f"Total images indexed: {image_index.ntotal}")

print("\n" + "=" * 60)
print("STEP 5 — Testing image search")
print("=" * 60)

valid_products = df_visual.iloc[valid_indices].reset_index(drop=True)

def search_by_text_clip(text_query, top_k=5):
    """Search using CLIP text encoder — same vector space as images"""
    inputs = processor(
        text=[text_query],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(
            p=2, dim=-1, keepdim=True
        )

    query_vec = text_features.cpu().numpy().astype(np.float32)
    scores, indices = image_index.search(query_vec, top_k)

    results = valid_products.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    return results[['productDisplayName', 'articleType',
                     'baseColour', 'gender', 'similarity_score']]

def search_by_image(image_path, top_k=5):
    """Find similar products given an image"""
    emb = get_image_embedding(image_path)
    if emb is None:
        return None

    query_vec = emb.reshape(1, -1).astype(np.float32)
    scores, indices = image_index.search(query_vec, top_k + 1)

    results = valid_products.iloc[indices[0][1:]].copy()
    results['similarity_score'] = scores[0][1:]
    return results[['productDisplayName', 'articleType',
                     'baseColour', 'gender', 'similarity_score']]

print("\nTesting CLIP text -> image search:")
test_queries = [
    "blue kurta women ethnic",
    "white sneakers men sports",
    "black formal shoes men",
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    results = search_by_text_clip(query, top_k=3)
    for _, row in results.iterrows():
        print(f"  [{row['similarity_score']:.3f}] "
              f"{row['productDisplayName']} "
              f"({row['baseColour']}, {row['gender']})")

print("\nTesting image -> image search:")
test_image = valid_products.iloc[0]['image_path']
test_product = valid_products.iloc[0]['productDisplayName']
print(f"Query image: {test_product}")

results = search_by_image(test_image, top_k=3)
if results is not None:
    for _, row in results.iterrows():
        print(f"  [{row['similarity_score']:.3f}] "
              f"{row['productDisplayName']} "
              f"({row['baseColour']}, {row['gender']})")

print("\n" + "=" * 60)
print("STEP 6 — Saving everything")
print("=" * 60)

faiss.write_index(image_index, 'embeddings/image_index.faiss')
print("Saved: embeddings/image_index.faiss")

valid_products.to_pickle('embeddings/image_product_metadata.pkl')
print("Saved: embeddings/image_product_metadata.pkl")

np.save('embeddings/image_embeddings.npy', embeddings_array)
print("Saved: embeddings/image_embeddings.npy")

print(f"\nImage index ready")
print(f"Total images indexed: {image_index.ntotal}")
print(f"Embedding dimension: {dimension} (CLIP ViT-B/32)")
print(f"Supports: text -> image search AND image -> image search")
