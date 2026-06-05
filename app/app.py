import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
from sentence_transformers import SentenceTransformer
import faiss
import os
from groq import Groq
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
# works both locally and on HF Spaces
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_PATH)

st.set_page_config(
    page_title="Fashion Search Engine",
    page_icon="👗",
    layout="wide"
)

# ── Load models + indexes (cached — runs once) ────────────────────────────────
@st.cache_resource
def load_text_search():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('embeddings/text_index.faiss')
    products = pd.read_pickle('embeddings/product_metadata.pkl')
    return model, index, products

@st.cache_resource
def load_image_search():
    if not CLIP_AVAILABLE:
        return None, None, None, None, None
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)
    image_index = faiss.read_index('embeddings/image_index.faiss')
    image_products = pd.read_pickle('embeddings/image_product_metadata.pkl')
    return clip_model, clip_processor, image_index, image_products, device

@st.cache_resource
def load_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── RAG context ───────────────────────────────────────────────────────────────
RAG_CONTEXT = """
You are a personal fashion stylist with expertise in Indian fashion.

INDIAN FASHION KNOWLEDGE:
- Kurtas: Versatile Indian top. Cotton for office/casual, silk/embroidered for festivals
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

OCCASION GUIDE:
- Office: Kurta + palazzo/straight pants + flats/block heels
- Casual outing: Kurti + jeans + sneakers/flats
- Festival: Salwar suit or saree + traditional jewelry
- Wedding: Lehenga or heavy saree + heels + statement jewelry
"""

# ── Helper functions ──────────────────────────────────────────────────────────
def text_search(query, text_model, text_index, products, top_k=10):
    query_vector = text_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = text_index.search(query_vector, top_k)
    results = products.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    results['image_path'] = results['id'].apply(
        lambda x: f"data/images/images/{x}.jpg"
    )
    return results.reset_index(drop=True)


def image_text_search(query, clip_model, clip_processor,
                      image_index, image_products, device, top_k=10):
    inputs = clip_processor(
        text=[query], return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    query_vec = text_features.cpu().numpy().astype(np.float32)
    scores, indices = image_index.search(query_vec, top_k)

    results = image_products.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    results['image_path'] = results['id'].apply(
        lambda x: f"data/images/images/{x}.jpg"
    )
    return results.reset_index(drop=True)


def image_similarity_search(uploaded_image, clip_model, clip_processor,
                             image_index, image_products, device, top_k=9):
    image = uploaded_image.convert('RGB')
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    query_vec = features.cpu().numpy().astype(np.float32)
    scores, indices = image_index.search(query_vec, top_k + 1)

    results = image_products.iloc[indices[0][1:]].copy()
    results['similarity_score'] = scores[0][1:]
    results['image_path'] = results['id'].apply(
        lambda x: f"data/images/images/{x}.jpg"
    )
    return results.reset_index(drop=True)


def get_styling_recommendation(query, search_results, groq_client):
    results_text = "\n".join([
        f"  {i+1}. {row['productDisplayName']} "
        f"({row['baseColour']}, {row['gender']}, {row['articleType']}) "
        f"— similarity: {row['similarity_score']:.3f}"
        for i, (_, row) in enumerate(search_results.head(5).iterrows())
    ])

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
1. TOP PICK (2 sentences): Best product and why it matches
2. STYLING TIP (2 sentences): What to pair it with
3. OCCASION FIT (1 sentence): Best occasion for this look

Use actual product names. Keep under 120 words. Sound like a stylist.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


def display_product_grid(results, cols=5):
    """Show product images + names in a responsive grid"""
    rows = [results.iloc[i:i+cols] for i in range(0, len(results), cols)]
    for row_df in rows:
        col_list = st.columns(cols)
        for col, (_, product) in zip(col_list, row_df.iterrows()):
            with col:
                img_path = product.get('image_path', '')
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).resize((200, 200))
                        st.image(img, use_container_width=True)
                    except:
                       st.image("https://placehold.co/200x200?text=Fashion", use_container_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/200x200?text=No+Image",
                        use_container_width=True
                    )
                name = product.get('productDisplayName', 'Unknown')
                score = product.get('similarity_score', 0)
                color = product.get('baseColour', '')
                st.caption(f"**{name[:30]}**")
                st.caption(f"{color} · {score:.2f}")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" Fashion Search Engine")
st.markdown("*Semantic search + CLIP image search + AI styling recommendations*")
st.divider()

# Load everything
with st.spinner("Loading models... (first run takes ~30 seconds)"):
    text_model, text_index, products = load_text_search()
    groq_client = load_groq()

image_search_ready = False
try:
    clip_model, clip_processor, image_index, image_products, device = load_image_search()
    image_search_ready = True
except Exception:
    st.warning("Image index not ready yet — run build_image_index.py first. Text search works!")

st.success(f"Ready — {text_index.ntotal:,} products indexed")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    search_mode = st.radio(
        "Search Mode",
        ["Text Search", "Visual Search (CLIP)", "Upload Image"],
        index=0
    )
    top_k = st.slider("Results to show", 5, 20, 10)
    show_recommendation = st.toggle("✨ AI Styling Recommendation", value=True)

    st.divider()
    st.markdown("**How it works**")
    if search_mode == "Text Search":
        st.info("Sentence Transformers + FAISS\n\nUnderstands meaning, not just keywords")
    elif search_mode == "Visual Search (CLIP)":
        st.info("CLIP maps text and images to the same space — finds visually matching products")
    else:
        st.info("Upload any clothing image — CLIP finds visually similar products")

# ── Main search area ──────────────────────────────────────────────────────────
if search_mode in ["Text Search", "Visual Search (CLIP)"]:
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. blue kurta for office women, casual white tshirt men...",
        key="search_box"
    )

    if st.button("🔍 Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching..."):
                if search_mode == "Text Search":
                    results = text_search(query, text_model, text_index, products, top_k)
                else:
                    results = image_text_search(
                        query, clip_model, clip_processor,
                        image_index, image_products, device, top_k
                    )

            # Show recommendation first
            if show_recommendation:
                with st.spinner("Getting styling recommendation..."):
                    rec = get_styling_recommendation(query, results, groq_client)

                st.markdown("###  Stylist Recommendation")
                st.info(rec)
                st.divider()

            st.markdown(f"###  Top {len(results)} Results")
            st.caption(f"Mode: **{search_mode}** · Query: *{query}*")
            display_product_grid(results, cols=5)

        else:
            st.warning("Please enter a search query.")

elif search_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a clothing image to find similar products",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        with col1:
            uploaded_img = Image.open(uploaded_file)
            st.image(uploaded_img, caption="Your uploaded image", width=200)

        with col2:
            st.markdown("**Finding visually similar products...**")
            with st.spinner("Running CLIP image search..."):
                results = image_similarity_search(
                    uploaded_img, clip_model, clip_processor,
                    image_index, image_products, device, top_k
                )

            if show_recommendation:
                query_desc = f"products similar to the uploaded image"
                with st.spinner("Getting styling recommendation..."):
                    rec = get_styling_recommendation(
                        query_desc, results, groq_client
                    )
                st.markdown("###  Stylist Recommendation")
                st.info(rec)

        st.divider()
        st.markdown(f"### Top {len(results)} Similar Products")
        display_product_grid(results, cols=5)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Sentence Transformers · CLIP · FAISS · Groq Llama 3.3 70B · Streamlit"
)
