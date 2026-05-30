import pandas as pd
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\fashion-search-engine')

df = pd.read_csv('data/styles.csv', on_bad_lines='skip')
print(f"Total products: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample data:")
print(df.head(5))

print(f"\nCategories:")
print(df['masterCategory'].value_counts().head(10))

print(f"\nSubcategories:")
print(df['subCategory'].value_counts().head(10))

print(f"\nProduct types:")
print(df['articleType'].value_counts().head(15))

print(f"\nGenders:")
print(df['gender'].value_counts())

print(f"\nColors:")
print(df['baseColour'].value_counts().head(10))

# Check how many images exist
image_dir = 'data/images/images'
image_files = os.listdir(image_dir)
print(f"\nTotal images available: {len(image_files)}")

# Match images to products
df['image_path'] = df['id'].apply(
    lambda x: f"{image_dir}/{x}.jpg"
)
df['image_exists'] = df['image_path'].apply(os.path.exists)
print(f"Products with images: {df['image_exists'].sum()}")

# Save clean version
clean_df = df[df['image_exists']].copy()
clean_df = clean_df.dropna(subset=['productDisplayName'])
clean_df.to_csv('data/products_clean.csv', index=False)
print(f"\nClean dataset saved: {len(clean_df)} products")
print(f"\nSample product names:")
print(clean_df['productDisplayName'].head(10).tolist())