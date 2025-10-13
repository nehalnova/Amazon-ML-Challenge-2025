import pandas as pd
import numpy as np
import re
import os

# Paths (update if needed)
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
OUTPUT_PATH = "dataset/processed_subset/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ========== 1️⃣ Load Data Efficiently ==========
print("Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Original Train shape: {train_df.shape}")
print(f"Original Test shape: {test_df.shape}")

# ========== 2️⃣ Create 25K Subset ==========
# Ensure train has at least 25k rows
train_subset = train_df.sample(n=25000, random_state=42)

# Some test sets may be smaller, so check before sampling
test_subset = test_df.sample(n=min(25000, len(test_df)), random_state=42)

print(f"Subset Train shape: {train_subset.shape}")
print(f"Subset Test shape: {test_subset.shape}")


# ========== 3️⃣ Handle Missing Values ==========
train_subset["catalog_content"] = train_subset["catalog_content"].fillna("")
test_subset["catalog_content"] = test_subset["catalog_content"].fillna("")
train_subset = train_subset.dropna(subset=["price"])

# ========== 4️⃣ Basic Text Cleaning ==========
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)       # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_subset["clean_text"] = train_subset["catalog_content"].apply(clean_text)
test_subset["clean_text"] = test_subset["catalog_content"].apply(clean_text)

# ========== 5️⃣ Handle Outliers in Price ==========
q1 = train_subset["price"].quantile(0.01)
q99 = train_subset["price"].quantile(0.99)
train_subset["price"] = np.clip(train_subset["price"], q1, q99)

# ========== 6️⃣ Save Cleaned Subsets ==========
train_subset.to_csv(os.path.join(OUTPUT_PATH, "train_clean_subset.csv"), index=False)
test_subset.to_csv(os.path.join(OUTPUT_PATH, "test_clean_subset.csv"), index=False)

print("\n✅ Subset preprocessing completed successfully!")
print(f"Cleaned train subset saved at: {OUTPUT_PATH}train_clean_subset.csv")
print(f"Cleaned test subset saved at: {OUTPUT_PATH}test_clean_subset.csv")
