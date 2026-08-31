import pandas as pd

file_path = "data/clinvar_conflicting.csv"

print("Loading ClinVar dataset...")
df = pd.read_csv(file_path, low_memory=False)

print("\n--- Dataset Summary ---")
print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- Target Class Distribution ---")
print(df["CLASS"].value_counts(dropna=False))

print("\n--- Sample Variant Records ---")
preview_cols = ["CHROM", "POS", "REF", "ALT", "CLASS"]
print(df[preview_cols].head())