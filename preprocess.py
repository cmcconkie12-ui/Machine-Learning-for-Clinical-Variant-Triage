import pandas as pd
import numpy as np

print("Loading ClinVar dataset...")
df = pd.read_csv("data/clinvar_conflicting.csv", low_memory=False)

y = df["CLASS"].astype(int)

print(f"Dataset was loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
print(f"Target distribution:\n{y.value_counts()}")


selected_numeric_cols = [
    "POS", "AF_ESP", "AF_EXAC", "AF_TGP",
    "CADD_PHRED", "CADD_RAW", "LoFtool", "BLOSUM62"
]
selected_categorical_cols = [
    "CHROM", "REF", "ALT", "IMPACT", "Consequence"
]

X_num = df[selected_numeric_cols].copy()
X_cat = df[selected_categorical_cols].copy()

print(f"\nExtracted {X_num.shape[1]} numeric and {X_cat.shape[1]} categorical columns.")
print("Numeric columns preview (missing values count):")
print(X_num.isnull().sum())


print("\nFilling in the missing numeric values with the column medians.....")
for col in selected_numeric_cols:
    X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
    median_val = X_num[col].median()
    X_num[col] = X_num[col].fillna(median_val if not np.isnan(median_val) else 0.0)

print("Number of missing values in numeric columns:")
print(X_num.isnull().sum())

print("")
print("Now we will clean the categorical features.......")

# 1. Fill any missing categorical text with a placeholder
X_cat = X_cat.fillna("Missing").astype(str)

# 2. Group long multi-base insertion/deletions into 'INDEL'
X_cat["REF"] = X_cat["REF"].apply(lambda x: x if len(x) == 1 else "INDEL")
X_cat["ALT"] = X_cat["ALT"].apply(lambda x: x if len(x) == 1 else "INDEL")

# 3. Perform One-Hot Encoding
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True, dtype=int)

# 4. Concatenate numerical and encoded categorical matrices horizontally
X = pd.concat([X_num, X_cat_encoded], axis=1)

print(f"Original categorical columns: {X_cat.shape[1]}")
print(f"One-Hot encoded columns produced: {X_cat_encoded.shape[1]}")
print(f"Total unified feature matrix (X) shape: {X.shape}")


# --- STEP 5: Stratified Split & Standard Scaling (Pure NumPy/Pandas) ---

print("\nSplitting into stratified Train (80%) and Test (20%) sets...")

# 1. Stratified split using pandas grouping
np.random.seed(42)

# Group by target y and sample 80% from each class
train_indices = X.groupby(y, group_keys=False).apply(
    lambda group: group.sample(frac=0.8, random_state=42)
).index

test_indices = X.index.difference(train_indices)

X_train, X_test = X.loc[train_indices].copy(), X.loc[test_indices].copy()
y_train, y_test = y.loc[train_indices].copy(), y.loc[test_indices].copy()

# 2. Standardize numerical features using training set statistics: (x - mean) / std
for col in selected_numeric_cols:
    mean_val = X_train[col].mean()
    std_val = X_train[col].std()
    
    # Avoid division by zero if std is 0
    std_val = std_val if std_val != 0 else 1.0
    
    X_train[col] = (X_train[col] - mean_val) / std_val
    X_test[col] = (X_test[col] - mean_val) / std_val

print("\n--- Final Preprocessing Results ---")
print(f"X_train shape: {X_train.shape} | y_train count: {len(y_train)}")
print(f"X_test shape:  {X_test.shape}  | y_test count:  {len(y_test)}")
print("\nTarget balance in Train set:")
print(y_train.value_counts(normalize=True).round(3))
print("\nTarget balance in Test set:")
print(y_test.value_counts(normalize=True).round(3))