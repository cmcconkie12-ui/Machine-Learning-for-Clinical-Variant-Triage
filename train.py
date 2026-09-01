import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. Load Data ---
print("Loading preprocessed training and test data...")
train_df = pd.read_csv("data/train_processed.csv")
test_df = pd.read_csv("data/test_processed.csv")

X_train = train_df.drop(columns=["CLASS"])
y_train = train_df["CLASS"]

X_test = test_df.drop(columns=["CLASS"])
y_test = test_df["CLASS"]

print(f"Train features: {X_train.shape} | Test features: {X_test.shape}")

# --- 2. Evaluation Helper Function ---
def evaluate_model(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_t)[:, 1]
        roc_auc = roc_auc_score(y_t, y_prob)
    else:
        roc_auc = 0.5

    print(f"\n==================== {name} ====================")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_t, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_t, y_pred, digits=4))

# --- 3. Model 1: Baseline Dummy ---
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
evaluate_model("Baseline (Dummy Classifier - Majority Class)", dummy, X_test, y_test)

# --- 4. Model 2: Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)
evaluate_model("Logistic Regression (Balanced)", log_reg, X_test, y_test)

# --- 5. Model 3: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate_model("Random Forest (100 Trees, Balanced)", rf, X_test, y_test)

# --- 6. Model 4: XGBoost ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
evaluate_model("XGBoost (Gradient Boosted Trees)", xgb, X_test, y_test)