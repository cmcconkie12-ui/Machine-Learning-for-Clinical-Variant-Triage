import streamlit as st
import pandas as pd
import numpy as np
import torch
from pytorch_model import VariantClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Hansoh Bio Variant Predictor", layout="wide")

st.title("🧬 Genetic Variant Clinical Predictor")
st.markdown("Developed for Hansoh Bio R&D - Predicting Conflicting Clinical Significance from NGS Data.")

# Load Data and Preprocessing artifacts (Simplified for Hackathon Demo)
@st.cache_data
def load_data():
    df = pd.read_csv("../data/clinvar_conflicting.csv", low_memory=False)
    numerical_cols = ['AF_ESP', 'AF_EXAC', 'AF_TGP', 'CADD_PHRED', 'CADD_RAW', 'BLOSUM62']
    categorical_cols = ['CHROM', 'REF', 'ALT', 'IMPACT', 'Consequence', 'CLNVC']
    
    df_features = df[numerical_cols + categorical_cols].copy()
    
    # Fill NA
    for col in numerical_cols:
        df_features[col] = df_features[col].fillna(df_features[col].median())
    for col in categorical_cols:
        df_features[col] = df_features[col].fillna('Missing')
        
    # Encoders
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_features[col] = df_features[col].astype(str)
        df_features[col] = le.fit_transform(df_features[col])
        encoders[col] = le
        
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)
    
    return df, df_features, scaled_features, scaler, encoders, numerical_cols, categorical_cols

try:
    df, df_features, scaled_features, scaler, encoders, num_cols, cat_cols = load_data()
    
    # Load PyTorch Model
    input_dim = scaled_features.shape[1]
    model = VariantClassifier(input_dim)
    
    try:
        model.load_state_dict(torch.load("variant_classifier.pth", weights_only=True))
        model.eval()
        st.success("PyTorch Model Loaded Successfully.")
    except FileNotFoundError:
        st.warning("Model weights not found. Please train the PyTorch model first.")
        
    st.sidebar.header("Select Variant from Dataset")
    sample_idx = st.sidebar.slider("Variant Index", 0, len(df)-1, 0)
    
    selected_row = df.iloc[sample_idx]
    
    st.subheader("Variant Details")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chromosome", str(selected_row['CHROM']))
    col2.metric("Position", str(selected_row['POS']))
    col3.metric("Reference Allele", str(selected_row['REF']))
    col4.metric("Alternate Allele", str(selected_row['ALT']))
    
    st.write(f"**Actual Class (0=Consistent, 1=Conflicting):** {selected_row['CLASS']}")
    
    if st.button("Predict Clinical Significance"):
        # Inference
        input_tensor = torch.tensor(scaled_features[sample_idx], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        pred_class = 1 if prob > 0.5 else 0
        
        st.subheader("Prediction")
        if pred_class == 1:
            st.error(f"**Conflicting Classification** (Probability: {prob:.2%})")
        else:
            st.success(f"**Consistent Classification** (Probability: {1-prob:.2%})")
            
        st.info("In a full deployment, this page would also display Captum Integrated Gradients (XAI) to explain which features contributed most to this prediction.")

except FileNotFoundError:
    st.error("Data file not found. Ensure `data/clinvar_conflicting.csv` exists.")
