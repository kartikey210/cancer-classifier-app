import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Optional UMAP
try:
    import importlib
    if importlib.util.find_spec("umap") is not None:
        umap = importlib.import_module("umap")
        UMAP_AVAILABLE = True
    else:
        UMAP_AVAILABLE = False
except:
    UMAP_AVAILABLE = False

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    top_genes = pickle.load(open(os.path.join(BASE_DIR, "top_genes.pkl"), "rb"))
except:
    st.error("❌ Model files not found!")
    st.stop()

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cancer Classifier", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🧬 Cancer Type Classification System")
st.subheader("Breast Cancer (BRCA) vs Lung Cancer (LUAD)")

st.markdown("""
This system uses **Machine Learning on gene expression data**  
to classify cancer types.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Controls")

option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Sample Input", "Manual Input", "Upload CSV"]
)

# Model info
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Features: Top Genes")
st.sidebar.write("Type: Binary Classification")

# -----------------------------
# SAMPLE INPUT
# -----------------------------
if option == "Sample Input":
    st.header("🧪 Sample Prediction")

    if st.button("Run Sample Prediction"):
        sample = np.random.rand(len(top_genes)).reshape(1, -1)
        sample_scaled = scaler.transform(sample)

        pred = model.predict(sample_scaled)
        proba = model.predict_proba(sample_scaled)

        st.subheader("Result")

        if pred[0] == 0:
            st.success("🟢 BRCA (Breast Cancer)")
        else:
            st.success("🔴 LUAD (Lung Cancer)")

        st.info(f"Confidence: {np.max(proba)*100:.2f}%")

# -----------------------------
# MANUAL INPUT
# -----------------------------
elif option == "Manual Input":
    st.header("⌨️ Manual Input")

    inputs = []
    for i in range(min(10, len(top_genes))):
        val = st.number_input(f"Gene {i+1}", value=0.0)
        inputs.append(val)

    if st.button("Predict"):
        full = inputs + [0]*(len(top_genes)-len(inputs))
        full = np.array(full).reshape(1, -1)

        full_scaled = scaler.transform(full)
        pred = model.predict(full_scaled)
        proba = model.predict_proba(full_scaled)

        if pred[0] == 0:
            st.success("🟢 BRCA")
        else:
            st.success("🔴 LUAD")

        st.info(f"Confidence: {np.max(proba)*100:.2f}%")

# -----------------------------
# CSV INPUT
# -----------------------------
elif option == "Upload CSV":
    st.header("📂 Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        if st.button("Predict CSV"):
            try:
                X = df.values[:, :len(top_genes)]
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)

                df["Prediction"] = preds
                df["Prediction"] = df["Prediction"].map({0: "BRCA", 1: "LUAD"})

                st.success("Prediction Done")
                st.dataframe(df)
            except:
                st.error("CSV format incorrect")

# -----------------------------
# PCA VISUALIZATION
# -----------------------------
st.markdown("---")
st.header("📉 PCA Visualization")

if st.button("Show PCA"):
    data = np.random.rand(100, len(top_genes))
    data_scaled = scaler.transform(data)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)

    fig, ax = plt.subplots()
    ax.scatter(reduced[:,0], reduced[:,1])
    ax.set_title("PCA Projection")

    st.pyplot(fig)

# -----------------------------
# UMAP VISUALIZATION
# -----------------------------
st.header("📌 UMAP Visualization")

if UMAP_AVAILABLE:
    if st.button("Show UMAP"):
        reducer = umap.UMAP()
        data = np.random.rand(100, len(top_genes))
        data_scaled = scaler.transform(data)

        embedding = reducer.fit_transform(data_scaled)

        fig, ax = plt.subplots()
        ax.scatter(embedding[:,0], embedding[:,1])
        ax.set_title("UMAP Projection")

        st.pyplot(fig)
else:
    st.warning("Install UMAP: pip install umap-learn")

# -----------------------------
# CONFUSION MATRIX (Demo)
# -----------------------------
st.markdown("---")
st.header("📊 Model Evaluation")

if st.button("Show Confusion Matrix"):
    y_true = np.random.randint(0,2,100)
    y_pred = np.random.randint(0,2,100)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    accuracy = np.mean(y_true == y_pred)
    st.info(f"Accuracy (demo): {accuracy*100:.2f}%")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.markdown("---")
st.header("🧬 Top Important Genes")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_

    top_idx = np.argsort(importances)[-10:]

    fig, ax = plt.subplots()
    ax.barh(range(len(top_idx)), importances[top_idx])
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_idx)
    ax.set_title("Top Gene Importance")

    st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🎓 Final Year B.Tech Bioinformatics Project")