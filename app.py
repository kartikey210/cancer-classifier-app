import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
import umap

# -----------------------------
# Load model safely
# -----------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    top_genes = pickle.load(open("top_genes.pkl", "rb"))
except:
    st.error("❌ Model files not found! Make sure all .pkl files are uploaded.")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("🧬 Cancer Type Classification System")
st.subheader("Breast Cancer (BRCA) vs Lung Cancer (LUAD)")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
option = st.sidebar.selectbox("Choose Input Method", ["Sample Input", "Upload CSV"])

# -----------------------------
# Function: Predict
# -----------------------------
def predict(data):
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    prob = model.predict_proba(data_scaled)
    return pred, prob

# -----------------------------
# SAMPLE INPUT
# -----------------------------
if option == "Sample Input":
    if st.button("Run Sample Prediction"):
        sample = np.random.rand(1, len(top_genes))
        pred, prob = predict(sample)

        result = "BRCA (Breast Cancer)" if pred[0] == 0 else "LUAD (Lung Cancer)"
        confidence = np.max(prob) * 100

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

# -----------------------------
# CSV UPLOAD
# -----------------------------
elif option == "Upload CSV":
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("Preview of Data:")
        st.dataframe(df.head())

        try:
            df = df[top_genes]  # match features
        except:
            st.error("❌ CSV must contain correct gene columns")
            st.stop()

        if st.button("Predict CSV"):
            pred, prob = predict(df)

            df["Prediction"] = ["BRCA" if p == 0 else "LUAD" for p in pred]
            df["Confidence"] = np.max(prob, axis=1)

            st.write("Results:")
            st.dataframe(df)

# -----------------------------
# PCA
# -----------------------------
if st.button("Show PCA"):
    sample = np.random.rand(100, len(top_genes))
    sample_scaled = scaler.transform(sample)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(sample_scaled)

    fig, ax = plt.subplots()
    ax.scatter(reduced[:, 0], reduced[:, 1])
    ax.set_title("PCA Projection")
    st.pyplot(fig)

# -----------------------------
# UMAP
# -----------------------------
if st.button("Show UMAP"):
    sample = np.random.rand(100, len(top_genes))
    sample_scaled = scaler.transform(sample)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(sample_scaled)

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    ax.set_title("UMAP Projection")
    st.pyplot(fig)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
if st.button("Show Confusion Matrix"):
    sample = np.random.rand(100, len(top_genes))
    y_true = np.random.randint(0, 2, 100)

    sample_scaled = scaler.transform(sample)
    y_pred = model.predict(sample_scaled)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

# -----------------------------
# ROC CURVE (NEW 🔥)
# -----------------------------
if st.button("Show ROC Curve"):
    sample = np.random.rand(100, len(top_genes))
    y_true = np.random.randint(0, 2, 100)

    sample_scaled = scaler.transform(sample)
    prob = model.predict_proba(sample_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
if st.button("Show Classification Report"):
    sample = np.random.rand(100, len(top_genes))
    y_true = np.random.randint(0, 2, 100)

    sample_scaled = scaler.transform(sample)
    y_pred = model.predict(sample_scaled)

    report = classification_report(y_true, y_pred)
    st.text(report)