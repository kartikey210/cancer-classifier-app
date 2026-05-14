import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.decomposition import PCA
import umap

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cancer Type Classification",
    layout="wide"
)

# =========================================================
# LOAD MODEL FILES
# =========================================================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    top_genes = joblib.load("top_genes.pkl")

    st.success("✅ Model files loaded successfully")

    if hasattr(model, "classes_"):
        st.write("Model Classes:", model.classes_)

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# =========================================================
# TITLE
# =========================================================
st.title("🧬 Cancer Type Classification System")
st.subheader("Breast Cancer (BRCA) vs Lung Cancer (LUAD)")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Controls")

option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Sample Input", "Upload CSV"]
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict(data):

    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)

    prob = model.predict_proba(data_scaled)

    return pred, prob

# =========================================================
# SAMPLE INPUT
# =========================================================
if option == "Sample Input":

    st.write("## Random Sample Prediction")

    if st.button("Run Sample Prediction"):

        sample = np.random.rand(1, len(top_genes))

        pred, prob = predict(sample)

        result = "LUAD (Lung Cancer)" if pred[0] == 0 else "BRCA (Breast Cancer)"

        confidence = np.max(prob) * 100

        st.success(f"Prediction: {result}")

        st.info(f"Confidence: {confidence:.2f}%")

# =========================================================
# CSV UPLOAD
# =========================================================
elif option == "Upload CSV":

    file = st.file_uploader(
        "Upload CSV File",
        type=["csv", "tsv", "txt"]
    )

    if file is not None:

        # =================================================
        # READ FILE
        # =================================================
        try:

            if file.name.endswith(".tsv") or file.name.endswith(".txt"):
                df = pd.read_csv(file, sep="\t")

            else:
                df = pd.read_csv(file)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

        st.write("## Preview of Uploaded Data")

        st.dataframe(df.head())

        # =================================================
        # HANDLE TCGA RNASEQ FORMAT
        # =================================================
        if "gene_name" in df.columns:

            st.info("Detected TCGA RNA-seq format")

            # expression column detection
            expression_col = None

            possible_cols = [
                "unstranded",
                "tpm_unstranded",
                "fpkm_unstranded",
                "fpkm_uq_unstranded"
            ]

            for col in possible_cols:
                if col in df.columns:
                    expression_col = col
                    break

            if expression_col is None:
                st.error("❌ No expression column found")
                st.stop()

            # build gene expression vector
            expression_data = df[["gene_name", expression_col]]

            expression_data = expression_data.dropna()

            expression_data.columns = ["gene", "expression"]

            expression_data = expression_data.groupby("gene").mean()

            expression_data = expression_data.transpose()

            df = expression_data

        else:

            # =================================================
            # NORMAL GENE MATRIX FORMAT
            # =================================================

            # CASE 1
            if "IDX" in df.columns:

                df = df.set_index("IDX")

                df = df.transpose()

            # CASE 2
            elif df.columns[0] not in top_genes:

                df = df.set_index(df.columns[0])

                df = df.transpose()

        # =================================================
        # MATCH GENES
        # =================================================
        common_genes = [g for g in top_genes if g in df.columns]

        st.write(f"### Matched Genes: {len(common_genes)} / {len(top_genes)}")

        if len(common_genes) < 10:

            st.error("❌ Not enough matching genes found")

            st.stop()

        # =================================================
        # CREATE INPUT DATA
        # =================================================
        input_df = pd.DataFrame(columns=top_genes)

        for gene in top_genes:

            if gene in df.columns:

                input_df[gene] = pd.to_numeric(
                    df[gene],
                    errors="coerce"
                )

            else:

                input_df[gene] = 0

        # =================================================
        # CLEAN DATA
        # =================================================
        input_df = input_df.fillna(0)

        input_df = input_df.replace(
            [np.inf, -np.inf],
            0
        )

        st.success("✅ Data formatted successfully")

        # =================================================
        # PREDICT BUTTON
        # =================================================
        if st.button("Predict CSV"):

            pred, prob = predict(input_df)

            results_df = pd.DataFrame()

            results_df["Prediction"] = [
                "LUAD" if p == 0 else "BRCA"
                for p in pred
            ]

            results_df["Confidence"] = np.max(prob, axis=1)

            # =================================================
            # RESULTS
            # =================================================
            st.write("## Prediction Results")

            st.dataframe(results_df)

            # =================================================
            # METRICS
            # =================================================
            brca_count = (
                results_df["Prediction"] == "BRCA"
            ).sum()

            luad_count = (
                results_df["Prediction"] == "LUAD"
            ).sum()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("BRCA Samples", brca_count)

            with col2:
                st.metric("LUAD Samples", luad_count)

            # =================================================
            # DISTRIBUTION PLOT
            # =================================================
            st.write("## Prediction Distribution")

            fig, ax = plt.subplots(figsize=(8, 5))

            results_df["Prediction"].value_counts().plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Prediction Distribution")

            ax.set_xlabel("Cancer Type")

            ax.set_ylabel("Count")

            st.pyplot(fig)

            # =================================================
            # CONFIDENCE HISTOGRAM
            # =================================================
            st.write("## Confidence Distribution")

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            ax2.hist(
                results_df["Confidence"],
                bins=20
            )

            ax2.set_title("Prediction Confidence")

            ax2.set_xlabel("Confidence")

            ax2.set_ylabel("Frequency")

            st.pyplot(fig2)

            # =================================================
            # PCA
            # =================================================
            st.write("## PCA Visualization")

            scaled_data = scaler.transform(input_df)

            pca = PCA(n_components=2)

            reduced = pca.fit_transform(scaled_data)

            fig3, ax3 = plt.subplots(figsize=(8, 6))

            ax3.scatter(
                reduced[:, 0],
                reduced[:, 1]
            )

            ax3.set_title("PCA Projection")

            st.pyplot(fig3)

            # =================================================
            # UMAP
            # =================================================
            st.write("## UMAP Visualization")

            reducer = umap.UMAP(random_state=42)

            embedding = reducer.fit_transform(scaled_data)

            fig4, ax4 = plt.subplots(figsize=(8, 6))

            ax4.scatter(
                embedding[:, 0],
                embedding[:, 1]
            )

            ax4.set_title("UMAP Projection")

            st.pyplot(fig4)

# =========================================================
# DEMO SECTION
# =========================================================
st.write("---")

st.write("# Model Evaluation Demonstration")

# =========================================================
# CONFUSION MATRIX
# =========================================================
if st.button("Show Confusion Matrix"):

    sample = np.random.rand(100, len(top_genes))

    y_true = np.random.randint(0, 2, 100)

    sample_scaled = scaler.transform(sample)

    y_pred = model.predict(sample_scaled)

    cm = confusion_matrix(y_true, y_pred)

    fig5, ax5 = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax5
    )

    ax5.set_title("Confusion Matrix")

    st.pyplot(fig5)

# =========================================================
# ROC CURVE
# =========================================================
if st.button("Show ROC Curve"):

    sample = np.random.rand(100, len(top_genes))

    y_true = np.random.randint(0, 2, 100)

    sample_scaled = scaler.transform(sample)

    prob = model.predict_proba(sample_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, prob)

    roc_auc = auc(fpr, tpr)

    fig6, ax6 = plt.subplots(figsize=(6, 5))

    ax6.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    ax6.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax6.set_title("ROC Curve")

    ax6.legend()

    st.pyplot(fig6)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================
if st.button("Show Classification Report"):

    sample = np.random.rand(100, len(top_genes))

    y_true = np.random.randint(0, 2, 100)

    sample_scaled = scaler.transform(sample)

    y_pred = model.predict(sample_scaled)

    report = classification_report(
        y_true,
        y_pred
    )

    st.text(report)
