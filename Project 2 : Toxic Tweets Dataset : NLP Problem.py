import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib # Library to load our saved models

# --- Caching: Load the pre-trained models and data ONCE ---
@st.cache_resource
def load_artifacts(filepath="ml_artifacts.joblib"):
    """Loads the pre-trained models, vectorizers, and test data."""
    try:
        artifacts = joblib.load(filepath)
        return artifacts
    except FileNotFoundError:
        return None

# --- Main Streamlit App Interface ---
st.set_page_config(layout="wide")
st.title("⚡️ Fast Toxic Tweets Classification Dashboard")
st.markdown("This app loads pre-trained models to deliver results instantly.")

# Load the artifacts. This is now extremely fast due to caching.
artifacts = load_artifacts()

if artifacts is None:
    st.error("Could not find 'ml_artifacts.joblib'.")
    st.info("Please run the `train_and_save_models.py` script first to generate the necessary file.")
else:
    st.success("Pre-trained models and data loaded successfully!")

    st.sidebar.header("Controls")
    vectorizer_choice = st.sidebar.selectbox("1. Choose a Vectorizer", list(artifacts["models"].keys()))
    model_choice = st.sidebar.selectbox("2. Choose a Model", list(artifacts["models"][vectorizer_choice].keys()))

    st.header(f"Results for: {model_choice} with {vectorizer_choice}")

    # --- Retrieve the correct pre-trained items from the loaded artifacts ---
    model = artifacts["models"][vectorizer_choice][model_choice]
    vectorizer = artifacts["vectorizers"][vectorizer_choice]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    # --- Perform fast, on-the-fly evaluation ---
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

    # --- Display Metrics and Plots ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    with col2:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

    st.subheader("ROC-AUC Curve")
    fig_roc, ax_roc = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
