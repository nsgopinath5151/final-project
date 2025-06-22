import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import time

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC # Import LinearSVC for a faster option
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

plt.style.use('ggplot')

# --- Caching Functions ---

@st.cache_data
def load_and_prepare_data(use_subset=False):
    """
    Loads Kannada MNIST data. If use_subset is True, it only loads 10% of the data
    for much faster development and testing.
    """
    data_dir = 'data'
    # ... (loading logic remains the same) ...
    X_train_raw = np.load(os.path.join(data_dir, 'X_kannada_MNIST_train.npz'))['arr_0']
    y_train = np.load(os.path.join(data_dir, 'y_kannada_MNIST_train.npz'))['arr_0']
    X_test_raw = np.load(os.path.join(data_dir, 'X_kannada_MNIST_test.npz'))['arr_0']
    y_test = np.load(os.path.join(data_dir, 'y_kannada_MNIST_test.npz'))['arr_0']

    if use_subset:
        # Use only 6000 training samples and 1000 test samples
        X_train_raw, y_train = X_train_raw[:6000], y_train[:6000]
        X_test_raw, y_test = X_test_raw[:1000], y_test[:1000]

    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255.0
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255.0
    
    return X_train, X_test, y_train, y_test

@st.cache_resource
def train_model(model_name, n_components, X_train, y_train):
    """
    Performs PCA and trains a selected model. Caches the fitted PCA and model.
    """
    # Use LinearSVC for a much faster SVM alternative if desired.
    # Note: LinearSVC doesn't directly support probability=True, so ROC AUC requires a workaround.
    # For this project, we'll stick to the original but be aware of this option.
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-NN Classifier": KNeighborsClassifier(n_jobs=-1),
        "SVM": SVC(random_state=42, probability=True, C=1.0) # Standard, but slow
    }
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    model = models[model_name]
    model.fit(X_train_pca, y_train)
    
    return pca, model

# --- Streamlit UI and Main Logic ---

st.set_page_config(layout="wide")
st.title("Kannada-MNIST Digit Classifier")

st.sidebar.header("1. Data Settings")
use_subset = st.sidebar.checkbox(
    "Use small subset (10%) for speed", 
    value=True, # Default to using the fast subset
    help="Train on only 6,000 images instead of 60,000 to get results quickly."
)

# Load data based on checkbox
X_train, X_test, y_train, y_test = load_and_prepare_data(use_subset)

if X_train is not None:
    st.success("Dataset loaded successfully!")
    col1, col2 = st.columns(2)
    col1.metric("Training Samples", f"{X_train.shape[0]:,}")
    col2.metric("Test Samples", f"{X_test.shape[0]:,}")

    st.sidebar.header("2. Classifier Settings")
    n_components = st.sidebar.slider(
        "Number of PCA Components", 
        min_value=5, max_value=50, value=20, step=5
    )
    
    model_name = st.sidebar.selectbox(
        "Choose a Classifier",
        ("Decision Tree", "Random Forest", "Gaussian Naive Bayes", "K-NN Classifier", "SVM")
    )

    if model_name == "SVM" and not use_subset:
        st.sidebar.error("🚨 Warning: Running SVM on the full dataset will be **EXTREMELY SLOW** (potentially 5-10+ minutes). It's recommended to use the subset for SVM.")

    if st.sidebar.button("🚀 Train and Evaluate Model", type="primary"):
        with st.spinner(f"Training {model_name}... Please wait."):
            start_time = time()
            pca, model = train_model(model_name, n_components, X_train, y_train)
            end_time = time()

        st.info(f"Model trained in **{end_time - start_time:.2f} seconds**.")

        st.header(f"Evaluation Results for {model_name}")
        
        # Predictions
        X_test_pca = pca.transform(X_test)
        y_pred = model.predict(X_test_pca)
        
        # Layout for results
        res1, res2 = st.columns([1, 1.5]) # Give more space to the matrix

        with res1:
            st.subheader("Metrics")
            try:
                y_pred_proba = model.predict_proba(X_test_pca)
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                st.metric(label="RoC-AUC Score", value=f"{auc_score:.4f}")
            except AttributeError:
                st.info("This model does not support `predict_proba` for ROC AUC.")
            
            # Using st.text to show formatted report
            report = classification_report(y_test, y_pred)
            st.text_area("Classification Report", report, height=350)

        with res2:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            st.pyplot(fig)
