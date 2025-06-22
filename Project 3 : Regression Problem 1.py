import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Project 3: Linear Regression",
    page_icon="🩺",
    layout="centered"
)

# --- Function to construct absolute path ---
def get_absolute_path(relative_path):
    """Constructs an absolute path from a relative path."""
    return os.path.join(os.path.dirname(__file__), relative_path)

# --- Caching Data Loading ---
@st.cache_data
def load_data(train_path, test_path):
    """Loads data using absolute paths and handles potential file errors."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except FileNotFoundError:
        st.error(f"Error: Data files not found. The script looked for:")
        st.code(train_path)
        st.code(test_path)
        st.info("Please ensure these files exist and the `data` folder is in the same directory as the app. Run `generate_data.py` if needed.")
        return None, None

# --- Main App ---
st.title("🩺 Project 3: Health Prediction (Linear Problem)")

st.markdown("""
This application trains and evaluates two linear regression models to predict an organism's health based on sensor data.
- **Dataset:** `p1-train.csv` and `p1-test.csv`
- **Goal:** Predict the `Health_Target` column.
- **Metrics:** Mean Squared Error (MSE) and Mean Absolute Error (MAE).
""")

# Construct absolute paths for the data files
TRAIN_FILE_PATH = get_absolute_path('data/p1-train.csv')
TEST_FILE_PATH = get_absolute_path('data/p1-test.csv')

# Load data
train_df, test_df = load_data(TRAIN_FILE_PATH, TEST_FILE_PATH)

# **FIXED LINE**: Check if dataframes are not None
if train_df is not None and test_df is not None:
    st.subheader("Training Data Preview")
    st.dataframe(train_df.head())

    # Prepare data for modeling
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    st.info("Click the button below to train the models on the training data and report the performance metrics on the test data.")

    # Button to trigger model training and evaluation
    if st.button("Train and Evaluate Linear Models", type="primary"):
        with st.spinner("Running models..."):
            time.sleep(1) # Simulate processing time

            # --- Model 1: Linear Regression ---
            st.subheader("1. Linear Regression")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X_test)
            lr_mse = mean_squared_error(y_test, y_pred_lr)
            lr_mae = mean_absolute_error(y_test, y_pred_lr)
            
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{lr_mse:.4f}")
            col2.metric("Mean Absolute Error (MAE)", f"{lr_mae:.4f}")

            # --- Model 2: Support Vector Regression (Linear) ---
            st.subheader("2. Support Vector Regression (Linear Kernel)")
            svr_model = SVR(kernel='linear')
            svr_model.fit(X_train, y_train)
            y_pred_svr = svr_model.predict(X_test)
            svr_mse = mean_squared_error(y_test, y_pred_svr)
            svr_mae = mean_absolute_error(y_test, y_pred_svr)

            col3, col4 = st.columns(2)
            col3.metric("Mean Squared Error (MSE)", f"{svr_mse:.4f}")
            col4.metric("Mean Absolute Error (MAE)", f"{svr_mae:.4f}")

        st.success("Evaluation Complete!")
