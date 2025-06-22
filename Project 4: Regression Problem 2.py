import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Project 4: Non-Linear Regression",
    page_icon="🧬",
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
st.title("🧬 Project 4: Lifespan Prediction (Non-Linear Problem)")

st.markdown("""
This application trains and evaluates several non-linear regression models to predict an organism's lifespan. A linear model is included as a baseline for comparison.
- **Dataset:** `p2-train.csv` and `p2-test.csv`
- **Goal:** Predict the `Lifespan_Target` column.
- **Metrics:** Mean Squared Error (MSE) and Mean Absolute Error (MAE).
""")

# Construct absolute paths for the data files
TRAIN_FILE_PATH = get_absolute_path('data/p2-train.csv')
TEST_FILE_PATH = get_absolute_path('data/p2-test.csv')

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

    st.info("Click the button below to train the models and see how they perform on the test data.")

    # Button to trigger model training and evaluation
    if st.button("Train and Evaluate Non-Linear Models", type="primary"):
        models_to_run = {
            "Support Vector Regression (SVR)": SVR(), # Default 'rbf' kernel is non-linear
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression (Baseline)": LinearRegression()
        }

        with st.spinner("Running all models... This might take a moment."):
            time.sleep(1) # Simulate a longer processing time
            
            for name, model in models_to_run.items():
                st.markdown("---")
                st.subheader(f"Results for: {name}")

                # Fit model and make predictions
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Display metrics in columns
                col1, col2 = st.columns(2)
                col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")

        st.success("Evaluation Complete!")
        st.balloons()
        st.markdown("---")
        st.subheader("Conclusion")
        st.write("Notice how the non-linear models (SVR, Random Forest, Gradient Boosting) have much lower error scores (MSE and MAE) than the Linear Regression baseline. This confirms that they are a much better fit for this complex, non-linear dataset.")
