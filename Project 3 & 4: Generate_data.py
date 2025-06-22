#
# This is the fully corrected version. Copy this entire block.
#
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_regression, make_friedman1

# --- Create the 'data' directory if it doesn't exist ---
if not os.path.exists('data'):
    os.makedirs('data')
    print("Created 'data' directory.")

# --- Project 3: Linear Data Generation (No changes here) ---
print("\nGenerating data for Project 3...")
X_train_p1, y_train_p1 = make_regression(
    n_samples=200, 
    n_features=2, 
    noise=15, 
    random_state=42
)
p1_train_df = pd.DataFrame(X_train_p1, columns=['Sensor 1', 'Sensor 2'])
p1_train_df['Health_Target'] = y_train_p1

X_test_p1, y_test_p1 = make_regression(
    n_samples=50, 
    n_features=2, 
    noise=15, 
    random_state=0 
)
p1_test_df = pd.DataFrame(X_test_p1, columns=['Sensor 1', 'Sensor 2'])
p1_test_df['Health_Target'] = y_test_p1

p1_train_df.to_csv('data/p1-train.csv', index=False)
p1_test_df.to_csv('data/p1-test.csv', index=False)
print("Successfully created 'data/p1-train.csv' and 'data/p1-test.csv'")


# --- Project 4: Non-Linear Data Generation (CORRECTED SECTION) ---
print("\nGenerating data for Project 4...")

# Generate training data
# FIX: Generate 5 features as required by make_friedman1
X_full_train, y_train_p2 = make_friedman1(
    n_samples=300, 
    n_features=5, # <-- CHANGED from 3 to 5
    noise=1.0, 
    random_state=42
)
# FIX: But only select the first 3 features to match the project requirement
X_train_p2 = X_full_train[:, :3] 
p2_train_df = pd.DataFrame(X_train_p2, columns=['Sensor A', 'Sensor B', 'Sensor C'])
p2_train_df['Lifespan_Target'] = y_train_p2

# Generate testing data
# FIX: Generate 5 features as required by make_friedman1
X_full_test, y_test_p2 = make_friedman1(
    n_samples=75, 
    n_features=5, # <-- CHANGED from 3 to 5
    noise=1.0, 
    random_state=0 
)
# FIX: But only select the first 3 features
X_test_p2 = X_full_test[:, :3]
p2_test_df = pd.DataFrame(X_test_p2, columns=['Sensor A', 'Sensor B', 'Sensor C'])
p2_test_df['Lifespan_Target'] = y_test_p2

# Save to CSV (This part remains the same)
p2_train_df.to_csv('data/p2-train.csv', index=False)
p2_test_df.to_csv('data/p2-test.csv', index=False)
print("Successfully created 'data/p2-train.csv' and 'data/p2-test.csv'")

print("\nData generation complete.")
