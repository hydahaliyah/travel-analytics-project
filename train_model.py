# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from clean_data import load_and_clean_hotel_data

print("Loading and preparing data...")
df = load_and_clean_hotel_data("hotel_bookings.csv")

# 1. Select predictive features
features = [
    'lead_time', 'deposit_type', 'total_of_special_requests', 
    'previous_cancellations', 'booking_changes', 'market_segment',
    'customer_type', 'required_car_parking_spaces'
]

X = df[features].copy()
y = df['is_canceled']

# 2. Encode categorical text data into numbers
encoders = {}
categorical_cols = ['deposit_type', 'market_segment', 'customer_type']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le  

# 3. IMPLEMENTING THE 80-20 SPLIT (Crucial Data Science Guardrail)
# Train set = 80% of data, Test set = 20% of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset split complete: Train size = {len(X_train)}, Test size = {len(X_test)}")
print("Training the Random Forest model on the 80% Train Set...")

# Initialize and train the model ONLY on the training data
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
model.fit(X_train, y_train)

# 4. PROVING IT WORKS: Evaluate accuracy strictly using the unseen 20% Test Set
test_accuracy = model.score(X_test, y_test)
print(f"Unseen Test Set Accuracy: {test_accuracy * 100:.2f}%")

# 5. IMPLEMENTING FEATURE IMPORTANCE
# Extract how much weight the Random Forest assigned to each column
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=True) # Sorted for a nice horizontal bar chart

# 6. Save all artifacts to your local binary file
model_artifacts = {
    "model": model,
    "encoders": encoders,
    "features": features,
    "test_accuracy": test_accuracy,
    "feature_importance": feature_importance_df  # Saving this to display in Streamlit
}

with open("hotel_cancellation_model.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)

print("Model and Feature Importances saved successfully as 'hotel_cancellation_model.pkl'!")