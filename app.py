# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from clean_data import load_and_clean_hotel_data

# PAGE CONFIG
st.set_page_config(page_title="Travel Analytics & AI Dashboard", layout="wide")

# TITLE
st.title("Hotel Booking Center: Analytics & Prediction")
st.markdown("---")

# AUTOMATIC LOAD & CACHE DATA
@st.cache_data
def get_dashboard_data():
    return load_and_clean_hotel_data(r"C:\Users\ASUS\travel-analytics-project\hotel_bookings.csv")

df = get_dashboard_data()

# CREATE TABS FOR NAVIGATION
tab1, tab2 = st.tabs(["Analytics Dashboard", "🔮 Cancellation Predictor"])

# ==============================================================================
# TAB 1: ANALYTICS DASHBOARD
# ==============================================================================
with tab1:
    st.sidebar.header("Filter Options")
    hotel_type = st.sidebar.multiselect("Select Hotel Type", options=df["hotel"].unique(), default=df["hotel"].unique())
    filtered_df = df[df["hotel"].isin(hotel_type)]

    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')
    csv_bytes = convert_df_to_csv(df)
    st.sidebar.download_button("Download cleaned_hotel_bookings.csv", data=csv_bytes, file_name="cleaned_hotel_bookings.csv", mime="text/csv")

    st.subheader("Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    total_bookings = len(filtered_df)
    cancelled = filtered_df["is_canceled"].sum()
    revenue = filtered_df["calculated_revenue"].sum()

    kpi_col1.metric("Total Bookings", f"{total_bookings:,}")
    cancel_rate = ((cancelled / total_bookings) * 100) if total_bookings > 0 else 0
    kpi_col2.metric("Cancelled Bookings", f"{cancelled:,}", delta=f"{cancel_rate:.1f}% Cancel Rate", delta_color="inverse")
    kpi_col3.metric("Total Revenue", f"${revenue:,.2f}")

    st.markdown("---")

    grid_col_left, grid_col_right = st.columns([3, 2])
    with grid_col_left:
        st.subheader("Monthly Booking Trends")
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        monthly = filtered_df["arrival_date_month"].value_counts().reindex(month_order).fillna(0)
        st.line_chart(monthly, color="#29b5e8")
        
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(filtered_df, use_container_width=True, height=400)

    with grid_col_right:
        st.subheader("Top Customer Countries")
        st.bar_chart(filtered_df["country"].value_counts().head(10), color="#ff4b4b")
        st.subheader("Hotel Segment Distribution")
        st.bar_chart(filtered_df["hotel"].value_counts(), color="#1c83e1")


# ==============================================================================
# TAB 2: LIVE MACHINE LEARNING PREDICTION & EXPLAINABILITY
# ==============================================================================
with tab2:
    try:
        # Load the trained model artifacts
        with open("hotel_cancellation_model.pkl", "rb") as f:
            artifacts = pickle.load(f)
            
        model = artifacts["model"]
        encoders = artifacts["encoders"]
        test_accuracy = artifacts["test_accuracy"]
        importance_df = artifacts["feature_importance"]
        
        # Split screen layout: Left = User Input Form, Right = Model Explainability
        predict_col, explain_col = st.columns([1, 1])
        
        with predict_col:
            st.subheader("🔮 Predict New Booking Status")
            st.write("Modify the booking parameters below to evaluate risk.")
            
            lead_time = st.slider("Lead Time (Days between booking & arrival)", 0, 365, 30)
            deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
            market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"])
            customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
            special_requests = st.number_input("Total Special Requests", min_value=0, max_value=5, value=0)
            prev_cancellations = st.number_input("Previous Cancellations by Guest", min_value=0, max_value=20, value=0)
            booking_changes = st.number_input("Number of Booking Modifications", min_value=0, max_value=10, value=0)
            parking_spaces = st.selectbox("Required Parking Spaces", [0, 1, 2])

            if st.button("Calculate Cancellation Probability", type="primary"):
                input_data = pd.DataFrame([{
                    'lead_time': lead_time, 'deposit_type': deposit_type,
                    'total_of_special_requests': special_requests, 'previous_cancellations': prev_cancellations,
                    'booking_changes': booking_changes, 'market_segment': market_segment,
                    'customer_type': customer_type, 'required_car_parking_spaces': parking_spaces
                }])
                
                for col in ['deposit_type', 'market_segment', 'customer_type']:
                    input_data[col] = encoders[col].transform(input_data[col].astype(str))
                
                probabilities = model.predict_proba(input_data)[0]
                cancel_probability = probabilities[1] * 100
                
                st.markdown("---")
                if cancel_probability < 35:
                    st.success(f"### Safe Booking! ({cancel_probability:.1f}% Risk)")
                elif cancel_probability < 70:
                    st.warning(f"### Moderate Risk! ({cancel_probability:.1f}% Risk)")
                else:
                    st.error(f"### High Risk Booking! ({cancel_probability:.1f}% Risk)")

        with explain_col:
            st.subheader("Model Performance & Transparency")
            st.write("This section proves how the machine learning model evaluates data.")
            
            # Display Validation Metric
            st.metric(
                label="Model Verification Accuracy (Evaluated on Unseen 20% Test Split)", 
                value=f"{test_accuracy * 100:.2f}%"
            )
            st.caption("A higher score on hidden data proves the algorithm is learning generalized business trends rather than memorizing historical rows.")
            
            st.markdown("---")
            st.write("##### Feature Importance (Which factors matter most to the AI?)")
            
            # Use Streamlit's native horizontal bar chart to showcase feature weights
            # Setting x='Importance' and y='Feature' shows a tidy ranking
            st.bar_chart(
                data=importance_df, 
                x="Importance", 
                y="Feature", 
                color="#29b5e8"
            )
            st.caption("This chart displays the mathematical weight the Random Forest classifier assigns to each feature when evaluating reservation patterns.")
                
    except FileNotFoundError:
        st.info("Model file not found. Please execute `python train_model.py` in your VS Code terminal first!")