import pandas as pd
import numpy as np

def load_and_clean_hotel_data(file_path: str = "hotels.csv") -> pd.DataFrame:
    """
    Loads a local raw CSV file, cleans missing values, fixes types,
    and returns a polished DataFrame with calculated dashboard metrics.
    """
    # 1. Read local raw data (treating variations of NULL strings as proper NaNs)
    df = pd.read_csv("hotel_bookings.csv", na_values=['NULL', 'Null', 'null', ' '])
    cleaned_df = df.copy()
    
    # 2. Handle Missing Values & Data Types
    if 'children' in cleaned_df.columns:
        cleaned_df['children'] = cleaned_df['children'].fillna(0).astype(int)
        
    if 'country' in cleaned_df.columns:
        cleaned_df['country'] = cleaned_df['country'].fillna('Unknown')
        
    for col in ['agent', 'company']:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(0).astype(int)

    # --- FIX APPLIED HERE ---
    if 'reservation_status_date' in cleaned_df.columns:
        cleaned_df['reservation_status_date'] = pd.to_datetime(cleaned_df['reservation_status_date'], format='mixed')

    # 3. Filter Out Invalid Rows & Duplicates
    guest_cols = ['adults', 'children', 'babies']
    if all(col in cleaned_df.columns for col in guest_cols):
        zero_guests_filter = (cleaned_df['adults'] == 0) & (cleaned_df['children'] == 0) & (cleaned_df['babies'] == 0)
        cleaned_df = cleaned_df[~zero_guests_filter]

    cleaned_df = cleaned_df.drop_duplicates()

    # 4. Feature Engineering for Dashboard
    if 'stays_in_weekend_nights' in cleaned_df.columns and 'stays_in_week_nights' in cleaned_df.columns:
        cleaned_df["total_stay_nights"] = cleaned_df["stays_in_weekend_nights"] + cleaned_df["stays_in_week_nights"]
        cleaned_df["total_stay_nights"] = cleaned_df["total_stay_nights"].replace(0, 1)
    
    if all(col in cleaned_df.columns for col in ["adr", "total_stay_nights", "is_canceled"]):
        cleaned_df["calculated_revenue"] = cleaned_df.apply(
            lambda row: row["adr"] * row["total_stay_nights"] if row["is_canceled"] == 0 else 0, 
            axis=1
        )

    return cleaned_df