import streamlit as st
import pandas as pd
import requests
from continuity_page import render_continuity_page

API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"

@st.cache_data
def load_df_raw():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def prepare_lotto_6_59(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Ensure required columns exist
    required = {"draw_date", "winning_numbers", "bonus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns from API: {missing}")

    df["draw_date"] = pd.to_datetime(df["draw_date"])

    parts = df["winning_numbers"].astype(str).str.strip().str.split(r"\s+", expand=True)
    if parts.shape[1] < 6:
        raise ValueError("winning_numbers did not split into 6 values.")

    parts = parts.iloc[:, :6].astype(int)
    parts.columns = ["num1","num2","num3","num4","num5","num6"]

    df["bonus"] = df["bonus"].astype(int)

    df = pd.concat([df[["draw_date","winning_numbers","bonus"]], parts], axis=1)
    df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    return df

# ---- Streamlit app starts here ----
st.title("Lotto 6/59 Continuity Dashboard")

df_raw = load_df_raw()
df = prepare_lotto_6_59(df_raw)

# Call the continuity page using prepared data
render_continuity_page(df)
