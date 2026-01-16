# continuity_page.py
# Streamlit page: Lotto 6/59 + Bonus (treated as equal 7th member)
# - Loads draws (expects draw_date, num1..num6, bonus OR draw_date, winning_numbers, bonus)
# - Computes continuity features + continuity score (regime breaks)
# - Plots continuity score and shows breakpoints
# - Persists draws + features to SQLite for reproducibility

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from typing import List

# ----------------------------
# SQLite persistence
# ----------------------------

DB_PATH = "lotto_continuity.db"

def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            draw_date TEXT PRIMARY KEY,
            num1 INTEGER, num2 INTEGER, num3 INTEGER, num4 INTEGER, num5 INTEGER, num6 INTEGER,
            bonus INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS continuity (
            draw_date TEXT PRIMARY KEY,
            sum INTEGER,
            range INTEGER,
            gap_mean REAL,
            gap_std REAL,
            gap_max INTEGER,
            consec_pairs INTEGER,
            odd INTEGER,
            even INTEGER,
            low INTEGER,
            high INTEGER,
            overlap_prev REAL,
            delta_mean_abs REAL,
            delta_max_abs REAL,
            delta_neg REAL,
            delta_zero REAL,
            delta_pos REAL,
            continuity_score REAL,
            continuity_z REAL,
            regime_break INTEGER
        )
    """)
    conn.commit()

def upsert_draws(conn: sqlite3.Connection, df7: pd.DataFrame) -> None:
    # df7 must contain: draw_date, num1..num6, bonus
    df7 = df7.copy()
    df7["draw_date"] = pd.to_datetime(df7["draw_date"]).dt.strftime("%Y-%m-%d")
    cols = ["draw_date","num1","num2","num3","num4","num5","num6","bonus"]
    df7 = df7[cols]

    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO draws (draw_date,num1,num2,num3,num4,num5,num6,bonus)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(draw_date) DO UPDATE SET
          num1=excluded.num1, num2=excluded.num2, num3=excluded.num3, num4=excluded.num4,
          num5=excluded.num5, num6=excluded.num6, bonus=excluded.bonus
    """, df7.values.tolist())
    conn.commit()

def upsert_continuity(conn: sqlite3.Connection, df_out: pd.DataFrame) -> None:
    # df_out must contain draw_date + continuity columns
    df_out = df_out.copy()
    df_out["draw_date"] = pd.to_datetime(df_out["draw_date"]).dt.strftime("%Y-%m-%d")

    insert_cols = [
        "draw_date","sum","range","gap_mean","gap_std","gap_max","consec_pairs",
        "odd","even","low","high",
        "overlap_prev","delta_mean_abs","delta_max_abs","delta_neg","delta_zero","delta_pos",
        "continuity_score","continuity_z","regime_break"
    ]
    df_out = df_out[insert_cols].copy()
    df_out["regime_break"] = df_out["regime_break"].fillna(False).astype(bool).astype(int)

    cur = conn.cursor()
    cur.executemany(f"""
        INSERT INTO continuity ({",".join(insert_cols)})
        VALUES ({",".join(["?"] * len(insert_cols))})
        ON CONFLICT(draw_date) DO UPDATE SET
          sum=excluded.sum,
          range=excluded.range,
          gap_mean=excluded.gap_mean,
          gap_std=excluded.gap_std,
          gap_max=excluded.gap_max,
          consec_pairs=excluded.consec_pairs,
          odd=excluded.odd,
          even=excluded.even,
          low=excluded.low,
          high=excluded.high,
          overlap_prev=excluded.overlap_prev,
          delta_mean_abs=excluded.delta_mean_abs,
          delta_max_abs=excluded.delta_max_abs,
          delta_neg=excluded.delta_neg,
          delta_zero=excluded.delta_zero,
          delta_pos=excluded.delta_pos,
          continuity_score=excluded.continuity_score,
          continuity_z=excluded.continuity_z,
          regime_break=excluded.regime_break
    """, df_out.values.tolist())
    conn.commit()

# ----------------------------
# Continuity computations (7-number equal-member)
# ----------------------------

def _sorted_state(nums7: List[int]) -> List[int]:
    s = sorted(int(v) for v in nums7)
    if len(s) != 7:
        raise ValueError("Expected exactly 7 values (6 main + bonus).")
    for v in s:
        if v < 1 or v > 59:
            raise ValueError(f"Out of range 1..59: {v}")
    return s

def compute_continuity_features(df: pd.DataFrame, cols7: List[str]) -> pd.DataFrame:
    states = []
    for _, row in df.iterrows():
        nums7 = [int(row[c]) for c in cols7]
        states.append(_sorted_state(nums7))

    feats = []
    for i, now in enumerate(states):
        gaps = np.diff(now)
        base = {
            "sum": int(np.sum(now)),
            "range": int(now[-1] - now[0]),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "gap_max": int(np.max(gaps)),
            "consec_pairs": int(np.sum(gaps == 1)),
        }
        arr = np.array(now, dtype=int)
        base.update({
            "odd": int(np.sum(arr % 2 == 1)),
            "even": int(np.sum(arr % 2 == 0)),
            "low": int(np.sum(arr <= 29)),
            "high": int(np.sum(arr >= 30)),
        })

        if i == 0:
            base.update({
                "overlap_prev": np.nan,
                "delta_mean_abs": np.nan,
                "delta_max_abs": np.nan,
                "delta_neg": np.nan,
                "delta_zero": np.nan,
                "delta_pos": np.nan,
            })
        else:
            prev = states[i - 1]
            base["overlap_prev"] = float(len(set(now).intersection(set(prev))))
            d = np.array(now, dtype=int) - np.array(prev, dtype=int)
            base.update({
                "delta_mean_abs": float(np.mean(np.abs(d))),
                "delta_max_abs": float(np.max(np.abs(d))),
                "delta_neg": float(np.sum(d < 0)),
                "delta_zero": float(np.sum(d == 0)),
                "delta_pos": float(np.sum(d > 0)),
            })

        feats.append(base)

    return pd.DataFrame(feats, index=df.index)

def compute_continuity_score(feat_df: pd.DataFrame, window: int = 30, z_threshold: float = 3.0) -> pd.DataFrame:
    use_cols = [
        "overlap_prev","sum","range","gap_std","gap_max","consec_pairs",
        "delta_mean_abs","delta_max_abs","delta_neg","delta_zero","delta_pos",
        "odd","low"
    ]
    use_cols = [c for c in use_cols if c in feat_df.columns]
    X = feat_df[use_cols].astype(float)

    mean_recent = X.rolling(window).mean()
    mean_prior = X.shift(window).rolling(window).mean()
    std_recent = X.rolling(window).std().replace(0, np.nan)

    diff = (mean_recent - mean_prior) / std_recent
    score = np.sqrt(np.nansum(diff.values ** 2, axis=1))
    score_s = pd.Series(score, index=feat_df.index)

    mu = score_s.rolling(window).mean()
    sd = score_s.rolling(window).std().replace(0, np.nan)
    z = (score_s - mu) / sd

    out = pd.DataFrame({
        "continuity_score": score_s,
        "continuity_z": z,
        "regime_break": (z >= z_threshold)
    }, index=feat_df.index)
    return out

# ----------------------------
# Data preparation helpers
# ----------------------------

def ensure_num_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - draw_date + num1..num6 + bonus
      - draw_date + winning_numbers + bonus   (winning_numbers is "02 18 24 27 29 39")
    Returns df with draw_date + num1..num6 + bonus as ints, sorted by draw_date asc.
    """
    df = df.copy()

    if "winning_numbers" in df.columns and not all(c in df.columns for c in ["num1","num2","num3","num4","num5","num6"]):
        parts = df["winning_numbers"].astype(str).str.strip().str.split(r"\s+", expand=True)
        if parts.shape[1] < 6:
            raise ValueError("winning_numbers does not split into 6 values.")
        parts = parts.iloc[:, :6].astype(int)
        parts.columns = ["num1","num2","num3","num4","num5","num6"]
        df = pd.concat([df.drop(columns=[]), parts], axis=1)

    required = ["draw_date","num1","num2","num3","num4","num5","num6","bonus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["draw_date"] = pd.to_datetime(df["draw_date"])
    for c in ["num1","num2","num3","num4","num5","num6","bonus"]:
        df[c] = df[c].astype(int)

    df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    return df

# ----------------------------
# Streamlit page
# ----------------------------

def render_continuity_page(df_raw: pd.DataFrame) -> None:
    st.header("Continuity Dashboard (Lotto 6/59 + Bonus)")

    with st.expander("Settings", expanded=True):
        window = st.slider("Continuity window (draws)", min_value=10, max_value=120, value=30, step=5)
        z_threshold = st.slider("Regime break threshold (z-score)", min_value=1.5, max_value=6.0, value=3.0, step=0.5)

        persist = st.checkbox("Persist draws + continuity to SQLite", value=True)
        db_path = st.text_input("SQLite DB path", value=DB_PATH)

    df = ensure_num_columns(df_raw)

    cols7 = ["num1","num2","num3","num4","num5","num6","bonus"]
    feat_df = compute_continuity_features(df, cols7)
    score_df = compute_continuity_score(feat_df, window=window, z_threshold=z_threshold)

    df_out = pd.concat([df, feat_df, score_df], axis=1)

    # Persist
    if persist:
        conn = sqlite3.connect(db_path)
        init_db(conn)
        upsert_draws(conn, df_out[["draw_date"] + cols7])
        upsert_continuity(conn, df_out[["draw_date"] + feat_df.columns.tolist() + score_df.columns.tolist()])
        conn.close()
        st.caption(f"Persisted to SQLite: {db_path}")

    # Plot continuity score
    st.subheader("Continuity score over time")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_out["draw_date"], df_out["continuity_score"], linewidth=1)
    ax.set_xlabel("Draw Date")
    ax.set_ylabel("Continuity Score")
    ax.set_title("Continuity Score (higher = more likely regime shift)")
    st.pyplot(fig)

    # Plot z-score with break markers
    st.subheader("Regime breaks (z-score)")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df_out["draw_date"], df_out["continuity_z"], linewidth=1)
    breaks = df_out[df_out["regime_break"] == True]
    if not breaks.empty:
        ax2.scatter(breaks["draw_date"], breaks["continuity_z"], s=25)
    ax2.axhline(y=z_threshold, linestyle="--", linewidth=1)
    ax2.set_xlabel("Draw Date")
    ax2.set_ylabel("Continuity Z-score")
    ax2.set_title("Continuity Z-score with break threshold")
    st.pyplot(fig2)

    # Break table
    st.subheader("Detected breakpoints")
    show_cols = ["draw_date","continuity_score","continuity_z","overlap_prev","sum","range","gap_std","delta_mean_abs","delta_max_abs"]
    show_cols = [c for c in show_cols if c in df_out.columns]
    st.dataframe(
        df_out[df_out["regime_break"] == True][show_cols].sort_values("draw_date", ascending=False),
        use_container_width=True
    )

    # Recent features
    st.subheader("Most recent draws (with continuity features)")
    tail_cols = ["draw_date"] + cols7 + ["overlap_prev","continuity_score","continuity_z","regime_break"]
    tail_cols = [c for c in tail_cols if c in df_out.columns]
    st.dataframe(df_out[tail_cols].tail(20).sort_values("draw_date", ascending=False), use_container_width=True)

# ----------------------------
# Example usage in app.py
# ----------------------------
# In your main Streamlit app, after loading df_raw:
#
#   from continuity_page import render_continuity_page
#   render_continuity_page(df_raw)
#
# df_raw should include either:
#   draw_date, winning_numbers, bonus
# OR
#   draw_date, num1..num6, bonus
