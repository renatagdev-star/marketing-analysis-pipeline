# -*- coding: utf-8 -*-
"""pipeline cleaned for GitHub"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load .env (lokalno) ili Streamlit Secrets (deployment)
load_dotenv()

# credentials se čitaju iz environmenta – NEMA hard-coded lozinki!
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB   = os.getenv("PG_DB")

# connect to Neon
engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

def _safe_div(numer, denom):
    """Vectorized division that avoids divide-by-zero errors."""
    return np.where(denom == 0, np.nan, numer / denom)

def run_pipeline(new_batch_df: pd.DataFrame):
    """
    1. Append new batch into stg_campaigns_raw
    2. Read full staging
    3. Clean + dedupe
    4. Feature engineering
    5. Refresh fact_campaigns_clean
    """

    # 1. append batch to staging in Neon and read full staging
    with engine.begin() as conn:
        # get actual column order from stg_campaigns_raw in DB
        table_info_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'stg_campaigns_raw'
            ORDER BY ordinal_position;
        """
        existing_cols = pd.read_sql(text(table_info_query), conn)["column_name"].tolist()

        # align incoming batch to staging schema
        batch_cols_to_use = [c for c in new_batch_df.columns if c in existing_cols]
        df_batch_aligned = new_batch_df[batch_cols_to_use].copy()

        # append into staging
        df_batch_aligned.to_sql(
            "stg_campaigns_raw",
            con=conn,
            if_exists="replace",
            index=False
        )

        # read full staging after append
        df_raw = pd.read_sql(text("SELECT * FROM stg_campaigns_raw;"), conn)

    # 2. cleaning

    # drop duplicate ".1" columns that are identical to the base column
    cols_to_drop = []
    for col in list(df_raw.columns):
        if col.endswith(".1"):
            base = col[:-2]
            if base in df_raw.columns and df_raw[base].equals(df_raw[col]):
                cols_to_drop.append(col)
    if cols_to_drop:
        df_raw = df_raw.drop(columns=cols_to_drop)

    # drop exact duplicate rows
    df_clean = df_raw.drop_duplicates()

    # drop rows missing critical business fields
    required_cols = ["c_date", "campaign_name", "impressions", "clicks", "mark_spent", "revenue"]
    existing_required = [c for c in required_cols if c in df_clean.columns]
    df_clean = df_clean.dropna(subset=existing_required)

    # impressions must be > 0
    if "impressions" in df_clean.columns:
        df_clean = df_clean[df_clean["impressions"] > 0]

    # numeric columns must be >= 0
    numeric_cols = ["impressions", "clicks", "leads", "orders", "mark_spent", "revenue"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] >= 0]

    # normalize date
    if "c_date" in df_clean.columns:
        df_clean["c_date"] = pd.to_datetime(df_clean["c_date"], errors="coerce")
        df_clean = df_clean.dropna(subset=["c_date"])
        df_clean["c_date"] = df_clean["c_date"].dt.strftime("%Y-%m-%d")

    # dedupe by id (keep latest by date)
    if "id" in df_clean.columns:
        tmp = df_clean.copy()
        tmp["__c_date_dt"] = pd.to_datetime(tmp["c_date"], errors="coerce")
        tmp = tmp.sort_values("__c_date_dt")
        df_clean = tmp.drop_duplicates(subset=["id"], keep="last").drop(columns="__c_date_dt")

    # 3. feature engineering
    df_feat = df_clean.copy()
    dt_tmp = pd.to_datetime(df_feat["c_date"], errors="coerce")

    # marketing KPIs
    df_feat["CTR_pct"] = _safe_div(df_feat.get("clicks", np.nan), df_feat.get("impressions", np.nan)) * 100
    df_feat["CPC"] = _safe_div(df_feat.get("mark_spent", np.nan), df_feat.get("clicks", np.nan))
    df_feat["CPA"] = _safe_div(df_feat.get("mark_spent", np.nan), df_feat.get("orders", np.nan))
    df_feat["ConversionRate_pct"] = _safe_div(df_feat.get("orders", np.nan), df_feat.get("clicks", np.nan)) * 100
    df_feat["ROAS"] = _safe_div(df_feat.get("revenue", np.nan), df_feat.get("mark_spent", np.nan))
    df_feat["Profit"] = df_feat.get("revenue", np.nan) - df_feat.get("mark_spent", np.nan)
    df_feat["LeadRate_pct"] = _safe_div(df_feat.get("leads", np.nan), df_feat.get("clicks", np.nan)) * 100

    # time features
    df_feat["Year"] = dt_tmp.dt.year
    df_feat["Month"] = dt_tmp.dt.month
    df_feat["Weekday"] = dt_tmp.dt.day_name()
    df_feat["Is_Weekend"] = dt_tmp.dt.weekday.isin([5, 6]).astype(int)

    # round readable KPIs
    round_cols = ["CTR_pct", "ConversionRate_pct", "LeadRate_pct", "CPC", "CPA", "ROAS", "Profit"]
    for c in round_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].round(2)

    # 4. rename columns to match Postgres table column names (all lowercase snake_case)
    rename_map = {
        "CTR_pct": "ctr_pct",
        "CPC": "cpc",
        "CPA": "cpa",
        "ConversionRate_pct": "conversionrate_pct",
        "ROAS": "roas",
        "Profit": "profit",
        "LeadRate_pct": "leadrate_pct",
        "Year": "year",
        "Month": "month",
        "Weekday": "weekday",
        "Is_Weekend": "is_weekend",
    }
    df_out = df_feat.rename(columns=rename_map)

    # 5. write snapshot to fact_campaigns_clean
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE fact_campaigns_clean;"))
        df_out.to_sql(
            "fact_campaigns_clean",
            con=conn,
            if_exists="append",
            index=False
        )

    print("✅ Pipeline finished successfully.")
    print(f" - Rows in staging: {len(df_raw)}")
    print(f" - Rows in fact_campaigns_clean: {len(df_out)}")
    print(" - Columns in fact_campaigns_clean:")
    print(df_out.columns.tolist())

    return df_out.head()
