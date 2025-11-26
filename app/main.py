# file: app/main.py
from fastapi import FastAPI, Query, HTTPException
from typing import Optional, List
from dotenv import load_dotenv
import os
import io
import pandas as pd
from functools import lru_cache
from datetime import datetime
from pydantic import BaseModel

# Try to import azure storage - optional if you will read directly from blob
try:
    from azure.storage.blob import BlobServiceClient, BlobClient
except Exception:
    BlobServiceClient = None
    BlobClient = None

load_dotenv()  # load .env in development if present

app = FastAPI(title="Forecast API - tailored to uploaded CSV")

# Environment settings
AZ_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  # optional for blob access
AZ_CONTAINER = os.getenv("AZURE_CONTAINER", "")
AZ_BLOB_PATH = os.getenv("AZURE_BLOB_PATH", "")
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", "final_predictions.csv")

# If you want to force using local file even if connection string is set, set FORCE_LOCAL=1
FORCE_LOCAL = os.getenv("FORCE_LOCAL", "0") == "1"

# Candidate numeric columns for prediction values (based on uploaded CSV)
CANDIDATE_NUMERIC_COLS = [
    "PREDICTION", "PREDICTED", "PREDICTED_SALES", "FORECAST",
    "TOTAL_SALES_COUNT", "VALUE", "SALES_COUNT", "PRED_LOG", "LOG_SALES"
]


def _load_local_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local CSV not found at: {path}")
    df = pd.read_csv(path)
    # try parse DATE if present
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


def _load_blob_csv(conn_str: str, container: str, blob_path: str) -> pd.DataFrame:
    if BlobServiceClient is None:
        raise RuntimeError("azure-storage-blob package not installed in environment.")
    svc = BlobServiceClient.from_connection_string(conn_str)
    blob = svc.get_blob_client(container=container, blob=blob_path)
    if not blob.exists():
        raise FileNotFoundError(f"Blob not found: container={container} blob={blob_path}")
    raw = blob.download_blob().readall()
    s = raw.decode("utf-8")
    df = pd.read_csv(io.StringIO(s))
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def _cached_load_csv(use_local_first: bool = True) -> pd.DataFrame:
    """
    Load CSV with caching. Priority:
      - If FORCE_LOCAL or LOCAL_CSV_PATH exists -> local
      - Else if AZ_CONN_STR exists -> blob
      - Else try local path anyway
    """
    # If user explicitly wants local or file exists locally
    if FORCE_LOCAL or os.path.exists(LOCAL_CSV_PATH):
        try:
            return _load_local_csv(LOCAL_CSV_PATH)
        except FileNotFoundError:
            # fall through to blob if available
            pass

    # Try blob if connection string present
    if AZ_CONN_STR and AZ_CONTAINER and AZ_BLOB_PATH:
        try:
            return _load_blob_csv(AZ_CONN_STR, AZ_CONTAINER, AZ_BLOB_PATH)
        except Exception as e:
            # bubble up as runtime error so caller can see message
            raise RuntimeError("Failed to load CSV from blob: " + str(e))

    # Last resort: try local path again
    if os.path.exists(LOCAL_CSV_PATH):
        return _load_local_csv(LOCAL_CSV_PATH)

    raise FileNotFoundError("No CSV found. Set LOCAL_CSV_PATH or provide Azure blob settings.")


def _detect_numeric_column(df: pd.DataFrame) -> Optional[str]:
    """Return first matching candidate numeric column or any numeric column other than DATE."""
    for c in CANDIDATE_NUMERIC_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback - find any numeric column except DATE
    for c in df.columns:
        if c.upper() == "DATE":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _apply_filters(df: pd.DataFrame,
                   start: Optional[str],
                   end: Optional[str],
                   region: Optional[str],
                   salesorg: Optional[str]) -> pd.DataFrame:
    if "DATE" in df.columns:
        if start:
            start_dt = datetime.fromisoformat(start)
            df = df[df["DATE"] >= start_dt]
        if end:
            end_dt = datetime.fromisoformat(end)
            df = df[df["DATE"] <= end_dt]
    if region and "REGION" in df.columns:
        df = df[df["REGION"].astype(str).str.lower() == region.lower()]
    if salesorg and "SALESORG" in df.columns:
        df = df[df["SALESORG"].astype(str).str.lower() == salesorg.lower()]
    return df


@app.get("/predictions")
def get_predictions(
    start: Optional[str] = Query(None, description="start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="end date YYYY-MM-DD"),
    region: Optional[str] = Query(None),
    salesorg: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    columns: Optional[str] = Query(None, description="comma-separated list of columns to return, e.g. DATE,REGION,PREDICTION"),
    refresh: bool = Query(False, description="set true to force reload CSV from source")
):
    """
    Return raw prediction rows with filters, pagination and optional column selection.
    """
    if refresh:
        _cached_load_csv.cache_clear()

    try:
        df = _cached_load_csv()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ensure DATE dtype
    if "DATE" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df_filtered = _apply_filters(df, start, end, region, salesorg)

    # select columns
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip() in df_filtered.columns]
        if not cols:
            raise HTTPException(status_code=400, detail="Requested columns not found in CSV")
        df_filtered = df_filtered[cols]
    else:
        # limit default to a subset of columns for performance
        cols_default = ["DATE", "REGION", "SALESORG", "SILHOUETTE", "TOTAL_SALES_COUNT", "PREDICTION"]
        cols_present = [c for c in cols_default if c in df_filtered.columns]
        # if none of defaults present, return all (but paginated)
        if cols_present:
            df_filtered = df_filtered[cols_present]

    # pagination
    df_page = df_filtered.iloc[offset: offset + limit].copy()

    # convert DATE to ISO strings for JSON
    if "DATE" in df_page.columns:
        df_page["DATE"] = df_page["DATE"].dt.strftime("%Y-%m-%d")

    return {"count": len(df_filtered), "offset": offset, "limit": limit, "rows": df_page.to_dict(orient="records")}


@app.get("/kpis")
def get_kpis(
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
    top_n_countries: int = Query(10, ge=1, le=100),
    numeric_column: Optional[str] = Query(None, description="override detected numeric column"),
    refresh: bool = Query(False, description="true to force reload CSV and clear local cache")
):
    """
    Returns:
      - numeric_column_used: column used for aggregation
      - total_predicted: sum over numeric column
      - by_country_top: top N countries by numeric sum (if COUNTRY present)
      - monthly: monthly aggregated series (YYYY-MM)
      - rows_considered: number of rows after date filters
    """
    if refresh:
        _cached_load_csv.cache_clear()

    try:
        df = _cached_load_csv()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ensure DATE dtype
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # apply date filters
    df_filtered = _apply_filters(df, start, end, None, None)

    # detect numeric column
    numeric_col = numeric_column or _detect_numeric_column(df_filtered)
    if numeric_col is None:
        raise HTTPException(status_code=400, detail="No numeric column found in CSV. Set numeric_column param to override.")

    # compute totals
    total_pred = float(df_filtered[numeric_col].sum())

    # by country
    by_country = []
    if "COUNTRY" in df_filtered.columns:
        country_agg = df_filtered.groupby("COUNTRY", dropna=True)[numeric_col].sum().sort_values(ascending=False).head(top_n_countries)
        by_country = [{"country": k, "value": float(v)} for k, v in country_agg.items()]

    # monthly
    monthly = []
    if "DATE" in df_filtered.columns:
        ts = df_filtered.set_index("DATE").resample("MS")[numeric_col].sum().reset_index()
        monthly = [{"month": row["DATE"].strftime("%Y-%m"), "value": float(row[numeric_col])} for _, row in ts.iterrows()]

    return {
        "numeric_column_used": numeric_col,
        "total_predicted": total_pred,
        "by_country_top": by_country,
        "monthly": monthly,
        "rows_considered": len(df_filtered)
    }


@app.get("/timeseries")
def timeseries(
    numeric_column: Optional[str] = Query(None, description="metric to aggregate (defaults to auto-detected)"),
    country: Optional[str] = Query(None),
    salesorg: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    refresh: bool = Query(False)
):
    """
    Returns monthly time series aggregated by the numeric column.
    Optionally filter by country or salesorg.
    """
    if refresh:
        _cached_load_csv.cache_clear()
    try:
        df = _cached_load_csv()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df_filtered = _apply_filters(df, start, end, None, None)

    if country and "COUNTRY" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["COUNTRY"].astype(str).str.lower() == country.lower()]

    if salesorg and "SALESORG" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["SALESORG"].astype(str).str.lower() == salesorg.lower()]

    col = numeric_column or _detect_numeric_column(df_filtered)
    if col is None:
        raise HTTPException(status_code=400, detail="No numeric column found. Provide numeric_column param.")

    ts = df_filtered.set_index("DATE").resample("MS")[col].sum().reset_index()
    series = [{"month": r["DATE"].strftime("%Y-%m"), "value": float(r[col])} for _, r in ts.iterrows()]
    return {"numeric_column": col, "series": series, "rows_considered": len(df_filtered)}


@app.get("/top-countries")
def top_countries(
    n: int = Query(10, ge=1, le=200),
    numeric_column: Optional[str] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    refresh: bool = Query(False)
):
    """
    Returns top N countries by numeric metric.
    """
    if refresh:
        _cached_load_csv.cache_clear()

    try:
        df = _cached_load_csv()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df_filtered = _apply_filters(df, start, end, None, None)

    col = numeric_column or _detect_numeric_column(df_filtered)
    if col is None:
        raise HTTPException(status_code=400, detail="No numeric column found. Provide numeric_column param.")

    if "COUNTRY" not in df_filtered.columns:
        raise HTTPException(status_code=400, detail="COUNTRY column not present in CSV.")

    agg = df_filtered.groupby("COUNTRY", dropna=True)[col].sum().sort_values(ascending=False).head(n)
    result = [{"country": k, "value": float(v)} for k, v in agg.items()]
    return {"numeric_column": col, "top_countries": result}
