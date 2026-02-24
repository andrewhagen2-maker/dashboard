"""
app.py
──────
Brand Channel Intelligence Dashboard
Main Streamlit entry point.

Phase 1 shell: loads seed data and snapshot cache, confirms data shape,
renders a status page. Charts and tables are built in Phase 3–4.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brand Channel Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ASINS_FILE = ROOT / "data" / "target_asins.csv"
SNAPSHOTS_FILE = ROOT / "data" / "cache" / "snapshots.csv"


# ── Data loaders ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_asins() -> pd.DataFrame:
    """Load the ASIN seed file."""
    df = pd.read_csv(ASINS_FILE)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(ttl=3600)
def load_snapshots() -> pd.DataFrame:
    """Load the snapshot cache. Returns empty DataFrame with correct schema if no data yet."""
    df = pd.read_csv(SNAPSHOTS_FILE, dtype={"asin": str, "seller_id": str})
    df.columns = df.columns.str.strip()
    if "date" in df.columns and len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "inventory_est" in df.columns:
        df["inventory_est"] = pd.to_numeric(df["inventory_est"], errors="coerce")
    return df


# ── Load data ───────────────────────────────────────────────────────────────────
asins_df = load_asins()
snapshots_df = load_snapshots()

has_snapshot_data = len(snapshots_df) > 0

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Brand Channel Intelligence")
    st.caption("Unauthorized seller monitoring · MAP compliance · Grey market detection")
    st.divider()

    # Tag filter
    all_tags = sorted(asins_df["tag"].dropna().unique().tolist())
    tag_options = ["All"] + all_tags
    selected_tag = st.selectbox("Filter by tag", tag_options, index=0)

    # Time range
    time_range = st.radio("Time range", [30, 60, 90], index=2, format_func=lambda x: f"{x} days")

    st.divider()

    # Last updated
    if has_snapshot_data:
        last_date = snapshots_df["date"].max()
        st.caption(f"**Last snapshot:** {last_date.strftime('%b %d, %Y')}")
    else:
        st.caption("**Last snapshot:** No data yet")

    st.caption(f"**ASINs tracked:** {len(asins_df)}")


# ── Main content ────────────────────────────────────────────────────────────────
st.title("Brand Channel Intelligence")
st.caption("Third-party seller monitoring across tracked ASINs · New condition · 3P only")

st.divider()

# ── Status panel (Phase 1 — replace with charts in Phase 3) ────────────────────
if not has_snapshot_data:
    st.info(
        "**No snapshot data yet.** Run `python fetch_snapshot.py` locally to pull the first "
        "day of data, then commit `data/cache/snapshots.csv` to the repo. "
        "The GitHub Action will take over from there.",
        icon="ℹ️",
    )
else:
    # Summary metrics from snapshot data
    filtered_snaps = snapshots_df.copy()
    if selected_tag != "All":
        filtered_snaps = filtered_snaps[filtered_snaps["tag"] == selected_tag]

    cutoff = snapshots_df["date"].max() - pd.Timedelta(days=time_range)
    filtered_snaps = filtered_snaps[filtered_snaps["date"] >= cutoff]

    latest = filtered_snaps[filtered_snaps["date"] == filtered_snaps["date"].max()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("3P Sellers (latest)", latest["seller_id"].nunique())
    col2.metric("3P Offers (latest)", len(latest))
    col3.metric(
        "Total Est. Inventory",
        f"{int(latest['inventory_est'].sum()):,}" if latest["inventory_est"].notna().any() else "—",
    )
    fba_count = len(latest[latest["fulfillment"] == "FBA"])
    fbm_count = len(latest[latest["fulfillment"] == "FBM"])
    col4.metric("FBA / FBM", f"{fba_count} / {fbm_count}")

st.divider()

# ── ASIN Manager ────────────────────────────────────────────────────────────────
with st.expander("📋 Tracked ASINs", expanded=not has_snapshot_data):
    st.caption(
        "Read-only view. To add or remove ASINs, edit `data/target_asins.csv` in the repo and redeploy."
    )
    display_cols = ["asin", "brand", "product_name", "category", "map_price", "tag"]
    st.dataframe(
        asins_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "asin": st.column_config.TextColumn("ASIN"),
            "brand": st.column_config.TextColumn("Brand"),
            "product_name": st.column_config.TextColumn("Product"),
            "category": st.column_config.TextColumn("Category"),
            "map_price": st.column_config.NumberColumn("MAP Price", format="$%.2f"),
            "tag": st.column_config.TextColumn("Tag"),
        },
    )

# ── Placeholder for Phase 3 charts ─────────────────────────────────────────────
if has_snapshot_data:
    st.info("📊 Charts and seller tables coming in Phase 3.", icon="🔧")
