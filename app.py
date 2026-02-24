"""
app.py
──────
Brand Channel Intelligence Dashboard
Main Streamlit entry point.

Reads from data/cache/snapshots.csv only — never calls Keepa directly.
Daily data is written by fetch_snapshot.py via GitHub Action.
"""

import pandas as pd
import plotly.graph_objects as go
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
    df = pd.read_csv(ASINS_FILE)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(ttl=3600)
def load_snapshots() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOTS_FILE, dtype={"asin": str, "seller_id": str})
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    if "date" in df.columns and len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "inventory_est" in df.columns:
        df["inventory_est"] = pd.to_numeric(df["inventory_est"], errors="coerce")
    if "is_map" in df.columns:
        df["is_map"] = df["is_map"].astype(str).str.lower().map({"true": True, "false": False})
    if "is_prime" in df.columns:
        df["is_prime"] = df["is_prime"].astype(str).str.lower().map({"true": True, "false": False})
    return df


# ── Chart helpers ────────────────────────────────────────────────────────────────
CHART_COLORS = {
    "primary": "#2563EB",
    "secondary": "#E8622A",
    "fba": "#2563EB",
    "fbm": "#E8622A",
}

CHART_LAYOUT = dict(
    height=260,
    margin=dict(l=0, r=0, t=36, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


def line_chart(df_daily: pd.DataFrame, y_col: str, title: str, color: str, y_label: str = "") -> go.Figure:
    """Single-line chart from a daily aggregated DataFrame with a 'date' column."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_daily["date"],
        y=df_daily[y_col],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
        name=y_label or y_col,
        hovertemplate="%{y}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), showlegend=False, **CHART_LAYOUT)
    if y_label:
        fig.update_yaxes(title_text=y_label)
    return fig


def fba_fbm_chart(df_daily: pd.DataFrame, title: str) -> go.Figure:
    """Two-line chart: FBA count vs FBM count over time."""
    fig = go.Figure()
    for col, label, color in [
        ("fba_count", "FBA", CHART_COLORS["fba"]),
        ("fbm_count", "FBM", CHART_COLORS["fbm"]),
    ]:
        fig.update_layout(showlegend=True)
        fig.add_trace(go.Scatter(
            x=df_daily["date"],
            y=df_daily[col],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=label,
            hovertemplate=f"{label}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), **CHART_LAYOUT)
    return fig


def build_chart_data(df: pd.DataFrame) -> dict:
    """
    Aggregate snapshot data into per-day series for all four charts.
    Returns a dict of DataFrames ready to plot.
    """
    if df.empty:
        empty = pd.DataFrame(columns=["date"])
        return {
            "seller_count": empty.assign(seller_count=[]),
            "offer_count": empty.assign(offer_count=[]),
            "inventory": empty.assign(inventory_est=[]),
            "fba_fbm": empty.assign(fba_count=[], fbm_count=[]),
        }

    by_date = df.groupby("date")

    seller_count = (
        by_date["seller_id"].nunique().reset_index().rename(columns={"seller_id": "seller_count"})
    )

    offer_count = (
        by_date.size().reset_index(name="offer_count")
    )

    inventory = (
        by_date["inventory_est"].sum().reset_index()
    )

    fba_fbm = (
        df.groupby(["date", "fulfillment"]).size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure both columns exist even if one fulfillment type is absent
    for col in ["FBA", "FBM"]:
        if col not in fba_fbm.columns:
            fba_fbm[col] = 0
    fba_fbm = fba_fbm.rename(columns={"FBA": "fba_count", "FBM": "fbm_count"})

    return {
        "seller_count": seller_count,
        "offer_count": offer_count,
        "inventory": inventory,
        "fba_fbm": fba_fbm,
    }


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


# ── Apply sidebar filters ────────────────────────────────────────────────────────
filtered_snaps = snapshots_df.copy()

if has_snapshot_data:
    # Tag filter
    if selected_tag != "All":
        filtered_snaps = filtered_snaps[filtered_snaps["tag"] == selected_tag]

    # Time range filter
    cutoff = snapshots_df["date"].max() - pd.Timedelta(days=time_range)
    filtered_snaps = filtered_snaps[filtered_snaps["date"] >= cutoff]

    # Latest day slice (for metric cards)
    latest = filtered_snaps[filtered_snaps["date"] == filtered_snaps["date"].max()]
else:
    latest = pd.DataFrame()


# ── Main content ────────────────────────────────────────────────────────────────
st.title("Brand Channel Intelligence")
st.caption("Third-party seller monitoring across tracked ASINs · New condition · 3P only")
st.divider()

# ── No data state ───────────────────────────────────────────────────────────────
if not has_snapshot_data:
    st.info(
        "**No snapshot data yet.** Run `python fetch_snapshot.py --limit 3` locally to pull the "
        "first day of data, then commit `data/cache/snapshots.csv` to the repo. "
        "The GitHub Action will take over from there.",
        icon="ℹ️",
    )

# ── Metric cards ────────────────────────────────────────────────────────────────
else:
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

    # ── Four line charts ─────────────────────────────────────────────────────────
    chart_data = build_chart_data(filtered_snaps)

    # Only one day of data — warn but still render (charts will show a single point)
    if filtered_snaps["date"].nunique() == 1:
        st.info(
            "Only one day of snapshot data is available so far. "
            "Charts will fill in as daily snapshots accumulate.",
            icon="📅",
        )

    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.plotly_chart(
            line_chart(chart_data["seller_count"], "seller_count",
                       "3P Seller Count", CHART_COLORS["primary"], "Sellers"),
            use_container_width=True,
        )
    with row1_right:
        st.plotly_chart(
            line_chart(chart_data["offer_count"], "offer_count",
                       "3P Offer Count", CHART_COLORS["primary"], "Offers"),
            use_container_width=True,
        )

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.plotly_chart(
            line_chart(chart_data["inventory"], "inventory_est",
                       "Total 3P Inventory (Est.)", CHART_COLORS["secondary"], "Units"),
            use_container_width=True,
        )
    with row2_right:
        st.plotly_chart(
            fba_fbm_chart(chart_data["fba_fbm"], "FBA vs FBM Offers"),
            use_container_width=True,
        )

    st.divider()

    # ── Placeholder for Phase 4 toggle table ────────────────────────────────────
    st.info("📊 Seller / ASIN toggle table coming in Phase 4.", icon="🔧")


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
