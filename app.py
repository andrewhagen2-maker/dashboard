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
    xaxis=dict(showgrid=False, zeroline=False, type="date", tickformat="%b %d"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


def line_chart(df_daily: pd.DataFrame, y_col: str, title: str, color: str, y_label: str = "") -> go.Figure:
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
    fig = go.Figure()
    for col, label, color in [
        ("fba_count", "FBA", CHART_COLORS["fba"]),
        ("fbm_count", "FBM", CHART_COLORS["fbm"]),
    ]:
        fig.add_trace(go.Scatter(
            x=df_daily["date"],
            y=df_daily[col],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=label,
            hovertemplate=f"{label}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), showlegend=True, **CHART_LAYOUT)
    return fig


def build_chart_data(df: pd.DataFrame) -> dict:
    """Aggregate snapshot rows into per-day series for the four line charts."""
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
        by_date["seller_id"].nunique().reset_index()
        .rename(columns={"seller_id": "seller_count"})
    )
    offer_count = by_date.size().reset_index(name="offer_count")
    inventory = by_date["inventory_est"].sum().reset_index()

    fba_fbm = (
        df.groupby(["date", "fulfillment"]).size()
        .unstack(fill_value=0).reset_index()
    )
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


def build_seller_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate latest-day snapshot into a by-seller summary table."""
    if df.empty:
        return pd.DataFrame(columns=["Seller ID", "# Offers", "# Unique ASINs", "Total Est. Stock"])
    latest = df[df["date"] == df["date"].max()]
    agg = (
        latest.groupby("seller_id")
        .agg(
            offers=("asin", "count"),
            unique_asins=("asin", "nunique"),
            total_stock=("inventory_est", "sum"),
        )
        .reset_index()
        .sort_values("offers", ascending=False)
        .rename(columns={
            "seller_id": "Seller ID",
            "offers": "# Offers",
            "unique_asins": "# Unique ASINs",
            "total_stock": "Total Est. Stock",
        })
    )
    agg["Total Est. Stock"] = agg["Total Est. Stock"].fillna(0).astype(int)
    return agg.reset_index(drop=True)


def build_asin_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate latest-day snapshot into a by-ASIN summary table."""
    if df.empty:
        return pd.DataFrame(columns=["ASIN", "Product", "# 3P Sellers", "# 3P Offers", "Total 3P Inventory"])
    latest = df[df["date"] == df["date"].max()]
    agg = (
        latest.groupby(["asin", "product_name"])
        .agg(
            sellers=("seller_id", "nunique"),
            offers=("seller_id", "count"),
            total_inventory=("inventory_est", "sum"),
        )
        .reset_index()
        .sort_values("offers", ascending=False)
        .rename(columns={
            "asin": "ASIN",
            "product_name": "Product",
            "sellers": "# 3P Sellers",
            "offers": "# 3P Offers",
            "total_inventory": "Total 3P Inventory",
        })
    )
    agg["Total 3P Inventory"] = agg["Total 3P Inventory"].fillna(0).astype(int)
    return agg.reset_index(drop=True)


# ── Load data ───────────────────────────────────────────────────────────────────
asins_df = load_asins()
snapshots_df = load_snapshots()
has_snapshot_data = len(snapshots_df) > 0

# ── Session state — drill-down filter ───────────────────────────────────────────
if "drill_type" not in st.session_state:
    st.session_state.drill_type = None   # "seller" | "asin" | None
if "drill_value" not in st.session_state:
    st.session_state.drill_value = None


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Brand Channel Intelligence")
    st.caption("Unauthorized seller monitoring · MAP compliance · Grey market detection")
    st.divider()

    all_brands = sorted(asins_df["brand"].dropna().unique().tolist())
    brand_options = ["All"] + all_brands
    selected_brand = st.selectbox("Filter by brand", brand_options, index=0)

    all_tags = sorted(asins_df["tag"].dropna().unique().tolist())
    tag_options = ["All"] + all_tags
    selected_tag = st.selectbox("Filter by tag", tag_options, index=0)

    time_range = st.radio("Time range", [30, 60, 90], index=2, format_func=lambda x: f"{x} days")

    st.divider()

    if has_snapshot_data:
        last_date = snapshots_df["date"].max()
        st.caption(f"**Last snapshot:** {last_date.strftime('%b %d, %Y')}")
    else:
        st.caption("**Last snapshot:** No data yet")

    st.caption(f"**ASINs tracked:** {len(asins_df)}")


# ── Apply sidebar filters ────────────────────────────────────────────────────────
filtered_snaps = snapshots_df.copy()

if has_snapshot_data:
    if selected_brand != "All":
        filtered_snaps = filtered_snaps[filtered_snaps["brand"] == selected_brand]

    if selected_tag != "All":
        filtered_snaps = filtered_snaps[filtered_snaps["tag"] == selected_tag]

    cutoff = snapshots_df["date"].max() - pd.Timedelta(days=time_range)
    filtered_snaps = filtered_snaps[filtered_snaps["date"] >= cutoff]

    latest = filtered_snaps[filtered_snaps["date"] == filtered_snaps["date"].max()]
else:
    latest = pd.DataFrame()

# ── Apply drill-down filter to chart data ────────────────────────────────────────
chart_snaps = filtered_snaps.copy()
if st.session_state.drill_type == "seller" and st.session_state.drill_value:
    chart_snaps = chart_snaps[chart_snaps["seller_id"] == st.session_state.drill_value]
elif st.session_state.drill_type == "asin" and st.session_state.drill_value:
    chart_snaps = chart_snaps[chart_snaps["asin"] == st.session_state.drill_value]


# ── Main content ────────────────────────────────────────────────────────────────
st.title("Brand Channel Intelligence")
st.caption("Third-party seller monitoring across tracked ASINs · New condition · 3P only")
st.divider()

if not has_snapshot_data:
    st.info(
        "**No snapshot data yet.** Run `python fetch_snapshot.py --limit 3` locally to pull the "
        "first day of data, then commit `data/cache/snapshots.csv` to the repo. "
        "The GitHub Action will take over from there.",
        icon="ℹ️",
    )
else:
    # ── Metric cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("3P Sellers (latest)", latest["seller_id"].nunique())
    col2.metric("3P Offers (latest)", len(latest))
    col3.metric(
        "Total Est. Inventory",
        f"{int(latest['inventory_est'].sum()):,}" if latest["inventory_est"].notna().any() else "—",
    )
    fba_c = len(latest[latest["fulfillment"] == "FBA"])
    fbm_c = len(latest[latest["fulfillment"] == "FBM"])
    col4.metric("FBA / FBM", f"{fba_c} / {fbm_c}")

    st.divider()

    # ── Active drill-down banner ──────────────────────────────────────────────────
    if st.session_state.drill_type and st.session_state.drill_value:
        label = (
            f"Seller: {st.session_state.drill_value}"
            if st.session_state.drill_type == "seller"
            else f"ASIN: {st.session_state.drill_value}"
        )
        banner_col, clear_col = st.columns([8, 1])
        banner_col.info(f"**Viewing:** {label}", icon="🔍")
        if clear_col.button("Clear", use_container_width=True):
            st.session_state.drill_type = None
            st.session_state.drill_value = None
            st.rerun()

    # ── Four line charts ──────────────────────────────────────────────────────────
    chart_data = build_chart_data(chart_snaps)

    if filtered_snaps["date"].nunique() == 1:
        st.info(
            "Only one day of snapshot data is available so far. "
            "Charts will fill in as daily snapshots accumulate.",
            icon="📅",
        )

    r1a, r1b = st.columns(2)
    with r1a:
        st.plotly_chart(
            line_chart(chart_data["seller_count"], "seller_count",
                       "3P Seller Count", CHART_COLORS["primary"], "Sellers"),
            use_container_width=True,
        )
    with r1b:
        st.plotly_chart(
            line_chart(chart_data["offer_count"], "offer_count",
                       "3P Offer Count", CHART_COLORS["primary"], "Offers"),
            use_container_width=True,
        )

    r2a, r2b = st.columns(2)
    with r2a:
        st.plotly_chart(
            line_chart(chart_data["inventory"], "inventory_est",
                       "Total 3P Inventory (Est.)", CHART_COLORS["secondary"], "Units"),
            use_container_width=True,
        )
    with r2b:
        st.plotly_chart(
            fba_fbm_chart(chart_data["fba_fbm"], "FBA vs FBM Offers"),
            use_container_width=True,
        )

    st.divider()

    # ── Toggle table: By Seller / By ASIN ────────────────────────────────────────
    toggle_col, _ = st.columns([3, 5])
    with toggle_col:
        table_view = st.radio(
            "View by",
            ["Seller", "ASIN"],
            horizontal=True,
            label_visibility="collapsed",
        )

    if table_view == "Seller":
        seller_table = build_seller_table(filtered_snaps)
        st.caption("Click a row to filter all charts to that seller.")
        sel = st.dataframe(
            seller_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Seller ID": st.column_config.TextColumn("Seller ID"),
                "# Offers": st.column_config.NumberColumn("# Offers"),
                "# Unique ASINs": st.column_config.NumberColumn("# Unique ASINs"),
                "Total Est. Stock": st.column_config.NumberColumn("Total Est. Stock", format="%d"),
            },
        )
        # Handle row selection
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            chosen_seller = seller_table.iloc[rows[0]]["Seller ID"]
            if (st.session_state.drill_type != "seller"
                    or st.session_state.drill_value != chosen_seller):
                st.session_state.drill_type = "seller"
                st.session_state.drill_value = chosen_seller
                st.rerun()

    else:  # ASIN view
        asin_table = build_asin_table(filtered_snaps)
        st.caption("Click a row to filter all charts to that ASIN.")
        sel = st.dataframe(
            asin_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "ASIN": st.column_config.TextColumn("ASIN"),
                "Product": st.column_config.TextColumn("Product"),
                "# 3P Sellers": st.column_config.NumberColumn("# 3P Sellers"),
                "# 3P Offers": st.column_config.NumberColumn("# 3P Offers"),
                "Total 3P Inventory": st.column_config.NumberColumn("Total 3P Inventory", format="%d"),
            },
        )
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            chosen_asin = asin_table.iloc[rows[0]]["ASIN"]
            if (st.session_state.drill_type != "asin"
                    or st.session_state.drill_value != chosen_asin):
                st.session_state.drill_type = "asin"
                st.session_state.drill_value = chosen_asin
                st.rerun()


# ── ASIN Manager ────────────────────────────────────────────────────────────────
st.divider()
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
