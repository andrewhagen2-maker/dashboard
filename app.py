"""
app.py
──────
Brand Channel Intelligence Dashboard
Main Streamlit entry point.

Reads from data/cache/snapshots.csv only — never calls Keepa directly.
Daily data is written by fetch_snapshot.py via GitHub Action.
"""

import numpy as np
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
SELLER_NAMES_FILE = ROOT / "data" / "seller_names.csv"


# ── Data loaders ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_asins() -> pd.DataFrame:
    df = pd.read_csv(ASINS_FILE)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(ttl=3600)
def load_seller_names() -> dict[str, str]:
    """Load seller_names.csv and return a {seller_id: seller_name} dict."""
    if not SELLER_NAMES_FILE.exists():
        return {}
    df = pd.read_csv(SELLER_NAMES_FILE, dtype=str)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["seller_id"])
    # Only include rows where seller_name is non-empty
    df = df[df["seller_name"].fillna("").str.strip() != ""]
    return dict(zip(df["seller_id"], df["seller_name"]))


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
    if "is_1p" in df.columns:
        df["is_1p"] = df["is_1p"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    else:
        # Backwards-compatible: old snapshots without the column are assumed 3P
        df["is_1p"] = False
    return df


# ── Enrichment ──────────────────────────────────────────────────────────────────
def enrich_snapshots(snaps: pd.DataFrame, asins: pd.DataFrame) -> pd.DataFrame:
    """
    Join map_price from target_asins.csv into snapshots, then derive:

      map_variance_pct  — signed % difference from MAP
                          negative = below MAP (discounting)
                          positive = above MAP (overpricing)

      disruption_score  — inventory-weighted discount severity
                          formula: inventory_est × (|discount_pct| / 100) ^ 1.5 × 10,000
                          only computed for below-MAP offers; 0 otherwise
                          ^1.5 exponent: non-linear — deep discounts score
                          disproportionately higher than shallow ones.
    """
    if snaps.empty:
        snaps["map_price"] = pd.NA
        snaps["map_variance_pct"] = pd.NA
        snaps["disruption_score"] = pd.NA
        return snaps

    # Join map_price — left join so all snapshot rows are preserved
    map_lookup = asins[["asin", "map_price"]].drop_duplicates("asin")
    df = snaps.merge(map_lookup, on="asin", how="left")

    # Signed % variance from MAP
    df["map_variance_pct"] = np.where(
        df["map_price"].notna() & (df["map_price"] > 0) & df["price"].notna(),
        ((df["price"] - df["map_price"]) / df["map_price"] * 100).round(2),
        np.nan,
    )

    # Disruption score — only for below-MAP offers with known inventory
    below_map = df["map_variance_pct"].notna() & (df["map_variance_pct"] < 0)
    has_inventory = df["inventory_est"].notna() & (df["inventory_est"] > 0)

    discount_fraction = (df["map_variance_pct"].abs() / 100)
    df["disruption_score"] = np.where(
        below_map & has_inventory,
        (df["inventory_est"] * discount_fraction ** 1.5 * 10_000).round(0),
        0,
    )

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


def dual_line_chart(
    df_daily: pd.DataFrame,
    total_col: str,
    disruptive_col: str,
    title: str,
    total_label: str,
    disruptive_label: str,
) -> go.Figure:
    """Two-line chart: total (muted) vs disruptive (orange) over time."""
    fig = go.Figure()
    # Total — muted blue, dashed
    fig.add_trace(go.Scatter(
        x=df_daily["date"],
        y=df_daily[total_col],
        mode="lines+markers",
        line=dict(color=CHART_COLORS["primary"], width=2, dash="dot"),
        marker=dict(size=4),
        name=total_label,
        hovertemplate=f"{total_label}: %{{y}}<extra></extra>",
    ))
    # Disruptive — orange, solid, slightly thicker
    fig.add_trace(go.Scatter(
        x=df_daily["date"],
        y=df_daily[disruptive_col],
        mode="lines+markers",
        line=dict(color=CHART_COLORS["secondary"], width=2.5),
        marker=dict(size=5),
        name=disruptive_label,
        hovertemplate=f"{disruptive_label}: %{{y}}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), showlegend=True, **CHART_LAYOUT)
    return fig


def build_chart_data(df: pd.DataFrame, total_asin_count: int) -> dict:
    """
    Aggregate snapshot rows into per-day series for all six line charts.

    'Disruptive' = any offer where map_variance_pct < 0 (priced below MAP).

    Charts produced:
      seller_count  — total sellers vs disruptive sellers per day
      offer_count   — total offers vs disruptive offers per day
      inventory     — total 3P inventory per day
      fba_fbm       — FBA vs FBM offer counts per day
      pct_clean     — % of tracked ASINs with zero below-MAP offers per day
      disruption_ts — sum of disruption_score per day
    """
    if df.empty:
        empty = pd.DataFrame(columns=["date"])
        return {
            "seller_count": empty.assign(seller_count=[], disruptive_seller_count=[]),
            "offer_count": empty.assign(offer_count=[], disruptive_offer_count=[]),
            "inventory": empty.assign(inventory_est=[], disruptive_inventory=[]),
            "fba_fbm": empty.assign(fba_count=[], fbm_count=[]),
            "pct_clean": empty.assign(pct_clean=[]),
            "disruption_ts": empty.assign(disruption_score=[]),
        }

    by_date = df.groupby("date")
    is_disruptive = df["map_variance_pct"].notna() & (df["map_variance_pct"] < 0)

    # ── Seller count: total vs disruptive ──
    total_sellers = (
        by_date["seller_id"].nunique().reset_index()
        .rename(columns={"seller_id": "seller_count"})
    )
    disruptive_sellers = (
        df[is_disruptive].groupby("date")["seller_id"].nunique().reset_index()
        .rename(columns={"seller_id": "disruptive_seller_count"})
    )
    seller_count = total_sellers.merge(disruptive_sellers, on="date", how="left").fillna(0)
    seller_count["disruptive_seller_count"] = seller_count["disruptive_seller_count"].astype(int)

    # ── Offer count: total vs disruptive ──
    total_offers = by_date.size().reset_index(name="offer_count")
    disruptive_offers = (
        df[is_disruptive].groupby("date").size().reset_index(name="disruptive_offer_count")
    )
    offer_count = total_offers.merge(disruptive_offers, on="date", how="left").fillna(0)
    offer_count["disruptive_offer_count"] = offer_count["disruptive_offer_count"].astype(int)

    # ── Inventory: total vs disruptive ──
    total_inventory = by_date["inventory_est"].sum().reset_index()
    disruptive_inventory = (
        df[is_disruptive].groupby("date")["inventory_est"].sum().reset_index()
        .rename(columns={"inventory_est": "disruptive_inventory"})
    )
    inventory = total_inventory.merge(disruptive_inventory, on="date", how="left").fillna(0)

    # ── FBA vs FBM ──
    fba_fbm = (
        df.groupby(["date", "fulfillment"]).size()
        .unstack(fill_value=0).reset_index()
    )
    for col in ["FBA", "FBM"]:
        if col not in fba_fbm.columns:
            fba_fbm[col] = 0
    fba_fbm = fba_fbm.rename(columns={"FBA": "fba_count", "FBM": "fbm_count"})

    # ── % Assortment Clean ──
    # Denominator: total_asin_count (all tracked ASINs, fixed)
    # Numerator: ASINs that had ZERO below-MAP offers on that day
    disrupted_asins_per_day = (
        df[is_disruptive].groupby("date")["asin"].nunique().reset_index()
        .rename(columns={"asin": "disrupted_asin_count"})
    )
    all_dates = pd.DataFrame({"date": df["date"].unique()})
    pct_clean = all_dates.merge(disrupted_asins_per_day, on="date", how="left").fillna(0)
    pct_clean["pct_clean"] = (
        (total_asin_count - pct_clean["disrupted_asin_count"]) / total_asin_count * 100
    ).clip(0, 100).round(1)

    # ── Disruption score over time ──
    disruption_ts = (
        by_date["disruption_score"].sum().reset_index()
    )

    return {
        "seller_count": seller_count,
        "offer_count": offer_count,
        "inventory": inventory,
        "fba_fbm": fba_fbm,
        "pct_clean": pct_clean,
        "disruption_ts": disruption_ts,
    }


def build_seller_table(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    """Aggregate latest-day snapshot into a by-seller summary table."""
    if df.empty:
        return pd.DataFrame(columns=["Seller Name", "Seller ID", "# Offers", "# Unique ASINs", "Total Est. Stock", "Avg Discount %", "Disruption Score"])
    latest = df[df["date"] == df["date"].max()]
    agg = (
        latest.groupby("seller_id")
        .agg(
            offers=("asin", "count"),
            unique_asins=("asin", "nunique"),
            total_stock=("inventory_est", "sum"),
            avg_discount=("map_variance_pct", lambda x: x[x < 0].mean()),
            disruption=("disruption_score", "sum"),
        )
        .reset_index()
        .sort_values("disruption", ascending=False)
        .rename(columns={
            "seller_id": "Seller ID",
            "offers": "# Offers",
            "unique_asins": "# Unique ASINs",
            "total_stock": "Total Est. Stock",
            "avg_discount": "Avg Discount %",
            "disruption": "Disruption Score",
        })
    )
    agg["Total Est. Stock"] = agg["Total Est. Stock"].fillna(0).astype(int)
    agg["Disruption Score"] = agg["Disruption Score"].fillna(0).astype(int)
    agg["Avg Discount %"] = agg["Avg Discount %"].round(1)
    # Join seller name — fall back to seller ID if not yet resolved
    agg["Seller Name"] = agg["Seller ID"].map(name_map).fillna("—")
    # Reorder: name first, then ID for reference
    cols = ["Seller Name", "Seller ID", "# Offers", "# Unique ASINs", "Total Est. Stock", "Avg Discount %", "Disruption Score"]
    return agg[cols].reset_index(drop=True)


def build_asin_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate latest-day snapshot into a by-ASIN summary table."""
    if df.empty:
        return pd.DataFrame(columns=["ASIN", "Product", "# 3P Sellers", "# 3P Offers", "Total 3P Inventory", "Avg Discount %", "Disruption Score"])
    latest = df[df["date"] == df["date"].max()]
    agg = (
        latest.groupby(["asin", "product_name"])
        .agg(
            sellers=("seller_id", "nunique"),
            offers=("seller_id", "count"),
            total_inventory=("inventory_est", "sum"),
            avg_discount=("map_variance_pct", lambda x: x[x < 0].mean()),
            disruption=("disruption_score", "sum"),
        )
        .reset_index()
        .sort_values("disruption", ascending=False)
        .rename(columns={
            "asin": "ASIN",
            "product_name": "Product",
            "sellers": "# 3P Sellers",
            "offers": "# 3P Offers",
            "total_inventory": "Total 3P Inventory",
            "avg_discount": "Avg Discount %",
            "disruption": "Disruption Score",
        })
    )
    agg["Total 3P Inventory"] = agg["Total 3P Inventory"].fillna(0).astype(int)
    agg["Disruption Score"] = agg["Disruption Score"].fillna(0).astype(int)
    agg["Avg Discount %"] = agg["Avg Discount %"].round(1)
    return agg.reset_index(drop=True)


# ── Load data ───────────────────────────────────────────────────────────────────
asins_df = load_asins()
snapshots_df = enrich_snapshots(load_snapshots(), asins_df)
seller_names = load_seller_names()
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

    # Charts and tables always show 3P only; 1P rows are stored in the snapshot
    # for reference but excluded from all display logic here.
    filtered_snaps = filtered_snaps[filtered_snaps["is_1p"] == False]

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
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("3P Sellers (latest)", latest["seller_id"].nunique())
    col2.metric("3P Offers (latest)", len(latest))
    col3.metric(
        "Total Est. Inventory",
        f"{int(latest['inventory_est'].sum()):,}" if latest["inventory_est"].notna().any() else "—",
    )
    fba_c = len(latest[latest["fulfillment"] == "FBA"])
    fbm_c = len(latest[latest["fulfillment"] == "FBM"])
    col4.metric("FBA / FBM", f"{fba_c} / {fbm_c}")
    total_disruption = latest["disruption_score"].sum() if "disruption_score" in latest.columns else 0
    col5.metric("Disruption Score", f"{int(total_disruption):,}")

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

    # ── Six line charts ───────────────────────────────────────────────────────────
    # total_asin_count: denominator for % clean (all tracked ASINs in the filtered set)
    total_asin_count = (
        filtered_snaps["asin"].nunique() if not filtered_snaps.empty
        else len(asins_df)
    )
    chart_data = build_chart_data(chart_snaps, total_asin_count)

    if filtered_snaps["date"].nunique() == 1:
        st.info(
            "Only one day of snapshot data is available so far. "
            "Charts will fill in as daily snapshots accumulate.",
            icon="📅",
        )

    # Row 1: seller count (total vs disruptive) | offer count (total vs disruptive)
    r1a, r1b = st.columns(2)
    with r1a:
        st.plotly_chart(
            dual_line_chart(
                chart_data["seller_count"],
                "seller_count", "disruptive_seller_count",
                "3P Seller Count", "Total Sellers", "Disruptive Sellers",
            ),
            use_container_width=True,
        )
    with r1b:
        st.plotly_chart(
            dual_line_chart(
                chart_data["offer_count"],
                "offer_count", "disruptive_offer_count",
                "3P Offer Count", "Total Offers", "Disruptive Offers",
            ),
            use_container_width=True,
        )

    # Row 2: total inventory | FBA vs FBM
    r2a, r2b = st.columns(2)
    with r2a:
        st.plotly_chart(
            dual_line_chart(
                chart_data["inventory"],
                "inventory_est", "disruptive_inventory",
                "3P Inventory (Est.)", "Total Inventory", "Disruptive Inventory",
            ),
            use_container_width=True,
        )
    with r2b:
        st.plotly_chart(
            fba_fbm_chart(chart_data["fba_fbm"], "FBA vs FBM Offers"),
            use_container_width=True,
        )

    # Row 3: % assortment clean | disruption score over time
    r3a, r3b = st.columns(2)
    with r3a:
        clean_fig = line_chart(chart_data["pct_clean"], "pct_clean",
                               "% of Assortment Clean", "#22A55A", "%")
        clean_fig.update_yaxes(range=[0, 100])
        st.plotly_chart(clean_fig, use_container_width=True)
    with r3b:
        st.plotly_chart(
            line_chart(chart_data["disruption_ts"], "disruption_score",
                       "Disruption Score Over Time", CHART_COLORS["secondary"], "Score"),
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
        seller_table = build_seller_table(filtered_snaps, seller_names)
        st.caption("Click a row to filter all charts to that seller.")
        sel = st.dataframe(
            seller_table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Seller Name": st.column_config.TextColumn("Seller Name"),
                "Seller ID": st.column_config.TextColumn("Seller ID"),
                "# Offers": st.column_config.NumberColumn("# Offers"),
                "# Unique ASINs": st.column_config.NumberColumn("# Unique ASINs"),
                "Total Est. Stock": st.column_config.NumberColumn("Total Est. Stock", format="%d"),
                "Avg Discount %": st.column_config.NumberColumn("Avg Discount %", format="%.1f%%"),
                "Disruption Score": st.column_config.NumberColumn("Disruption Score", format="%d"),
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
                "Avg Discount %": st.column_config.NumberColumn("Avg Discount %", format="%.1f%%"),
                "Disruption Score": st.column_config.NumberColumn("Disruption Score", format="%d"),
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
