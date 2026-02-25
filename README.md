# Brand Channel Intelligence Dashboard

A live dashboard for monitoring third-party seller activity across Amazon marketplace listings — tracking unauthorized sellers, MAP compliance, and grey market signals using real offer data.

Built as a portfolio project demonstrating brand protection and distribution hygiene workflows.

**Live:** [dashboard.andrewhagen.work](https://dashboard.andrewhagen.work)

---

## What it does

- Tracks 3P seller activity across a monitored ASIN list (currently Yeti, Crocs, and LEGO products)
- Flags sellers pricing below MAP and computes a per-seller disruption score weighted by inventory depth
- Shows FBA vs FBM fulfillment breakdown and estimated stock levels
- Provides a toggle table — drill down by seller or by ASIN — with row-click filtering across all charts
- Updates automatically every day via GitHub Actions

---

## Architecture

```
data/target_asins.csv        ← ASIN seed file (edit here to add/remove products)
        ↓
fetch_snapshot.py            ← Daily runner (GitHub Action, 6am UTC)
        ↓
data/cache/snapshots.csv     ← Append-only: one row per seller-ASIN pair per day
data/seller_names.csv        ← Seller ID → display name lookup, built incrementally
        ↓
app.py                       ← Streamlit app — reads CSVs only, never calls Keepa
```

The live app has no API dependency. All Keepa calls happen in `fetch_snapshot.py` during the daily GitHub Action run. The Streamlit app is purely a read layer on top of the snapshot cache.

---

## Data model

**`data/target_asins.csv`** — ASIN seed file

| Column | Description |
|---|---|
| `asin` | Amazon ASIN |
| `brand` | Brand name (used for sidebar filtering) |
| `product_name` | Display name |
| `category` | Product category |
| `map_price` | Minimum Advertised Price in USD |
| `tag` | Filter tag (e.g. `Tumbler`, `Clogs`) |

**`data/cache/snapshots.csv`** — Append-only daily snapshot

| Column | Description |
|---|---|
| `date` | Snapshot date |
| `asin` | Amazon ASIN |
| `brand`, `product_name`, `tag` | From seed file |
| `seller_id` | Amazon seller ID |
| `seller_name` | Display name (joined from seller_names.csv) |
| `condition` | Always `New` |
| `fulfillment` | `FBA` or `FBM` |
| `price` | Current offer price in USD |
| `inventory_est` | Estimated stock from Keepa stockCSV |
| `ships_from_country` | Country flag from Keepa |
| `is_map` | Keepa MAP flag |
| `is_prime` | Prime eligible |

Amazon 1P rows are excluded. Used condition rows are excluded.

---

## Running locally

**Requirements:** Python 3.11+, a Keepa API key.

```bash
# Clone and install
git clone <repo-url>
cd brand-channel-dashboard
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your KEEPA_API_KEY

# Pull a snapshot manually (all ASINs)
python fetch_snapshot.py

# Pull a snapshot for the first N ASINs only (useful for dev/testing)
python fetch_snapshot.py --limit 5

# Run the dashboard
streamlit run app.py
```

---

## GitHub Action

The workflow at `.github/workflows/daily_snapshot.yml` runs `fetch_snapshot.py` at 6am UTC daily, commits the updated `snapshots.csv` and `seller_names.csv` back to the repo, and triggers a Streamlit Cloud redeploy.

The `KEEPA_API_KEY` is stored as a GitHub Actions secret — it is never committed to the repo. Token budget is managed automatically: `fetch_snapshot.py` pauses and waits for refill if the token bank drops below the minimum threshold before moving to the next ASIN.

---

## Stack

- **Data source:** [Keepa API](https://keepa.com/) via the `keepa` Python library
- **Automation:** GitHub Actions
- **Dashboard:** Streamlit + Plotly
- **Hosting:** Streamlit Cloud (data pipeline via GitHub)
