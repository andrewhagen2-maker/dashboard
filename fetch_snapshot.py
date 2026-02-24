"""
fetch_snapshot.py
─────────────────
Daily snapshot runner. Pulls current 3P offers for every ASIN in
data/target_asins.csv and appends one row per seller-ASIN pair to
data/cache/snapshots.csv.

Run manually:
    python fetch_snapshot.py

Run automatically:
    GitHub Action (.github/workflows/daily_snapshot.yml) calls this
    script once per day at 6am UTC, then commits the updated
    snapshots.csv back to the repo.

Never import this file from app.py — the live Streamlit app only
reads snapshots.csv, it never calls Keepa directly.
"""

import csv
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Allow running from any working directory
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.keepa_client import get_product_with_offers, extract_live_offers, token_status

ASINS_FILE = ROOT / "data" / "target_asins.csv"
SNAPSHOTS_FILE = ROOT / "data" / "cache" / "snapshots.csv"

SNAPSHOT_HEADERS = [
    "date",
    "asin",
    "brand",
    "product_name",
    "tag",
    "seller_id",
    "seller_name",
    "condition",
    "fulfillment",
    "price",
    "inventory_est",
    "ships_from_country",
    "is_map",
    "is_prime",
]

# Keepa seller ID for Amazon retail (1P) — excluded from snapshots
AMAZON_SELLER_ID = "ATVPDKIKX0DER"

# Token management
# With stock=True + buybox=True, each ASIN costs ~10-15 tokens.
# We pause when the bank gets low and wait for refill (1 token/min).
TOKEN_MINIMUM = 15            # pause if bank drops to this level
TOKEN_WAIT_SECONDS = 65       # wait this long before rechecking (slightly over 1 min)


def run():
    today = date.today().isoformat()
    print(f"\n{'='*60}")
    print(f"fetch_snapshot.py — {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    # Load ASIN seed file
    if not ASINS_FILE.exists():
        print(f"ERROR: ASIN seed file not found at {ASINS_FILE}")
        sys.exit(1)

    asins_df = pd.read_csv(ASINS_FILE)
    asins_df.columns = asins_df.columns.str.strip()
    total_asins = len(asins_df)
    print(f"Loaded {total_asins} ASINs from {ASINS_FILE.name}")

    # Check token status before starting
    try:
        tokens = token_status()
        print(f"Keepa tokens available: {tokens['tokens_left']}")
    except Exception as e:
        print(f"WARNING: Could not retrieve token status — {e} (non-fatal, continuing)")

    # Ensure snapshots file exists with headers
    SNAPSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOTS_FILE.exists() or SNAPSHOTS_FILE.stat().st_size == 0:
        with open(SNAPSHOTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_HEADERS)
            writer.writeheader()
        print(f"Created new snapshots file: {SNAPSHOTS_FILE}")

    # Check if today's snapshot already exists (avoid double-runs)
    existing_df = pd.read_csv(SNAPSHOTS_FILE, dtype=str)
    existing_df = existing_df.dropna(how="all")  # ignore blank rows
    if "date" in existing_df.columns and today in existing_df["date"].values:
        print(f"WARNING: Snapshot for {today} already exists. Skipping to avoid duplicates.")
        print("Delete today's rows from snapshots.csv manually if you need to re-run.")
        sys.exit(0)

    # Pull offers for each ASIN and write rows
    new_rows = []
    errors = []

    for idx, asin_row in asins_df.iterrows():
        asin = str(asin_row["asin"]).strip()
        product_name = asin_row["product_name"]
        print(f"\n[{idx + 1}/{total_asins}] {asin} — {product_name}")

        # Token check — wait if bank is too low before making the call
        while True:
            try:
                tokens = token_status()
                available = tokens["tokens_left"]
            except Exception:
                available = TOKEN_MINIMUM + 1  # assume OK if we can't check

            if available >= TOKEN_MINIMUM:
                break

            wait_until = datetime.now().strftime("%H:%M:%S")
            print(f"  ⏳ Token bank low ({available} left). Waiting {TOKEN_WAIT_SECONDS}s for refill... [{wait_until}]")
            time.sleep(TOKEN_WAIT_SECONDS)

        product = get_product_with_offers(asin)

        if not product:
            print(f"  → No product returned (API error or invalid ASIN)")
            errors.append(asin)
            continue

        offers = extract_live_offers(product)

        if not offers:
            print(f"  → No live 3P offers found")
            errors.append(asin)
            continue

        asin_rows_added = 0
        for offer in offers:
            row = {
                "date": today,
                "asin": asin,
                "brand": asin_row["brand"],
                "product_name": asin_row["product_name"],
                "tag": asin_row["tag"],
                "seller_id": offer["seller_id"],
                "seller_name": "",          # not available at offer level in Keepa
                "condition": "New",
                "fulfillment": "FBA" if offer["is_fba"] else "FBM",
                "price": offer["price_usd"],
                "inventory_est": offer["inventory_est"],
                "ships_from_country": "CN" if offer["ships_from_china"] else "US",
                "is_map": offer["is_map"],
                "is_prime": offer["is_prime"],
            }
            new_rows.append(row)
            asin_rows_added += 1

        print(f"  → {asin_rows_added} 3P seller rows captured")

    # Append all new rows to snapshots.csv
    if new_rows:
        with open(SNAPSHOTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_HEADERS)
            writer.writerows(new_rows)
        print(f"\n✓ Appended {len(new_rows)} rows to {SNAPSHOTS_FILE.name}")
    else:
        print("\nWARNING: No rows written — check API key and ASIN list")

    if errors:
        print(f"\nASINs with errors or no offers: {', '.join(errors)}")

    # Final token status
    try:
        tokens = token_status()
        print(f"Keepa tokens remaining: {tokens['tokens_left']}")
    except Exception:
        pass

    print(f"\nDone — {datetime.now().strftime('%H:%M:%S UTC')}")


if __name__ == "__main__":
    run()
