"""
backfill_seller_names.py
------------------------
One-off script to resolve blank seller names in seller_names.csv.
Reads all rows with an empty seller_name, queries Keepa in batches of 100,
and writes the resolved names back in-place.

Run once:
    python backfill_seller_names.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.keepa_client import get_seller_names, token_status

SELLER_NAMES_FILE = ROOT / "data" / "seller_names.csv"


def run():
    if not SELLER_NAMES_FILE.exists():
        print("seller_names.csv not found — nothing to backfill.")
        return

    df = pd.read_csv(SELLER_NAMES_FILE, dtype=str)
    df.columns = df.columns.str.strip()

    blank_mask = df["seller_name"].fillna("").str.strip() == ""
    blank_ids = df.loc[blank_mask, "seller_id"].dropna().tolist()

    if not blank_ids:
        print("No blank seller names found — nothing to do.")
        return

    print(f"Found {len(blank_ids)} seller(s) with blank names. Querying Keepa...")

    try:
        tokens = token_status()
        print(f"Keepa tokens available: {tokens['tokens_left']}")
    except Exception:
        pass

    name_map = get_seller_names(blank_ids)

    resolved = {sid: name for sid, name in name_map.items() if name}
    print(f"Resolved {len(resolved)} of {len(blank_ids)} seller name(s).")

    if not resolved:
        print("No names resolved — check API key or try again later.")
        return

    df["seller_name"] = df.apply(
        lambda row: resolved.get(row["seller_id"], row["seller_name"]),
        axis=1,
    )

    df.to_csv(SELLER_NAMES_FILE, index=False)
    print(f"Written back to {SELLER_NAMES_FILE.name}")

    for sid, name in resolved.items():
        print(f"  {sid} -> {name}")

    still_blank = [sid for sid in blank_ids if sid not in resolved]
    if still_blank:
        print(f"\n{len(still_blank)} seller(s) still unresolved (Keepa returned no data):")
        for sid in still_blank:
            print(f"  {sid}")


if __name__ == "__main__":
    run()
