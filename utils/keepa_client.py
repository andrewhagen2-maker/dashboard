"""
keepa_client.py
Thin wrapper around the Keepa Python library.
Only used by fetch_snapshot.py — never called from the live Streamlit app.

Confirmed offer object fields (from diagnostic run 2026-02-24):
    sellerId        — seller ID string (no sellerName at offer level)
    isFBA           — bool
    shipsFromChina  — bool (only China flag available, not full country)
    isMAP           — bool (Keepa MAP flag)
    isPrime         — bool
    condition       — int (1 = New)
    stockCSV        — [keepa_time, qty, ...] pairs; last qty = current stock
    offerCSV        — price history; use keepa.convert_offer_history() — returns dollars (not cents)
    liveOffersOrder — product-level list of indices into offers[] that are currently active
"""

import os
import keepa
from dotenv import load_dotenv

load_dotenv()

_api = None


def _get_api() -> keepa.Keepa:
    """Lazy-initialize the Keepa API client (one instance per process)."""
    global _api
    if _api is None:
        api_key = os.getenv("KEEPA_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "KEEPA_API_KEY not set. Add it to your .env file or Streamlit Secrets."
            )
        _api = keepa.Keepa(api_key)
    return _api


def get_product_with_offers(asin: str) -> dict | None:
    """
    Return the full product dict for a single ASIN, including live offers,
    stock quantities, and buy box data.

    Token cost: ~10-15 tokens per ASIN (base + offers pages + buybox).

    Returns None if the query fails or returns no products.
    """
    api = _get_api()
    try:
        products = api.query(
            asin,
            history=False,       # skip full price history — saves tokens
            offers=20,           # request up to 20 offers
            only_live_offers=True,
            stock=True,          # include stockCSV for each live offer
            buybox=True,         # include buy box history (+2 tokens)
        )
    except Exception as exc:
        print(f"[keepa_client] ERROR querying ASIN {asin}: {exc}")
        return None

    if not products:
        return None

    return products[0]


def extract_live_offers(product: dict) -> list[dict]:
    """
    Extract and enrich live 3P offers from a product dict.

    Returns a list of dicts with normalised fields:
        seller_id, is_fba, ships_from_china, is_map, is_prime,
        price_usd, inventory_est
    """
    all_offers = product.get("offers", []) or []
    live_indices = product.get("liveOffersOrder", []) or []

    AMAZON_SELLER_ID = "ATVPDKIKX0DER"
    results = []

    for idx in live_indices:
        if idx >= len(all_offers):
            continue

        offer = all_offers[idx]

        # Skip Amazon 1P
        seller_id = offer.get("sellerId", "")
        if seller_id == AMAZON_SELLER_ID:
            continue

        # Skip non-New condition (condition 1 = New)
        if offer.get("condition", 0) != 1:
            continue

        # Price: last value from offerCSV price history
        price_usd = None
        offer_csv = offer.get("offerCSV")
        if offer_csv:
            try:
                _, prices = keepa.convert_offer_history(offer_csv)
                if len(prices) > 0:
                    raw = prices[-1]
                    # convert_offer_history already returns dollars (not cents)
                    if raw and raw > 0:
                        price_usd = round(float(raw), 2)
            except Exception:
                pass

        # Inventory: last value from stockCSV [time, qty, time, qty, ...]
        inventory_est = None
        stock_csv = offer.get("stockCSV")
        if stock_csv and len(stock_csv) >= 2:
            last_qty = stock_csv[-1]
            if isinstance(last_qty, (int, float)) and last_qty >= 0:
                inventory_est = int(last_qty)

        results.append({
            "seller_id": seller_id,
            "is_fba": bool(offer.get("isFBA", False)),
            "ships_from_china": bool(offer.get("shipsFromChina", False)),
            "is_map": bool(offer.get("isMAP", False)),
            "is_prime": bool(offer.get("isPrime", False)),
            "price_usd": price_usd,
            "inventory_est": inventory_est,
        })

    return results


def get_seller_names(seller_ids: list[str]) -> dict[str, str]:
    """
    Look up seller display names for a list of seller IDs via the Keepa
    seller API. Returns a dict of {seller_id: seller_name}.

    Unknown or failed lookups are omitted from the result — the caller
    should handle missing keys gracefully (fall back to seller_id).

    Token cost: low — typically 1 token per seller, batched in one call.
    """
    if not seller_ids:
        return {}

    api = _get_api()
    try:
        sellers = api.seller(seller_ids)
    except Exception as exc:
        print(f"[keepa_client] ERROR looking up seller names: {exc}")
        return {}

    result = {}
    for sid, data in sellers.items():
        if data is None:
            continue
        # Keepa returns 'sellerName' in the seller profile object
        name = data.get("sellerName") or data.get("name")
        if name:
            result[sid] = str(name).strip()

    return result


def token_status() -> dict:
    """
    Return remaining Keepa API tokens for the current key.

    api.tokens_left is only populated after an API call is made (it comes from
    response headers). To get a fresh count at any point, we make a minimal
    seller query with an empty list — zero cost, updates the counter.
    """
    api = _get_api()
    try:
        # Empty seller query: costs 0 tokens, but forces a round-trip that
        # populates api.tokens_left from the response headers.
        api.seller([])
    except Exception:
        pass  # non-fatal — fall through to whatever value is cached
    return {
        "tokens_left": api.tokens_left,
    }
