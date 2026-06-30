"""Load and query Menengai asset catalog from Menengai_Ids.xlsx."""

from functools import lru_cache

import pandas as pd

from config import BASE_DIR
from wialon_core import normalize_plate

ASSETS_FILE = BASE_DIR / "Menengai_Ids.xlsx"


def _name_column(df: pd.DataFrame) -> str:
    for candidate in ("reportname", "name", "unit", "unitname"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Assets file must contain columns like 'ReportName' (or 'Name') and 'itemId'."
    )


@lru_cache(maxsize=1)
def load_assets_catalog() -> tuple[dict, ...]:
    if not ASSETS_FILE.exists():
        raise FileNotFoundError(f"Assets file not found: {ASSETS_FILE.name}")

    df = pd.read_excel(ASSETS_FILE)
    df.columns = [col.strip().lower() for col in df.columns]
    name_col = _name_column(df)
    if "itemid" not in df.columns:
        raise ValueError("Assets file must contain an 'itemId' column.")

    assets = []
    for _, row in df.iterrows():
        item_id = row["itemid"]
        if pd.isna(item_id):
            continue
        name = str(row[name_col]).strip()
        if not name or name.lower() == "nan":
            continue
        assets.append(
            {
                "item_id": int(item_id),
                "name": name,
                "normalized_name": normalize_plate(name),
            }
        )

    assets.sort(key=lambda item: item["name"].upper())
    return tuple(assets)


def clear_assets_cache() -> None:
    load_assets_catalog.cache_clear()


def assets_to_json(limit: int | None = None, query: str = "") -> list[dict]:
    catalog = load_assets_catalog()
    q = query.strip().upper()
    q_norm = normalize_plate(query) if query else ""
    results = []
    for asset in catalog:
        if q:
            name_upper = asset["name"].upper()
            if (
                q not in name_upper
                and (not q_norm or q_norm not in asset["normalized_name"])
            ):
                continue
        results.append(
            {
                "item_id": asset["item_id"],
                "name": asset["name"],
                "normalized_name": asset["normalized_name"],
            }
        )
        if limit and len(results) >= limit:
            break
    return results


def get_asset_by_item_id(item_id: int) -> dict | None:
    for asset in load_assets_catalog():
        if asset["item_id"] == int(item_id):
            return dict(asset)
    return None


def find_asset_by_truck(truck_number_norm: str | None) -> dict | None:
    if not truck_number_norm:
        return None

    catalog = load_assets_catalog()
    for asset in catalog:
        if asset["normalized_name"] == truck_number_norm:
            return dict(asset)

    for asset in catalog:
        if truck_number_norm in asset["normalized_name"]:
            return dict(asset)

    return None


def read_asset_id_from_excel(_excel_file, truck_number_norm):
    """Backward-compatible helper used by legacy modules."""
    asset = find_asset_by_truck(truck_number_norm)
    if asset:
        return asset["item_id"], asset["name"]
    return None, None
