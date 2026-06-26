import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

CSV_PATH = "monster_ev.csv"
MANUAL_PRICE_PATH = "manual_prices.json"
MANUAL_PRICE_EXAMPLE_PATH = "manual_prices.example.json"
ELEMENT_PREFIXES = ("Ele_", "ELE_", "Element_", "ELEMENT_")
PORING_COIN_KEY = "__uaro_poring_coin__"
PORING_COIN_NAME = "Poring Coin"
PORING_COIN_CHANCE = 500.0  # 10000 = 100%, so 500 = 5%.
GREAT_NATURE_AVERAGE_GREEN_LIVES = 7.5
UARO_PRICE_OVERRIDES = {
    "Yellow_Live": 400,
    "Green_Live": 400,
    "Mastela_Fruit": 3500,
    "Mastela": 3500,
    "Crystal_Mirror": 6000,
    "Royal_Jelly": 2750,
}
UARO_NAME_PRICE_OVERRIDES = {
    "green live": 400,
    "green lives": 400,
    "mastela fruit": 3500,
    "mastela": 3500,
    "crystal mirror": 6000,
    "royal jelly": 2750,
}
GREAT_NATURE_KEYS = {"Great_Nature"}
GREAT_NATURE_NAMES = {"great nature"}
GREEN_LIVE_KEYS = {"Yellow_Live", "Green_Live"}
SORT_LABELS = {
    "expected_value": "Expected Value",
    "map_value_score": "Map score",
    "best_map_count": "Best-map spawns",
    "level": "Level",
    "hp": "HP",
    "name": "Monster name",
}
HELP_TEXT = {
    "best_farms": """
Use this tab as the main mob search table.

- **Expected Value** is the average zeny value of one kill after the current drop multiplier, Overcharge setting, UARO pricing, manual prices, and optional Poring Coin value are applied.
- **Map score** is `Expected Value * Best-map spawns`. It is a density proxy, not a true zeny-per-hour estimate.
- **Main value drops** shows the drops contributing most of the mob's value. The percentage is that drop's share of the mob's total expected value.
- **Best map** and **Best-map spawns** come from parsed spawn data. A mob can appear on other maps too.
- Boss-flagged monsters and monsters with MVP drops are hidden by default in the sidebar.

Select a row to open the drop breakdown and zeny/hour estimate underneath the table.
    """,
    "maps": """
Use this tab when you want to start from a location instead of a specific mob.

- The top table groups all matching mobs by map.
- **Total spawns** is the sum of parsed spawn counts for matching mobs on that map.
- **Total map score** is the sum of each mob's `Expected Value * spawn count` on that map.
- **Average EV** and **Best EV** summarize the expected values of mobs found there.
- **Highest-score mob** is the mob contributing the highest individual map score on that map.

Select a map row to open the list of mobs that spawn there.
    """,
    "items": """
Use this tab to inspect and override item prices.

- **NPC Sell** shows the baseline sell value currently used by the app. If UARO pricing is enabled, UARO-adjusted item prices and Great Nature conversion are reflected here.
- **Manual Price** is optional. Fill it in to override NPC/UARO/conversion pricing for that item.
- **Active Price** is the final price used in EV calculations after manual pricing and Overcharge rules are applied.
- Manual prices override NPC/UARO/conversion prices and are not multiplied by Overcharge.

After editing Manual Price values, press **Apply manual prices**.
    """,
    "raw": """
Use this tab as the escape hatch for inspecting the generated dataset.

- It shows the filtered rows after sidebar filters and current price assumptions are applied.
- The generated `drops_json` column is hidden here because it is large and mainly used internally for recalculation.
- Use **Download filtered CSV** if you want to inspect the current filtered result elsewhere.
- For data problems, regenerate `monster_ev.csv` from the source monster, item, and spawn data.
    """,
}

st.set_page_config(page_title="Mob Value Planner", layout="wide")


def apply_layout_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"],
        section[data-testid="stMain"] > div,
        section.main > div.block-container,
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: min(1700px, calc(100vw - 3rem)) !important;
            width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"],
            section[data-testid="stMain"] > div,
            section.main > div.block-container,
            [data-testid="stAppViewContainer"] .main .block-container {
                max-width: 100% !important;
                width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """.strip(),
        unsafe_allow_html=True,
    )


def render_tab_help(tab_key: str) -> None:
    text = HELP_TEXT.get(tab_key, "").strip()
    if not text:
        return
    with st.expander("Help / explanations", expanded=False):
        st.markdown(text)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace("%", "").replace(",", "")
            if not value:
                return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def pretty_enum(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    for prefix in ELEMENT_PREFIXES + ("RC_", "Size_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.replace("_", " ").strip()


def clean_item_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    return text.replace("_", " ").strip()


def normalized_name(value: Any) -> str:
    return clean_item_name(value).lower()


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def safe_min_max(series: pd.Series, default_min: int = 0, default_max: int = 0) -> Tuple[int, int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return default_min, default_max
    return int(numeric.min()), int(numeric.max())


def rounded_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
    return out


def format_percent(value: Any) -> str:
    number = as_float(value, 0.0)
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def format_zeny(value: Any) -> str:
    number = as_float(value, 0.0)
    return f"{number:,.0f}z"


@st.cache_data
def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run `python generate_monster_ev.py` and commit the CSV.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    numeric_defaults = {
        "id": 0,
        "level": 0,
        "hp": 0,
        "best_map_count": 0,
        "map_count": 0,
        "total_spawn_count": 0,
        "drop_count": 0,
        "mvp_drop_count": 0,
        "missing_item_count": 0,
        "expected_value_raw": 0.0,
        "expected_value": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
            if isinstance(default, int):
                df[col] = df[col].astype(int)

    for col in ["is_boss", "has_mvp_drops", "ev_generation_cap_drop_rate", "mvp_drops_included_in_ev"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    for col in ["name", "sprite_name", "internal_name", "best_map", "race", "size", "element", "spawn_summary", "drops_summary", "drops_json"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["element_display"] = df["element"].apply(pretty_enum) if "element" in df.columns else ""
    return df


def parse_drops_json(value: Any) -> List[Dict[str, Any]]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [drop for drop in parsed if isinstance(drop, dict)] if isinstance(parsed, list) else []


def parse_spawn_summary(value: Any) -> Dict[str, int]:
    spawns: Dict[str, int] = {}
    text = str(value or "").strip()
    if not text:
        return spawns
    for piece in text.split(";"):
        part = piece.strip()
        if not part or ":" not in part:
            continue
        map_name, count_text = part.rsplit(":", 1)
        map_name = map_name.strip()
        count = as_int(count_text.strip(), 0)
        if map_name and count > 0:
            spawns[map_name] = spawns.get(map_name, 0) + count
    return spawns


def drop_item_key(drop: Dict[str, Any]) -> str:
    return str(drop.get("aegis_name") or drop.get("key") or drop.get("name") or "").strip()


def drop_item_name(drop: Dict[str, Any]) -> str:
    return clean_item_name(drop.get("name") or drop.get("aegis_name") or drop.get("key"))


def normalize_manual_prices(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("prices"), dict):
        raw = raw["prices"]
    prices: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        item_key = str(key).strip()
        if not item_key:
            continue
        if isinstance(value, dict):
            price = as_float(value.get("price"), -1.0)
            name = str(value.get("name") or value.get("item") or item_key).strip()
        else:
            price = as_float(value, -1.0)
            name = item_key
        if price >= 0:
            prices[item_key] = {"name": name or item_key, "price": float(price)}
    return prices


def load_price_file(path: str | Path) -> Dict[str, Dict[str, Any]]:
    price_path = Path(path)
    if not price_path.exists():
        return {}
    try:
        return normalize_manual_prices(json.loads(price_path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def export_price_payload(prices: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    clean = {}
    for key, value in sorted(prices.items(), key=lambda item: str(item[1].get("name") or item[0]).lower()):
        price = as_float(value.get("price"), 0.0)
        clean[str(key)] = {"name": str(value.get("name") or key), "price": int(price) if price.is_integer() else price}
    return {"name": name, "format": "mob-value-planner.price-table.v1", "prices": clean}


def init_price_state() -> None:
    if "personal_prices" not in st.session_state:
        st.session_state.personal_prices = load_price_file(MANUAL_PRICE_PATH)


def manual_price_for_key(item_key: str, item_name: str, manual_prices: Dict[str, Dict[str, Any]]) -> float | None:
    candidates = [str(item_key or "").strip(), str(item_name or "").strip()]
    for key in candidates:
        if key in manual_prices:
            price = as_float(manual_prices[key].get("price"), -1.0)
            if price >= 0:
                return price
    target_name = normalized_name(item_name)
    if target_name:
        for value in manual_prices.values():
            if normalized_name(value.get("name")) == target_name:
                price = as_float(value.get("price"), -1.0)
                if price >= 0:
                    return price
    return None


def manual_price_for_drop(drop: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> float | None:
    return manual_price_for_key(drop_item_key(drop), drop_item_name(drop), manual_prices)


def uaro_override_price(item_key: str, item_name: str) -> float | None:
    if item_key in UARO_PRICE_OVERRIDES:
        return float(UARO_PRICE_OVERRIDES[item_key])
    name = normalized_name(item_name)
    if name in UARO_NAME_PRICE_OVERRIDES:
        return float(UARO_NAME_PRICE_OVERRIDES[name])
    return None


def is_green_live(item_key: str, item_name: str) -> bool:
    return item_key in GREEN_LIVE_KEYS or normalized_name(item_name) in {"green live", "green lives"}


def is_great_nature(item_key: str, item_name: str) -> bool:
    return item_key in GREAT_NATURE_KEYS or normalized_name(item_name) in GREAT_NATURE_NAMES


def base_sell_price_for_item(item_key: str, item_name: str, base_sell: Any, use_uaro_prices: bool) -> Tuple[float, str]:
    npc_sell = as_float(base_sell, 0.0)
    if not use_uaro_prices:
        return npc_sell, "NPC"

    if is_great_nature(item_key, item_name):
        green_live_base = uaro_override_price("Yellow_Live", "Green Live") or 400.0
        return green_live_base * GREAT_NATURE_AVERAGE_GREEN_LIVES, "UARO conversion"

    override = uaro_override_price(item_key, item_name)
    if override is not None:
        return override, "UARO"

    return npc_sell, "NPC"


def apply_overcharge(price: float, use_overcharge: bool, overcharge_rate: float, ignore_overcharge: bool) -> float:
    if use_overcharge and not ignore_overcharge:
        return int(price * overcharge_rate)
    return price


def adjusted_drop_chance(raw_chance: Any, multiplier: float, fixed_chance: bool = False) -> float:
    raw = as_float(raw_chance, 0.0)
    if fixed_chance:
        return min(max(raw, 0.0), 10000.0)
    return min(max(raw * multiplier, 0.0), 10000.0)


def synthetic_poring_coin_drop(price: float) -> Dict[str, Any]:
    return {
        "key": PORING_COIN_KEY,
        "name": PORING_COIN_NAME,
        "raw_chance": PORING_COIN_CHANCE,
        "base_sell_price": price,
        "sell_price": price,
        "ignore_overcharge": True,
        "is_mvp_drop": False,
        "missing_item": False,
        "fixed_chance": True,
        "synthetic": True,
    }


def drops_with_synthetic(drops: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = list(drops)
    if settings.get("include_poring_coin"):
        out.append(synthetic_poring_coin_drop(as_float(settings.get("poring_coin_price"), 12000.0)))
    return out


def resolve_drop_price(drop: Dict[str, Any], settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    item_key = drop_item_key(drop)
    item_name = drop_item_name(drop)
    manual = manual_price_for_drop(drop, manual_prices)
    if manual is not None:
        return {
            "base_price": manual,
            "active_price": manual,
            "manual_price": manual,
            "source": "Manual",
        }

    if bool(drop.get("synthetic")):
        price = as_float(drop.get("base_sell_price", drop.get("sell_price")), 0.0)
        return {
            "base_price": price,
            "active_price": price,
            "manual_price": None,
            "source": "Custom",
        }

    base_price, base_source = base_sell_price_for_item(
        item_key,
        item_name,
        drop.get("base_sell_price", drop.get("sell_price")),
        bool(settings.get("use_uaro_prices")),
    )
    ignore_overcharge = bool(drop.get("ignore_overcharge"))
    active = apply_overcharge(base_price, bool(settings.get("use_overcharge")), as_float(settings.get("overcharge_rate"), 1.24), ignore_overcharge)
    source = base_source
    if bool(settings.get("use_overcharge")) and not ignore_overcharge:
        source = f"{source} + Overcharge"
    return {
        "base_price": base_price,
        "active_price": active,
        "manual_price": None,
        "source": source,
    }


def drop_details_dataframe(drops: Iterable[Dict[str, Any]], settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for drop in drops:
        raw = as_float(drop.get("raw_chance"), 0.0)
        adjusted = adjusted_drop_chance(raw, settings["multiplier"], bool(drop.get("fixed_chance")))
        pricing = resolve_drop_price(drop, settings, manual_prices)
        sell = as_float(pricing["active_price"], 0.0)
        ev = 0.0 if bool(drop.get("missing_item")) else sell * adjusted / 10000.0
        rows.append(
            {
                "Item": drop_item_name(drop),
                "Item ID": "" if is_blank(drop.get("item_id")) else as_int(drop.get("item_id")),
                "Expected Value": ev,
                "EV Share": 0.0,
                "Adjusted Chance": adjusted / 100.0,
                "Effective Sell": int(sell) if float(sell).is_integer() else sell,
                "Price Source": pricing["source"],
                "Base Chance": raw / 100.0,
                "NPC Sell": int(pricing["base_price"]) if float(pricing["base_price"]).is_integer() else pricing["base_price"],
                "Manual Price": "" if pricing["manual_price"] is None else int(pricing["manual_price"]) if float(pricing["manual_price"]).is_integer() else pricing["manual_price"],
                "Type": "Custom" if bool(drop.get("synthetic")) else "MVP" if bool(drop.get("is_mvp_drop")) else "Normal",
                "Missing Item": bool(drop.get("missing_item")),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    total = as_float(detail["Expected Value"].sum(), 0.0)
    detail["EV Share"] = detail["Expected Value"] / total * 100.0 if total > 0 else 0.0
    preferred = ["Item", "Item ID", "Expected Value", "EV Share", "Adjusted Chance", "Effective Sell", "Price Source", "NPC Sell", "Manual Price", "Base Chance", "Type", "Missing Item"]
    detail = detail[preferred].sort_values("Expected Value", ascending=False, kind="stable").reset_index(drop=True)
    detail = rounded_table(detail, ["Expected Value", "Effective Sell", "NPC Sell"])
    for percent_col in ["EV Share", "Adjusted Chance", "Base Chance"]:
        detail[percent_col] = detail[percent_col].apply(format_percent)
    return detail


def summarize_drops(detail: pd.DataFrame, limit: int = 3) -> str:
    if detail.empty:
        return ""
    return ", ".join(f"{row['Item']} ({as_float(row['EV Share']):.0f}%)" for _, row in detail.head(limit).iterrows())


def top_value_share_from_detail(detail: pd.DataFrame) -> float:
    if detail.empty or as_float(detail["Expected Value"].sum(), 0.0) <= 0:
        return 0.0
    total = as_float(detail["Expected Value"].sum(), 0.0)
    return as_float(detail["Expected Value"].max(), 0.0) / total * 100.0


def recalc_monster(row: pd.Series, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    drops = drops_with_synthetic(parse_drops_json(row.get("drops_json", "")), settings)
    detail = drop_details_dataframe(drops, settings, manual_prices)
    return {
        "expected_value": as_float(detail["Expected Value"].sum(), 0.0) if not detail.empty else as_float(row.get("expected_value"), 0.0),
        "top_drops": summarize_drops(detail),
        "top_value_share": top_value_share_from_detail(detail),
    }


def apply_ui_ev_settings(df: pd.DataFrame, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    if "drops_json" in df.columns:
        derived = df.apply(lambda row: recalc_monster(row, settings, manual_prices), axis=1)
        df["expected_value"] = derived.apply(lambda d: d["expected_value"])
        df["top_drops"] = derived.apply(lambda d: d["top_drops"])
        df["top_value_share"] = derived.apply(lambda d: d["top_value_share"])
    else:
        df["top_drops"] = df.get("drops_summary", "")
        df["top_value_share"] = 0.0
    df["map_value_score"] = numeric_series(df, "expected_value") * numeric_series(df, "best_map_count")
    return df


def extract_item_catalog(df: pd.DataFrame, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    catalog: Dict[str, Dict[str, Any]] = {}
    if "drops_json" not in df.columns:
        return pd.DataFrame(columns=["_key", "Item", "Item ID", "NPC Sell", "Manual Price", "Active Price", "Price Source", "Used By"])

    for _, monster in df.iterrows():
        monster_id = as_int(monster.get("id"), 0)
        for drop in parse_drops_json(monster.get("drops_json", "")):
            key = drop_item_key(drop)
            if not key:
                continue
            name = drop_item_name(drop)
            entry = catalog.setdefault(
                key,
                {
                    "_key": key,
                    "Item": name,
                    "Item ID": "" if is_blank(drop.get("item_id")) else as_int(drop.get("item_id")),
                    "base_sell_price": as_float(drop.get("base_sell_price", drop.get("sell_price")), 0.0),
                    "ignore_overcharge": bool(drop.get("ignore_overcharge")),
                    "monster_ids": set(),
                },
            )
            if monster_id:
                entry["monster_ids"].add(monster_id)

    rows = []
    for key, entry in catalog.items():
        probe = {
            "key": key,
            "name": entry["Item"],
            "item_id": entry["Item ID"],
            "base_sell_price": entry["base_sell_price"],
            "ignore_overcharge": entry["ignore_overcharge"],
            "missing_item": False,
        }
        pricing = resolve_drop_price(probe, settings, manual_prices)
        manual = pricing["manual_price"]
        rows.append(
            {
                "_key": key,
                "Item": entry["Item"],
                "Item ID": entry["Item ID"],
                "NPC Sell": pricing["base_price"],
                "Manual Price": None if manual is None else manual,
                "Active Price": pricing["active_price"],
                "Price Source": pricing["source"],
                "Used By": len(entry["monster_ids"]),
            }
        )

    if settings.get("include_poring_coin"):
        coin_drop = synthetic_poring_coin_drop(as_float(settings.get("poring_coin_price"), 12000.0))
        pricing = resolve_drop_price(coin_drop, settings, manual_prices)
        rows.append(
            {
                "_key": PORING_COIN_KEY,
                "Item": PORING_COIN_NAME,
                "Item ID": "",
                "NPC Sell": pricing["base_price"],
                "Manual Price": settings.get("poring_coin_price"),
                "Active Price": pricing["active_price"],
                "Price Source": pricing["source"],
                "Used By": len(df),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["_key", "Item", "Item ID", "NPC Sell", "Manual Price", "Active Price", "Price Source", "Used By"])

    out = pd.DataFrame(rows).sort_values(["Item", "_key"], kind="stable").reset_index(drop=True)
    return rounded_table(out, ["NPC Sell", "Manual Price", "Active Price"])


def prices_from_items_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    prices: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return prices
    for _, row in df.iterrows():
        key = str(row.get("_key") or "").strip()
        if not key or key == PORING_COIN_KEY:
            continue
        value = row.get("Manual Price")
        if is_blank(value):
            continue
        price = as_float(value, -1.0)
        if price < 0:
            continue
        name = str(row.get("Item") or key).strip() or key
        prices[key] = {"name": name, "price": float(price)}
    return prices


def render_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    st.sidebar.header("Assumptions")
    multiplier = st.sidebar.number_input("Drop rate multiplier", min_value=0.0, value=5.0, step=0.5)
    use_overcharge = st.sidebar.checkbox("Apply merchant Overcharge (+24%)", value=True)
    overcharge_rate = st.sidebar.number_input("Overcharge multiplier", min_value=1.0, value=1.24, step=0.01, format="%.2f", disabled=not use_overcharge)
    use_uaro_prices = st.sidebar.checkbox("Use UARO adjusted prices", value=True)
    st.sidebar.caption("UARO pricing applies adjusted item prices and Great Nature conversion before Overcharge.")

    st.sidebar.header("Server bonuses")
    include_poring_coin = st.sidebar.checkbox("Include Poring Coin drops", value=False)
    poring_coin_price = st.sidebar.number_input("Poring Coin price", min_value=0, value=12000, step=500, disabled=not include_poring_coin)
    if include_poring_coin:
        st.sidebar.caption("Adds a fixed 5% Poring Coin drop to every monster. This chance is not multiplied by the drop-rate setting.")

    st.sidebar.header("Farm filters")
    name_query = st.sidebar.text_input("Monster name contains", value="")
    map_query = st.sidebar.text_input("Map contains", value="")
    element_map: Dict[str, List[str]] = {}
    if "element" in df.columns and not df.empty:
        pairs = df[["element", "element_display"]].drop_duplicates().sort_values(["element_display", "element"])
        for _, row in pairs.iterrows():
            raw = str(row.get("element") or "").strip()
            label = str(row.get("element_display") or raw).strip()
            if raw:
                element_map.setdefault(label, []).append(raw)
    selected_elements = st.sidebar.multiselect("Element", list(element_map.keys()), default=[])
    lvl_min, lvl_max = safe_min_max(df["level"] if "level" in df.columns else pd.Series(dtype=int))
    level_range = st.sidebar.slider("Level range", lvl_min, lvl_max, (lvl_min, lvl_max))
    sp_min, sp_max = safe_min_max(df["best_map_count"] if "best_map_count" in df.columns else pd.Series(dtype=int))
    spawn_range = st.sidebar.slider("Best-map spawn count", sp_min, sp_max, (max(1, sp_min) if sp_max >= 1 else sp_min, sp_max))
    min_ev = st.sidebar.number_input("Minimum EV", min_value=0.0, value=0.0, step=1.0)
    include_boss = st.sidebar.checkbox("Include boss-flagged monsters", value=False) if "is_boss" in df.columns else True
    include_mvp = st.sidebar.checkbox("Include monsters with MVP drops", value=False) if "has_mvp_drops" in df.columns else True
    sort_candidates = [c for c in ["expected_value", "map_value_score", "best_map_count", "level", "hp", "name"] if c in df.columns or c == "map_value_score"]
    sort_by = st.sidebar.selectbox("Sort by", sort_candidates, index=0 if sort_candidates else None, format_func=lambda c: SORT_LABELS.get(c, c))
    ascending = st.sidebar.checkbox("Ascending sort", value=False)
    return {
        "multiplier": multiplier,
        "use_overcharge": use_overcharge,
        "overcharge_rate": overcharge_rate,
        "use_uaro_prices": use_uaro_prices,
        "include_poring_coin": include_poring_coin,
        "poring_coin_price": poring_coin_price,
        "name_query": name_query,
        "map_query": map_query,
        "element_map": element_map,
        "selected_elements": selected_elements,
        "level_range": level_range,
        "spawn_range": spawn_range,
        "min_ev": min_ev,
        "include_boss": include_boss,
        "include_mvp": include_mvp,
        "sort_by": sort_by,
        "ascending": ascending,
    }


def filter_dataframe(df: pd.DataFrame, s: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if s["selected_elements"] and "element" in out.columns:
        allowed = {raw for label in s["selected_elements"] for raw in s["element_map"].get(label, [])}
        out = out[out["element"].isin(allowed)]
    if "level" in out.columns:
        out = out[(out["level"] >= s["level_range"][0]) & (out["level"] <= s["level_range"][1])]
    if "best_map_count" in out.columns:
        out = out[(out["best_map_count"] >= s["spawn_range"][0]) & (out["best_map_count"] <= s["spawn_range"][1])]
    if "expected_value" in out.columns:
        out = out[out["expected_value"] >= s["min_ev"]]
    if s["name_query"].strip() and "name" in out.columns:
        q = s["name_query"].strip().lower()
        mask = out["name"].str.lower().str.contains(q, regex=False)
        for col in ["sprite_name", "internal_name"]:
            if col in out.columns:
                mask = mask | out[col].str.lower().str.contains(q, regex=False)
        out = out[mask]
    if s["map_query"].strip() and "best_map" in out.columns:
        q = s["map_query"].strip().lower()
        mask = out["best_map"].str.lower().str.contains(q, regex=False)
        if "spawn_summary" in out.columns:
            mask = mask | out["spawn_summary"].str.lower().str.contains(q, regex=False)
        out = out[mask]
    if not s["include_boss"] and "is_boss" in out.columns:
        out = out[~out["is_boss"]]
    if not s["include_mvp"] and "has_mvp_drops" in out.columns:
        out = out[~out["has_mvp_drops"]]
    if s["sort_by"]:
        out = out.sort_values(s["sort_by"], ascending=s["ascending"], kind="stable")
    return out


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["name", "expected_value", "map_value_score", "top_drops", "level", "hp", "element_display", "best_map", "best_map_count", "is_boss", "has_mvp_drops", "id"]
    visible = [c for c in cols if c in df.columns]
    table = df[visible].rename(columns={"name": "Monster", "expected_value": "Expected Value", "map_value_score": "Map score", "top_drops": "Main value drops", "element_display": "Element", "best_map": "Best map", "best_map_count": "Best-map spawns", "is_boss": "Boss", "has_mvp_drops": "Has MVP drops", "id": "ID", "level": "Level", "hp": "HP"})
    return rounded_table(table, ["Expected Value", "Map score"])


def select_row_from_table(table_df: pd.DataFrame, source_df: pd.DataFrame, fallback_label: str) -> pd.Series | None:
    selected_position = None
    try:
        event = st.dataframe(table_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        rows = getattr(getattr(event, "selection", None), "rows", [])
        if rows:
            selected_position = int(rows[0])
    except TypeError:
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        if not source_df.empty:
            opts = {str(label): idx for idx, label in enumerate(table_df.iloc[:, 0].astype(str).tolist())}
            chosen = st.selectbox(fallback_label, [""] + list(opts.keys()), index=0)
            if chosen:
                selected_position = opts[chosen]
    if selected_position is None or selected_position < 0 or selected_position >= len(source_df):
        return None
    return source_df.reset_index(drop=True).iloc[selected_position]


def render_metrics(df: pd.DataFrame, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> None:
    cols = st.columns(6)
    cols[0].metric("Matching mobs", f"{len(df):,}")
    cols[1].metric("Highest EV", f"{df['expected_value'].max() if len(df) and 'expected_value' in df else 0:,.0f}")
    cols[2].metric("Median EV", f"{df['expected_value'].median() if len(df) and 'expected_value' in df else 0:,.0f}")
    cols[3].metric("Highest map score", f"{df['map_value_score'].max() if len(df) and 'map_value_score' in df else 0:,.0f}")
    cols[4].metric("Price profile", "UARO" if settings["use_uaro_prices"] else "NPC")
    cols[5].metric("Manual prices", f"{len(manual_prices):,}")


def render_zeny_per_hour(row: pd.Series) -> None:
    st.subheader("Zeny/hour estimate")
    ev = as_float(row.get("expected_value"), 0.0)
    kills_per_30 = st.number_input(
        "Monsters killed per 30 min",
        min_value=0,
        value=0,
        step=25,
        help="Enter your own observed or expected kill count for this monster.",
        key=f"kills_per_30_{as_int(row.get('id'), 0)}",
    )
    kills_per_hour = kills_per_30 * 2
    zeny_per_hour = ev * kills_per_hour
    cols = st.columns(3)
    cols[0].metric("EV per kill", format_zeny(ev))
    cols[1].metric("Kills/hour", f"{kills_per_hour:,}")
    cols[2].metric("Estimated zeny/hour", format_zeny(zeny_per_hour))
    st.caption("This is an expected-value estimate. Actual drops will vary.")


def render_selected_monster_drops(row: pd.Series, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> None:
    st.subheader(f"Drops for {row.get('name', 'selected monster')}")
    cols = st.columns(6)
    cols[0].metric("Monster ID", str(row.get("id", "")))
    cols[1].metric("Level", str(row.get("level", "")))
    cols[2].metric("Element", row.get("element_display") or "-")
    cols[3].metric("Best map", row.get("best_map") or "-")
    cols[4].metric("Spawns", f"{as_int(row.get('best_map_count')):,}")
    cols[5].metric("EV", f"{as_float(row.get('expected_value')):,.0f}")

    render_zeny_per_hour(row)

    drops = drops_with_synthetic(parse_drops_json(row.get("drops_json", "")), settings)
    detail = drop_details_dataframe(drops, settings, manual_prices)
    if detail.empty:
        st.info(row.get("drops_summary") or "No drop details are available. Regenerate the CSV with drops_json if needed.")
        return
    capped = int(detail["Adjusted Chance"].apply(lambda value: as_float(value) >= 100).sum())
    st.caption(f"Main value: {summarize_drops(detail, 5) or '-'} | capped drops: {capped}")
    st.dataframe(detail, use_container_width=True, hide_index=True)
    with st.expander("Spawn summary"):
        st.write(str(row.get("spawn_summary") or "No spawn summary available."))


def render_best_farms(df: pd.DataFrame, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> None:
    st.subheader("Mobs")
    render_tab_help("best_farms")
    st.caption("Filtered and sorted mob table. Select a row to inspect its drop value breakdown below.")
    if df.empty:
        st.info("No monsters match the current filters.")
        return
    source_df = df.reset_index(drop=True)
    selected = select_row_from_table(clean_table(source_df), source_df, "Inspect monster drops")
    if selected is not None:
        render_selected_monster_drops(selected, settings, manual_prices)
    else:
        st.info("Select a monster row above to show its drop value breakdown here.")


def build_map_monster_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_idx, row in df.reset_index(drop=True).iterrows():
        spawns = parse_spawn_summary(row.get("spawn_summary", ""))
        if not spawns and str(row.get("best_map") or "").strip():
            spawns = {str(row.get("best_map")).strip(): as_int(row.get("best_map_count"), 0)}
        for map_name, count in spawns.items():
            ev = as_float(row.get("expected_value"), 0.0)
            rows.append(
                {
                    "source_idx": source_idx,
                    "Map": map_name,
                    "Monster": row.get("name"),
                    "Spawn count": count,
                    "Expected Value": ev,
                    "Map score": ev * count,
                    "Main value drops": row.get("top_drops"),
                    "Level": as_int(row.get("level"), 0),
                    "HP": as_int(row.get("hp"), 0),
                    "Element": row.get("element_display"),
                    "ID": row.get("id"),
                }
            )
    return pd.DataFrame(rows)


def render_maps(df: pd.DataFrame) -> None:
    st.subheader("Maps")
    render_tab_help("maps")
    if df.empty:
        st.info("No map data is available under the current filters.")
        return
    map_monsters = build_map_monster_rows(df)
    if map_monsters.empty:
        st.info("No parsed spawn locations are available.")
        return
    best_idx = map_monsters.groupby("Map")["Map score"].idxmax()
    best = map_monsters.loc[best_idx, ["Map", "Monster", "Map score"]].rename(columns={"Monster": "Highest-score mob", "Map score": "Highest mob score"})
    grouped = (
        map_monsters.groupby("Map")
        .agg(
            Monsters=("Monster", "count"),
            Total_spawns=("Spawn count", "sum"),
            Total_map_score=("Map score", "sum"),
            Average_EV=("Expected Value", "mean"),
            Best_EV=("Expected Value", "max"),
        )
        .reset_index()
        .merge(best, on="Map", how="left")
        .sort_values("Total_map_score", ascending=False)
        .reset_index(drop=True)
    )
    grouped_display = grouped.rename(columns={"Total_spawns": "Total spawns", "Total_map_score": "Total map score", "Average_EV": "Average EV", "Best_EV": "Best EV"})
    selected = select_row_from_table(rounded_table(grouped_display, ["Total map score", "Average EV", "Best EV", "Highest mob score"]), grouped, "Inspect map monsters")
    if selected is None:
        st.info("Select a map row above to show matching monsters here.")
        return
    selected_map = str(selected.get("Map") or "")
    st.subheader(f"Monsters on {selected_map}")
    detail = map_monsters[map_monsters["Map"] == selected_map].sort_values("Map score", ascending=False, kind="stable")
    st.dataframe(rounded_table(detail.drop(columns=["source_idx"], errors="ignore"), ["Expected Value", "Map score"]), use_container_width=True, hide_index=True)


def render_items(raw_df: pd.DataFrame, settings: Dict[str, Any], manual_prices: Dict[str, Dict[str, Any]]) -> None:
    st.subheader("Items")
    render_tab_help("items")
    catalog = extract_item_catalog(raw_df, settings, manual_prices)
    if catalog.empty:
        st.info("No item catalog is available. Regenerate monster_ev.csv with drops_json.")
        return

    query = st.text_input("Search items", value="")
    display = catalog
    if query.strip():
        q = query.strip().lower()
        mask = display["Item"].astype(str).str.lower().str.contains(q, regex=False)
        if "Item ID" in display.columns:
            mask = mask | display["Item ID"].astype(str).str.lower().str.contains(q, regex=False)
        display = display[mask]

    st.caption("Edit Manual Price directly. Blank values use NPC/UARO/conversion pricing.")
    edited_df = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="item_price_editor",
        disabled=["Item", "Item ID", "NPC Sell", "Active Price", "Price Source", "Used By"],
        column_config={
            "_key": None,
            "Item": st.column_config.TextColumn("Item"),
            "Item ID": st.column_config.TextColumn("Item ID"),
            "NPC Sell": st.column_config.NumberColumn("NPC Sell", format="%d z"),
            "Manual Price": st.column_config.NumberColumn("Manual Price", min_value=0, step=100, format="%d z"),
            "Active Price": st.column_config.NumberColumn("Active Price", format="%d z"),
            "Price Source": st.column_config.TextColumn("Price Source"),
            "Used By": st.column_config.NumberColumn("Used By"),
        },
    )

    b1, b2, b3 = st.columns(3)
    if b1.button("Apply manual prices", use_container_width=True):
        updated = prices_from_items_dataframe(catalog)
        edited_updates = prices_from_items_dataframe(edited_df)
        visible_keys = set(edited_df["_key"].astype(str).tolist()) if "_key" in edited_df.columns else set()
        updated = {key: value for key, value in updated.items() if key not in visible_keys}
        updated.update(edited_updates)
        st.session_state.personal_prices = updated
        st.rerun()
    if b2.button("Clear manual prices", disabled=not bool(st.session_state.personal_prices), use_container_width=True):
        st.session_state.personal_prices = {}
        st.rerun()
    b3.download_button(
        "Export manual prices",
        json.dumps(export_price_payload(st.session_state.personal_prices, "Manual prices"), ensure_ascii=False, indent=2),
        file_name="manual_prices.json",
        mime="application/json",
        use_container_width=True,
        disabled=not bool(st.session_state.personal_prices),
    )

    st.divider()
    st.subheader("Import manual prices")
    upload = st.file_uploader("Upload manual price JSON", type=["json"])
    pasted = st.text_area("Or paste JSON", height=120)
    replace_existing = st.checkbox("Replace current manual prices", value=True)
    if st.button("Import", use_container_width=True):
        try:
            raw_text = upload.getvalue().decode("utf-8") if upload is not None else pasted
            raw = json.loads(raw_text)
            imported = normalize_manual_prices(raw)
            if imported:
                st.session_state.personal_prices = imported if replace_existing else {**st.session_state.personal_prices, **imported}
                st.success(f"Imported {len(imported):,} manual price override(s).")
                st.rerun()
            else:
                st.warning("No prices found in that JSON.")
        except Exception as exc:
            st.error(f"Could not import manual prices: {exc}")


def render_raw(df: pd.DataFrame) -> None:
    st.subheader("Raw data")
    render_tab_help("raw")
    st.download_button("Download filtered CSV", df.to_csv(index=False).encode("utf-8"), file_name="mob_value_filtered_monsters.csv", mime="text/csv", disabled=df.empty)
    st.dataframe(df.drop(columns=["drops_json"], errors="ignore"), use_container_width=True, hide_index=True)


def main() -> None:
    apply_layout_css()
    st.title("Mob Value Planner")
    st.caption("Monster value explorer, farming comparison tool, and price-table sandbox.")
    init_price_state()
    try:
        raw_df = load_data(CSV_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    if raw_df.empty:
        st.warning("`monster_ev.csv` is empty or missing usable rows. Run `python generate_monster_ev.py` with source data, commit the generated CSV, and redeploy Streamlit.")
        st.stop()

    settings = render_sidebar(raw_df)
    manual_prices = st.session_state.personal_prices
    df = apply_ui_ev_settings(raw_df, settings, manual_prices)
    filtered = filter_dataframe(df, settings)
    render_metrics(filtered, settings, manual_prices)

    tabs = st.tabs(["Best farms", "Maps", "Items", "Raw data"])
    with tabs[0]:
        render_best_farms(filtered, settings, manual_prices)
    with tabs[1]:
        render_maps(filtered)
    with tabs[2]:
        render_items(raw_df, settings, manual_prices)
    with tabs[3]:
        render_raw(filtered)

    with st.expander("How the numbers are interpreted"):
        st.markdown(f"""
- `Expected Value` is recalculated from `drops_json` using the current drop multiplier: **x{settings['multiplier']:g}**.
- Each normal drop slot is capped at **100%** before EV is summed.
- UARO pricing is **{'on' if settings['use_uaro_prices'] else 'off'}**. When on, adjusted item prices and Great Nature conversion are used before Overcharge.
- Manual prices override NPC/UARO/conversion prices and do **not** receive Overcharge.
- Poring Coin is **{'included' if settings['include_poring_coin'] else 'not included'}**. When included, it adds a fixed 5% custom drop valued at **{format_zeny(settings['poring_coin_price'])}**.
- `Map score` is `Expected Value * spawn count`; it is a simple density proxy, not zeny/hour.
- Boss-flagged monsters and monsters with MVP drops are hidden by default.
        """.strip())


if __name__ == "__main__":
    main()
