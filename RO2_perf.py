import inspect
import json
from typing import Any

import pandas as pd
import streamlit as st

import RO2 as app


def defaults(df):
    lvl = app.safe_min_max(df["level"] if "level" in df.columns else pd.Series(dtype=int))
    sp = app.safe_min_max(df["best_map_count"] if "best_map_count" in df.columns else pd.Series(dtype=int))
    return {
        "multiplier": 5.0,
        "use_overcharge": True,
        "overcharge_rate": 1.24,
        "use_uaro_prices": True,
        "include_poring_coin": False,
        "poring_coin_price": 12000,
        "name_query": "",
        "map_query": "",
        "selected_elements": [],
        "level_range": lvl,
        "spawn_range": (max(1, sp[0]) if sp[1] >= 1 else sp[0], sp[1]),
        "min_ev": 0.0,
        "include_boss": False,
        "include_mvp": False,
        "sort_by": "expected_value",
        "ascending": False,
    }


def init_state(df):
    if "personal_prices" not in st.session_state:
        st.session_state.personal_prices = app.load_price_file(app.MANUAL_PRICE_PATH)
    base = defaults(df)
    if "applied_settings" not in st.session_state:
        st.session_state.applied_settings = base
    else:
        for k, v in base.items():
            st.session_state.applied_settings.setdefault(k, v)


def price_only(s):
    return {
        "multiplier": app.as_float(s.get("multiplier"), 5.0),
        "use_overcharge": bool(s.get("use_overcharge")),
        "overcharge_rate": app.as_float(s.get("overcharge_rate"), 1.24),
        "use_uaro_prices": bool(s.get("use_uaro_prices")),
        "include_poring_coin": bool(s.get("include_poring_coin")),
        "poring_coin_price": app.as_float(s.get("poring_coin_price"), 12000.0),
    }


def pkey(s):
    return json.dumps(price_only(s), sort_keys=True, separators=(",", ":"))


def mkey(p):
    return json.dumps(app.export_price_payload(p, "cache"), sort_keys=True, separators=(",", ":"))


def loads_payload(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value


def cached_settings(df, pk):
    settings = price_only(defaults(df))
    loaded = loads_payload(pk, {})
    if isinstance(loaded, dict):
        settings.update(loaded)
    return settings


def cached_manual_prices(mk):
    loaded = loads_payload(mk, {})
    return app.normalize_manual_prices(loaded)


def apply_ev_settings(df, settings, manual_prices):
    try:
        signature = inspect.signature(app.apply_ui_ev_settings)
    except (TypeError, ValueError):
        return app.apply_ui_ev_settings(df, settings, manual_prices)

    params = list(signature.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if has_varargs or len(positional) >= 3:
        return app.apply_ui_ev_settings(df, settings, manual_prices)
    return app.apply_ui_ev_settings(df, settings)


@st.cache_data(show_spinner=False)
def cached_ev(df, pk, mk):
    return apply_ev_settings(df, cached_settings(df, pk), cached_manual_prices(mk))


def clamp_range(v: Any, fallback, mn, mx):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return fallback
    a = min(max(app.as_int(v[0], fallback[0]), mn), mx)
    b = min(max(app.as_int(v[1], fallback[1]), mn), mx)
    return (min(a, b), max(a, b))


def sidebar(df):
    s = dict(st.session_state.applied_settings)
    st.sidebar.header("Assumptions")
    with st.sidebar.form("assumptions_form"):
        multiplier = st.number_input("Drop rate multiplier", min_value=0.0, value=app.as_float(s.get("multiplier"), 5.0), step=0.5)
        use_overcharge = st.checkbox("Apply merchant Overcharge (+24%)", value=bool(s.get("use_overcharge", True)))
        overcharge_rate = st.number_input("Overcharge multiplier", min_value=1.0, value=app.as_float(s.get("overcharge_rate"), 1.24), step=0.01, format="%.2f", disabled=not use_overcharge)
        use_uaro_prices = st.checkbox("Use UARO adjusted prices", value=bool(s.get("use_uaro_prices", True)))
        include_poring_coin = st.checkbox("Include Poring Coin drops", value=bool(s.get("include_poring_coin", False)))
        poring_coin_price = st.number_input("Poring Coin price", min_value=0, value=app.as_int(s.get("poring_coin_price"), 12000), step=500, disabled=not include_poring_coin)
        if st.form_submit_button("Apply assumptions", use_container_width=True):
            s.update({"multiplier": multiplier, "use_overcharge": use_overcharge, "overcharge_rate": overcharge_rate, "use_uaro_prices": use_uaro_prices, "include_poring_coin": include_poring_coin, "poring_coin_price": poring_coin_price})
            st.session_state.applied_settings = s
    st.sidebar.caption("Changes above are batched until Apply assumptions is pressed.")

    st.sidebar.header("Farm filters")
    emap = {}
    if "element" in df.columns and not df.empty:
        pairs = df[["element", "element_display"]].drop_duplicates().sort_values(["element_display", "element"])
        for _, r in pairs.iterrows():
            raw = str(r.get("element") or "").strip()
            lab = str(r.get("element_display") or raw).strip()
            if raw:
                emap.setdefault(lab, []).append(raw)
    lvl = app.safe_min_max(df["level"] if "level" in df.columns else pd.Series(dtype=int))
    sp = app.safe_min_max(df["best_map_count"] if "best_map_count" in df.columns else pd.Series(dtype=int))
    sdef = (max(1, sp[0]) if sp[1] >= 1 else sp[0], sp[1])
    choices = [c for c in ["expected_value", "map_value_score", "best_map_count", "level", "hp", "name"] if c in df.columns or c == "map_value_score"]
    selected_elements = [x for x in s.get("selected_elements", []) if x in emap]
    sort_by_default = s.get("sort_by") if s.get("sort_by") in choices else (choices[0] if choices else None)
    with st.sidebar.form("filters_form"):
        name_query = st.text_input("Monster name contains", value=str(s.get("name_query", "")))
        map_query = st.text_input("Map contains", value=str(s.get("map_query", "")))
        selected_elements = st.multiselect("Element", list(emap.keys()), default=selected_elements)
        level_range = st.slider("Level range", lvl[0], lvl[1], clamp_range(s.get("level_range"), lvl, lvl[0], lvl[1]))
        spawn_range = st.slider("Best-map spawn count", sp[0], sp[1], clamp_range(s.get("spawn_range"), sdef, sp[0], sp[1]))
        min_ev = st.number_input("Minimum EV", min_value=0.0, value=app.as_float(s.get("min_ev"), 0.0), step=1.0)
        include_boss = st.checkbox("Include boss-flagged monsters", value=bool(s.get("include_boss", False))) if "is_boss" in df.columns else True
        include_mvp = st.checkbox("Include monsters with MVP drops", value=bool(s.get("include_mvp", False))) if "has_mvp_drops" in df.columns else True
        sort_by = st.selectbox("Sort by", choices, index=choices.index(sort_by_default) if sort_by_default in choices else 0, format_func=lambda c: app.SORT_LABELS.get(c, c)) if choices else None
        ascending = st.checkbox("Ascending sort", value=bool(s.get("ascending", False)))
        if st.form_submit_button("Apply filters", use_container_width=True):
            s.update({"name_query": name_query, "map_query": map_query, "selected_elements": selected_elements, "level_range": level_range, "spawn_range": spawn_range, "min_ev": min_ev, "include_boss": include_boss, "include_mvp": include_mvp, "sort_by": sort_by, "ascending": ascending})
            st.session_state.applied_settings = s
    st.sidebar.caption("Filters are batched until Apply filters is pressed.")
    return dict(st.session_state.applied_settings, element_map=emap)


def main():
    app.apply_layout_css()
    st.title("Mob Value Planner")
    st.caption("Monster value explorer, farming comparison tool, and price-table sandbox.")
    try:
        raw = app.load_data(app.CSV_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    if raw.empty:
        st.warning("`monster_ev.csv` is empty or missing usable rows. Run `python generate_monster_ev.py` with source data, commit the generated CSV, and redeploy Streamlit.")
        st.stop()

    init_state(raw)
    settings = sidebar(raw)
    prices = st.session_state.personal_prices
    df = cached_ev(raw, pkey(settings), mkey(prices))
    filtered = app.filter_dataframe(df, settings)
    app.render_metrics(filtered, settings, prices)

    tabs = st.tabs(["Best farms", "Maps", "Items", "Raw data"])
    with tabs[0]:
        app.render_best_farms(filtered, settings, prices)
    with tabs[1]:
        app.render_maps(filtered)
    with tabs[2]:
        app.render_items(raw, settings, prices)
    with tabs[3]:
        app.render_raw(filtered)

    with st.expander("How the numbers are interpreted"):
        st.markdown(f"""
- `Expected Value` is recalculated from `drops_json` using the applied drop multiplier: **x{settings['multiplier']:g}**.
- Each normal drop slot is capped at **100%** before EV is summed.
- UARO pricing is **{'on' if settings['use_uaro_prices'] else 'off'}**. When on, adjusted item prices and Great Nature conversion are used before Overcharge.
- Manual prices override NPC/UARO/conversion prices and do **not** receive Overcharge.
- Poring Coin is **{'included' if settings['include_poring_coin'] else 'not included'}**. When included, it adds a fixed 5% custom drop valued at **{app.format_zeny(settings['poring_coin_price'])}**.
""")