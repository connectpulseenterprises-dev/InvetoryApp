# inventory_app_full.py
"""
Inventory & Kits Manager — Streamlit + Supabase/Postgres (full)
- Requires: streamlit, pandas, sqlalchemy, psycopg2-binary, python-dateutil
- Put DATABASE_URL (Session Pooler URL) in Streamlit secrets (URL-encode password)
Run:
streamlit run inventory_app_full.py
"""

import io
import time
import random
import logging
from datetime import datetime, date
from typing import List, Dict

import streamlit as st
import pandas as pd
from dateutil import parser as dateparser
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# ---------------- Page config ----------------
st.set_page_config(page_title="Inventory & Kits Manager", page_icon="📦", layout="wide")
logger = logging.getLogger(__name__)

# ---------------- Secrets & DB engine ----------------
if "DATABASE_URL" not in st.secrets:
    st.error("DATABASE_URL missing in Streamlit secrets. Add Supabase Session Pooler URL (URL-encode password).")
    st.stop()

def ensure_sslmode(url: str) -> str:
    if "sslmode" in url.lower():
        return url
    return url + ("&sslmode=require" if "?" in url else "?sslmode=require")

raw_db_url = st.secrets["DATABASE_URL"]
DB_URL = ensure_sslmode(raw_db_url)

def make_engine_with_pool(db_url: str, max_retries: int = 3):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            eng = create_engine(db_url, pool_size=3, max_overflow=6, pool_pre_ping=True, future=True)
            # quick smoke test
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("DB connected on attempt %s", attempt)
            return eng
        except Exception as e:
            last_exc = e
            logger.exception("DB connect attempt %s failed: %s", attempt, e)
            time.sleep((2 ** (attempt - 1)) + random.random())
    raise last_exc

try:
    engine = make_engine_with_pool(DB_URL, max_retries=4)
except Exception as e:
    logger.exception("Failed to connect to DB")
    st.error("Failed to connect to database. Check DATABASE_URL in st.secrets and Supabase project. See logs.")
    st.stop()

# ---------------- Default templates ----------------
DEFAULTS = {
    "purchases": pd.DataFrame(columns=[
        "Date", "Material ID", "Material Name", "Vendor", "Packs", "Qty Per Pack",
        "Cost Per Pack", "Pieces", "Price Per Piece"
    ]),
    "kits_bom": pd.DataFrame(columns=[
        "Kit ID", "Kit Name", "Material ID", "Material Name", "Qty Per Kit", "Unit Price Ref"
    ]),
    "created_kits": pd.DataFrame(columns=["Date", "Kit ID", "Kit Name", "Qty", "Notes"]),
    "sold_kits": pd.DataFrame(columns=[
        "Date", "Order ID", "Kit ID", "Kit Name", "Qty", "Platform", "Price", "Fees",
        "Amount Received", "Cost Price", "Profit", "Notes"
    ]),
    "defective": pd.DataFrame(columns=["Date", "Material ID", "Qty", "Reason"]),
    "restock": pd.DataFrame(columns=["Material ID", "Desired Level", "Notes"]),
    "master": pd.DataFrame(columns=[
        "ID", "Name", "Vendor", "Available Pieces", "Current Price Per Piece",
        "Total Value", "Total Purchased", "Total Consumed"
    ]),
    "raw_uploads": pd.DataFrame(columns=["uploaded_at", "filename", "platform", "storage_path", "rows", "notes"])
}

# ---------------- DB init ----------------
def init_db():
    ddl = """
    CREATE TABLE IF NOT EXISTS purchases (
        id SERIAL PRIMARY KEY,
        date DATE,
        material_id TEXT,
        material_name TEXT,
        vendor TEXT,
        packs INTEGER,
        qty_per_pack INTEGER,
        cost_per_pack NUMERIC,
        pieces NUMERIC,
        price_per_piece NUMERIC
    );
    CREATE TABLE IF NOT EXISTS kits_bom (
        id SERIAL PRIMARY KEY,
        kit_id TEXT,
        kit_name TEXT,
        material_id TEXT,
        material_name TEXT,
        qty_per_kit NUMERIC,
        unit_price_ref NUMERIC
    );
    CREATE TABLE IF NOT EXISTS created_kits (
        id SERIAL PRIMARY KEY,
        date DATE,
        kit_id TEXT,
        kit_name TEXT,
        qty INTEGER,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS sold_kits (
        id SERIAL PRIMARY KEY,
        date DATE,
        order_id TEXT,
        kit_id TEXT,
        kit_name TEXT,
        qty INTEGER,
        platform TEXT,
        price NUMERIC,
        fees NUMERIC,
        amount_received NUMERIC,
        cost_price NUMERIC,
        profit NUMERIC,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS defective (
        id SERIAL PRIMARY KEY,
        date DATE,
        material_id TEXT,
        qty NUMERIC,
        reason TEXT
    );
    CREATE TABLE IF NOT EXISTS restock (
        id SERIAL PRIMARY KEY,
        material_id TEXT,
        desired_level INTEGER,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS master (
        id TEXT PRIMARY KEY,
        name TEXT,
        vendor TEXT,
        available_pieces NUMERIC,
        current_price_per_piece NUMERIC,
        total_value NUMERIC,
        total_purchased NUMERIC,
        total_consumed NUMERIC
    );
    CREATE TABLE IF NOT EXISTS raw_uploads (
        id SERIAL PRIMARY KEY,
        uploaded_at TIMESTAMP,
        filename TEXT,
        platform TEXT,
        storage_path TEXT,
        rows INTEGER,
        notes TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

try:
    init_db()
except Exception as e:
    logger.exception("init_db failed")
    st.error("Database initialization failed. See logs.")
    st.stop()

# ---------------- Cached reads / writes ----------------
@st.cache_data(ttl=60, show_spinner=False)
def read_table(name: str) -> pd.DataFrame:
    try:
        q = f"SELECT * FROM {name} ORDER BY id NULLS LAST"
        df = pd.read_sql(q, engine)
        if df.empty:
            return DEFAULTS.get(name, pd.DataFrame()).copy()
        return df
    except Exception:
        try:
            df = pd.read_sql(f"SELECT * FROM {name}", engine)
            if df.empty:
                return DEFAULTS.get(name, pd.DataFrame()).copy()
            return df
        except Exception as e:
            logger.exception("read_table failed for %s: %s", name, e)
            return DEFAULTS.get(name, pd.DataFrame()).copy()

def _safe_float(x):
    try:
        if x in (None, ""):
            return 0.0
        return float(x)
    except Exception:
        return 0.0

# ---------------- Normalize purchases (auto compute pieces & price_per_piece) ----------------
def normalize_purchases_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    w = df.copy()
    # map common header variants
    col_map = {}
    for c in w.columns:
        lc = c.strip().lower()
        if lc in ("date", "purchase_date", "order_date"):
            col_map[c] = "date"
        elif lc in ("material id", "material_id", "materialid", "id"):
            col_map[c] = "material_id"
        elif lc in ("material name", "material_name", "name"):
            col_map[c] = "material_name"
        elif lc in ("vendor", "supplier"):
            col_map[c] = "vendor"
        elif lc in ("packs", "pack_count"):
            col_map[c] = "packs"
        elif lc in ("qty per pack", "qty_per_pack", "qtyperpack", "qty_per"):
            col_map[c] = "qty_per_pack"
        elif lc in ("cost per pack", "cost_per_pack", "costperpack", "cost"):
            col_map[c] = "cost_per_pack"
        elif lc in ("pieces", "pcs", "piece_count"):
            col_map[c] = "pieces"
        elif lc in ("price per piece", "price_per_piece", "ppp"):
            col_map[c] = "price_per_piece"
    w = w.rename(columns=col_map)
    # ensure numeric columns exist
    for n in ["packs", "qty_per_pack", "cost_per_pack", "pieces", "price_per_piece"]:
        if n not in w.columns:
            w[n] = None
    # coerce types
    w["packs"] = pd.to_numeric(w["packs"], errors="coerce").fillna(0).astype(int)
    w["qty_per_pack"] = pd.to_numeric(w["qty_per_pack"], errors="coerce").fillna(0).astype(int)
    w["cost_per_pack"] = pd.to_numeric(w["cost_per_pack"], errors="coerce").fillna(0.0)
    # compute pieces
    def compute_pieces(row):
        pieces = row.get("pieces")
        try:
            if pieces is None or (pd.notna(pieces) and float(pieces) == 0):
                return int(row["packs"] * row["qty_per_pack"])
            return int(float(pieces))
        except Exception:
            return int(row["packs"] * row["qty_per_pack"])
    w["pieces"] = w.apply(compute_pieces, axis=1)
    # compute price per piece
    def compute_ppp(row):
        ppp = row.get("price_per_piece")
        try:
            if ppp is None or (pd.notna(ppp) and float(ppp) == 0):
                if row["qty_per_pack"] > 0:
                    return round(float(row["cost_per_pack"]) / float(row["packs"] * row["qty_per_pack"]), 6)
                return 0.0
            return float(ppp)
        except Exception:
            if row["qty_per_pack"] > 0:
                return round(float(row["cost_per_pack"]) / float(row["packs"] * row["qty_per_pack"]), 6)
            return 0.0
    w["price_per_piece"] = w.apply(compute_ppp, axis=1)
    # normalize date strings
    if "date" in w.columns:
        try:
            w["date"] = pd.to_datetime(w["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    final_cols = ["date", "material_id", "material_name", "vendor", "packs", "qty_per_pack", "cost_per_pack", "pieces", "price_per_piece"]
    for c in final_cols:
        if c not in w.columns:
            w[c] = None
    return w[final_cols]

# ---------------- Write helpers ----------------
def write_rows(name: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    target_df = df.copy()
    if name == "purchases":
        target_df = normalize_purchases_df(target_df)
    try:
        # Use to_sql append
        target_df.to_sql(name, engine, if_exists="append", index=False, method="multi", chunksize=250)
    except Exception as e:
        logger.exception("to_sql append failed, fallback single inserts: %s", e)
        with engine.begin() as conn:
            for r in target_df.fillna("").to_dict(orient="records"):
                keys = ", ".join(r.keys())
                vals = ", ".join([f":{k}" for k in r.keys()])
                stmt = text(f"INSERT INTO {name} ({keys}) VALUES ({vals})")
                conn.execute(stmt, r)
    try:
        read_table.clear()
    except Exception:
        pass

def overwrite_table(name: str, df: pd.DataFrame):
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {name}"))
    if df is not None and not df.empty:
        to_insert = df.copy()
        if name == "purchases":
            to_insert = normalize_purchases_df(to_insert)
        write_rows(name, to_insert)
    try:
        read_table.clear()
    except Exception:
        pass

# ---------------- Snapshot logic ----------------
def build_material_events(purchases: pd.DataFrame,
                          created_kits: pd.DataFrame,
                          defective: pd.DataFrame,
                          bom: pd.DataFrame) -> dict:
    events_by_mat = {}
    # purchases
    if purchases is not None and not purchases.empty:
        for r in purchases.to_dict(orient="records"):
            try:
                d = pd.to_datetime(r.get("date") or r.get("Date"), errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            mid = str(r.get("material_id") or r.get("Material ID") or "")
            pieces = _safe_float(r.get("pieces") or r.get("Pieces") or 0)
            if pieces == 0:
                packs = _safe_float(r.get("packs") or r.get("Packs") or 0)
                qpp = _safe_float(r.get("qty_per_pack") or r.get("Qty Per Pack") or 0)
                pieces = packs * qpp
            ppp = _safe_float(r.get("price_per_piece") or r.get("Price Per Piece") or r.get("price") or 0)
            ev = {"date": d, "delta": pieces, "price_per_piece": ppp, "type": "purchase", "vendor": r.get("vendor") or r.get("Vendor", ""), "name": r.get("material_name") or r.get("Material Name", "")}
            events_by_mat.setdefault(mid, []).append(ev)
    # created kits -> expand using BOM
    if bom is not None and not bom.empty and created_kits is not None and not created_kits.empty:
        bom_map = {}
        for br in bom.to_dict(orient="records"):
            k = str(br.get("kit_id") or br.get("Kit ID") or "")
            bom_map.setdefault(k, []).append(br)
        for cr in created_kits.to_dict(orient="records"):
            try:
                d = pd.to_datetime(cr.get("date") or cr.get("Date"), errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            kit_id = str(cr.get("kit_id") or cr.get("Kit ID") or "")
            qty_kits = _safe_float(cr.get("qty") or cr.get("Qty") or 0)
            if kit_id in bom_map:
                for br in bom_map[kit_id]:
                    mid = str(br.get("material_id") or br.get("Material ID") or "")
                    qty_per_kit = _safe_float(br.get("qty_per_kit") or br.get("Qty Per Kit") or 0)
                    total_consume = qty_per_kit * qty_kits
                    ev = {"date": d, "delta": -total_consume, "price_per_piece": None, "type": "consume", "kit_id": kit_id}
                    events_by_mat.setdefault(mid, []).append(ev)
    # defective
    if defective is not None and not defective.empty:
        for dr in defective.to_dict(orient="records"):
            try:
                d = pd.to_datetime(dr.get("date") or dr.get("Date"), errors="coerce")
            except Exception:
                d = pd.to_datetime(datetime.today())
            mid = str(dr.get("material_id") or dr.get("Material ID") or "")
            qty = _safe_float(dr.get("qty") or dr.get("Qty") or 0)
            ev = {"date": d, "delta": -qty, "price_per_piece": None, "type": "defect"}
            events_by_mat.setdefault(mid, []).append(ev)
    # sort
    for mid, evs in events_by_mat.items():
        evs_sorted = sorted(evs, key=lambda x: pd.to_datetime(x.get("date")))
        events_by_mat[mid] = evs_sorted
    return events_by_mat

def compute_snapshot_from_events(events_by_mat: dict) -> pd.DataFrame:
    rows = []
    for mid, events in events_by_mat.items():
        total_pieces = 0.0
        total_cost = 0.0
        total_purchased = 0.0
        total_consumed = 0.0
        name = ""
        vendor = ""
        for ev in events:
            if ev.get("type") == "purchase":
                pieces = _safe_float(ev.get("delta") or 0)
                price = _safe_float(ev.get("price_per_piece") or 0)
                total_pieces += pieces
                total_cost += pieces * price
                total_purchased += pieces
                if ev.get("name"):
                    name = ev.get("name")
                if ev.get("vendor"):
                    vendor = ev.get("vendor")
            else:
                consume_pieces = -_safe_float(ev.get("delta") or 0)
                if total_pieces > 0:
                    avg_cost = (total_cost / total_pieces) if total_pieces != 0 else 0.0
                    remove_cost = min(consume_pieces, total_pieces) * avg_cost
                    total_cost -= remove_cost
                    total_pieces -= min(consume_pieces, total_pieces)
                else:
                    total_pieces -= consume_pieces
                total_consumed += consume_pieces
        available_pieces = round(total_pieces, 6)
        current_price = round((total_cost / total_pieces) if total_pieces > 0 else 0.0, 4)
        total_value = round(available_pieces * current_price, 2)
        rows.append({
            "ID": mid,
            "Name": name,
            "Vendor": vendor,
            "Available Pieces": available_pieces,
            "Current Price Per Piece": current_price,
            "Total Value": total_value,
            "Total Purchased": round(total_purchased, 6),
            "Total Consumed": round(total_consumed, 6)
        })
    if not rows:
        return DEFAULTS["master"].copy()
    df = pd.DataFrame(rows)
    df = df.sort_values("ID").reset_index(drop=True)
    return df

def recompute_master_snapshot_and_save():
    purchases = read_table("purchases")
    created = read_table("created_kits")
    defective = read_table("defective")
    bom = read_table("kits_bom")
    events = build_material_events(purchases, created, defective, bom)
    snapshot = compute_snapshot_from_events(events)
    if not snapshot.empty:
        df = snapshot.rename(columns={"ID": "id", "Name": "name", "Vendor": "vendor", "Available Pieces": "available_pieces", "Current Price Per Piece": "current_price_per_piece", "Total Value": "total_value", "Total Purchased": "total_purchased", "Total Consumed": "total_consumed"})
        overwrite_table("master", df)
    else:
        overwrite_table("master", pd.DataFrame())
    return snapshot

# ---------------- Parsers (marketplace CSV) ----------------
def normalize_header(h: str) -> str:
    return str(h or "").strip().lower().replace(" ", "_")

def parse_amazon(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame([])
    cols_map = {c: normalize_header(c) for c in df_raw.columns}
    df = df_raw.rename(columns=cols_map)
    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None
    date_k = pick(["purchase_date","order_date","date","sale_date"])
    sku_k = pick(["sku","seller_sku","product_sku","asin","item_sku"])
    title_k = pick(["title","product_title","item_name","description"])
    qty_k = pick(["quantity","qty_shipped","qty","quantity_ordered"])
    price_k = pick(["item_price","price","item_subtotal","sale_price","unit_price"])
    fees_k = pick(["fees","fee","amazon_fees","commission"])
    amount_k = pick(["amount","total","amount_received","net_amount"])
    out = []
    for r in df.to_dict(orient="records"):
        try:
            d = r.get(date_k,"")
            dstr = pd.to_datetime(d).strftime("%Y-%m-%d") if d else ""
        except Exception:
            dstr = ""
        qty = int(_safe_float(r.get(qty_k,1)))
        price = _safe_float(r.get(price_k,0))
        fees = _safe_float(r.get(fees_k,0))
        amount = _safe_float(r.get(amount_k, price*qty))
        out.append({
            "Date": dstr,
            "Order ID": r.get(pick(["order_id","amazon_order_id","order_number"]),""),
            "Kit ID": r.get(sku_k,""),
            "Kit Name": r.get(title_k,""),
            "Qty": qty,
            "Platform": "Amazon",
            "Price": price,
            "Fees": fees,
            "Amount Received": amount,
            "Notes": ""
        })
    return pd.DataFrame(out)

def parse_flipkart(df_raw: pd.DataFrame) -> pd.DataFrame:
    parsed = parse_amazon(df_raw)
    if not parsed.empty:
        parsed["Platform"] = "Flipkart"
    return parsed

def parse_meesho(df_raw: pd.DataFrame) -> pd.DataFrame:
    parsed = parse_amazon(df_raw)
    if not parsed.empty:
        parsed["Platform"] = "Meesho"
    return parsed

# ---------------- Admin helpers: PK & delete ----------------
def get_primary_key_column(table_name: str) -> str:
    try:
        insp = inspect(engine)
        pk = insp.get_pk_constraint(table_name).get("constrained_columns", [])
        if pk:
            return pk[0]
    except Exception:
        pass
    return "id"

def delete_rows_by_pk(table_name: str, pk_values: List):
    if not pk_values:
        return 0
    pk_col = get_primary_key_column(table_name)
    params = {f"v{i}": v for i, v in enumerate(pk_values)}
    placeholders = ", ".join([f":v{i}" for i in range(len(pk_values))])
    sql = text(f"DELETE FROM {table_name} WHERE {pk_col} IN ({placeholders})")
    with engine.begin() as conn:
        conn.execute(sql, params)
    try:
        read_table.clear()
    except Exception:
        pass
    return len(pk_values)

# ---------------- UI: Navigation ----------------
MAIN_SECTIONS = ["📊 Dashboards", "📦 Inventory Management", "🧩 Kits Management", "🧾 Sales", "⬇️ Data"]
main_section = st.sidebar.selectbox("Section", MAIN_SECTIONS)

# ---------------- Dashboards ----------------
if main_section == "📊 Dashboards":
    dash_choice = st.sidebar.radio("Choose Dashboard", ["📦 Inventory Dashboard", "💰 Sales Dashboard"])
    if dash_choice == "📦 Inventory Dashboard":
        st.title("📦 Inventory Dashboard")
        master_snapshot = read_table("master")
        total_materials = len(master_snapshot) if not master_snapshot.empty else 0
        total_pieces = int(master_snapshot["available_pieces"].sum()) if not master_snapshot.empty and "available_pieces" in master_snapshot.columns else (int(master_snapshot["Available Pieces"].sum()) if not master_snapshot.empty and "Available Pieces" in master_snapshot.columns else 0)
        inventory_value = float(master_snapshot["total_value"].sum()) if not master_snapshot.empty and "total_value" in master_snapshot.columns else (float(master_snapshot["Total Value"].sum()) if not master_snapshot.empty and "Total Value" in master_snapshot.columns else 0.0)
        try:
            bom_df = read_table("kits_bom")
            total_kit_types = bom_df["kit_id"].nunique() if not bom_df.empty and "kit_id" in bom_df.columns else bom_df["Kit ID"].nunique() if not bom_df.empty and "Kit ID" in bom_df.columns else 0
        except Exception:
            total_kit_types = 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Materials", f"{total_materials}")
        c2.metric("Available Pieces", f"{total_pieces}")
        c3.metric("Inventory Value", f"₹{inventory_value:,.2f}")
        c4.metric("Kit Types", f"{total_kit_types}")
        st.markdown("---")
        st.subheader("Inventory Snapshot")
        with st.expander("View full inventory snapshot"):
            st.dataframe(master_snapshot)
    else:
        st.title("💰 Sales Dashboard")
        sold_df = read_table("sold_kits")
        if sold_df.empty:
            st.info("No sales recorded.")
        else:
            sold_local = sold_df.copy()
            sold_local["Date_dt"] = pd.to_datetime(sold_local["date"] if "date" in sold_local.columns else sold_local["Date"], errors="coerce")
            st.markdown("### 🔍 Filters")
            col1, col2, col3 = st.columns([2,2,2])
            min_date = sold_local["Date_dt"].min().date() if not sold_local["Date_dt"].isna().all() else date.today()
            max_date = sold_local["Date_dt"].max().date() if not sold_local["Date_dt"].isna().all() else date.today()
            with col1:
                dr = st.date_input("Date Range", value=(min_date, max_date))
            platforms = sorted(sold_local["platform"].dropna().unique().tolist()) if "platform" in sold_local.columns else sorted(sold_local["Platform"].dropna().unique().tolist())
            with col2:
                pf = st.multiselect("Platforms", options=platforms, default=platforms)
            kits = sorted(sold_local["kit_name"].dropna().unique().tolist()) if "kit_name" in sold_local.columns else sorted(sold_local["Kit Name"].dropna().unique().tolist())
            with col3:
                kit_filter = st.multiselect("Kits", options=kits, default=kits)
            mask = pd.Series(True, index=sold_local.index)
            if isinstance(dr, tuple) and len(dr) == 2:
                mask &= (sold_local["Date_dt"].dt.date >= dr[0]) & (sold_local["Date_dt"].dt.date <= dr[1])
            if pf:
                if "platform" in sold_local.columns:
                    mask &= sold_local["platform"].isin(pf)
                else:
                    mask &= sold_local["Platform"].isin(pf)
            if kit_filter:
                if "kit_name" in sold_local.columns:
                    mask &= sold_local["kit_name"].isin(kit_filter)
                else:
                    mask &= sold_local["Kit Name"].isin(kit_filter)
            sold_filtered = sold_local.loc[mask]
            ensure_cols = ["amount_received","cost_price","profit","qty","Amount Received","Cost Price","Profit","Qty"]
            ensure_numeric(sold_filtered, ensure_cols)
            rev = float(sold_filtered["amount_received"].sum()) if "amount_received" in sold_filtered.columns else float(sold_filtered["Amount Received"].sum()) if "Amount Received" in sold_filtered.columns else 0.0
            cost = float(sold_filtered["cost_price"].sum()) if "cost_price" in sold_filtered.columns else float(sold_filtered["Cost Price"].sum()) if "Cost Price" in sold_filtered.columns else 0.0
            prof = float(sold_filtered["profit"].sum()) if "profit" in sold_filtered.columns else float(sold_filtered["Profit"].sum()) if "Profit" in sold_filtered.columns else 0.0
            qty = int(sold_filtered["qty"].sum()) if "qty" in sold_filtered.columns else int(sold_filtered["Qty"].sum()) if "Qty" in sold_filtered.columns else 0
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Revenue", f"₹{rev:,.2f}")
            d2.metric("Cost", f"₹{cost:,.2f}")
            d3.metric("Profit", f"₹{prof:,.2f}")
            d4.metric("Units Sold", f"{qty}")
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📊 Charts", "📦 Kits", "📋 Raw Data"])
            with tab1:
                st.subheader("Profit by Platform")
                try:
                    prof_pf = sold_filtered.groupby("platform", as_index=False)["profit"].sum().sort_values("profit", ascending=False) if "platform" in sold_filtered.columns else sold_filtered.groupby("Platform", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
                    if not prof_pf.empty:
                        st.bar_chart(prof_pf.set_index(prof_pf.columns[0]))
                except Exception:
                    st.write("Not enough data for charts.")
                st.subheader("Profit Over Time (Weekly)")
                try:
                    pt = sold_filtered.groupby(pd.Grouper(key="Date_dt", freq="W"))["profit"].sum().fillna(0)
                    st.line_chart(pt)
                except Exception:
                    st.write("Not enough data for time chart.")
            with tab2:
                st.subheader("Top-selling Kits (by Qty)")
                try:
                    top_k = sold_filtered.groupby("kit_name", as_index=False)["qty"].sum().sort_values("qty", ascending=False).head(20) if "kit_name" in sold_filtered.columns else sold_filtered.groupby("Kit Name", as_index=False)["Qty"].sum().sort_values("Qty", ascending=False).head(20)
                    st.table(top_k)
                except Exception:
                    st.write("No kit data.")
            with tab3:
                st.subheader("Filtered Sales Table")
                st.dataframe(sold_filtered.sort_values("Date_dt", ascending=False))

# ---------------- Inventory Management ----------------
elif main_section == "📦 Inventory Management":
    st.title("📦 Inventory Management")
    sub = st.sidebar.radio("Inventory", ["Master Inventory (Purchases)", "Restock Planner", "Defective Items", "Purchase History"])
    if sub == "Master Inventory (Purchases)":
        st.subheader("Add Purchase (new or existing material)")
        purchases_df = read_table("purchases")
        master_snapshot = read_table("master")
        mat_options = [""] + sorted(list(dict.fromkeys((purchases_df.get("material_id", pd.Series()).astype(str).fillna("").tolist() + master_snapshot.get("id", pd.Series()).astype(str).fillna("").tolist()))))
        selected = st.selectbox("Select existing Material ID (leave blank to add new)", mat_options)
        if selected:
            prev = purchases_df[purchases_df.get("material_id", "") == selected]
            if not prev.empty:
                last = prev.sort_values("date").iloc[-1]
                default_name = last.get("material_name","") or last.get("Material Name","")
                default_vendor = last.get("vendor","") or last.get("Vendor","")
            else:
                row = master_snapshot[master_snapshot.get("id","") == selected]
                default_name = row["name"].iat[0] if not row.empty and "name" in row.columns else ""
                default_vendor = row["vendor"].iat[0] if not row.empty and "vendor" in row.columns else ""
            col1, col2 = st.columns(2)
            with col1:
                m_name = st.text_input("Material Name", value=default_name)
                vendor = st.text_input("Vendor", value=default_vendor)
                date_p = st.date_input("Purchase Date", value=date.today())
            with col2:
                packs = st.number_input("Packs bought", min_value=0, step=1, value=0)
                qty_per_pack = st.number_input("Qty per Pack", min_value=0, step=1, value=0)
                cost_per_pack = st.number_input("Cost per Pack (₹)", min_value=0.0, step=0.01, value=0.0)
            pieces = int(packs * qty_per_pack)
            price_per_piece = float((cost_per_pack / (packs * qty_per_pack)) if qty_per_pack > 0 else 0.0)
            st.markdown(f"**Pieces (computed):** {pieces} — **Price per piece:** ₹{price_per_piece:.4f}")
            if st.button("Add Purchase (update existing material)"):
                if not selected:
                    st.error("Select an existing ID or enter a new one.")
                else:
                    row = {
                        "date": date_p.strftime("%Y-%m-%d"),
                        "material_id": selected,
                        "material_name": m_name.strip(),
                        "vendor": vendor.strip(),
                        "packs": int(packs),
                        "qty_per_pack": int(qty_per_pack),
                        "cost_per_pack": float(cost_per_pack),
                        "pieces": pieces,
                        "price_per_piece": price_per_piece
                    }
                    write_rows("purchases", pd.DataFrame([row]))
                    recompute_master_snapshot_and_save()
                    st.success(f"Purchase added for {selected}: +{pieces} pcs at ₹{price_per_piece:.4f}/pc")
        else:
            st.markdown("**Add brand new material purchase**")
            col1, col2 = st.columns(2)
            with col1:
                new_id = st.text_input("New Material ID (unique)")
                m_name = st.text_input("Material Name")
                vendor = st.text_input("Vendor")
                date_p = st.date_input("Purchase Date", value=date.today())
            with col2:
                packs = st.number_input("Packs bought", min_value=0, step=1, value=0)
                qty_per_pack = st.number_input("Qty per Pack", min_value=0, step=1, value=0)
                cost_per_pack = st.number_input("Cost per Pack (₹)", min_value=0.0, step=0.01, value=0.0)
            pieces = int(packs * qty_per_pack)
            price_per_piece = float((cost_per_pack / (packs * qty_per_pack)) if qty_per_pack > 0 else 0.0)
            st.markdown(f"**Pieces (computed):** {pieces} — **Price per piece:** ₹{price_per_piece:.4f}")
            if st.button("Add Purchase (new material)"):
                if not new_id:
                    st.error("Material ID required.")
                else:
                    purchases_df = read_table("purchases")
                    master_snapshot = read_table("master")
                    exists = ((purchases_df.get("material_id", pd.Series()) == new_id).any()) or ((master_snapshot.get("id", pd.Series()) == new_id).any())
                    if exists:
                        st.error("Material ID already exists. Choose it from dropdown to update instead.")
                    else:
                        row = {
                            "date": date_p.strftime("%Y-%m-%d"),
                            "material_id": new_id.strip(),
                            "material_name": m_name.strip(),
                            "vendor": vendor.strip(),
                            "packs": int(packs),
                            "qty_per_pack": int(qty_per_pack),
                            "cost_per_pack": float(cost_per_pack),
                            "pieces": pieces,
                            "price_per_piece": price_per_piece
                        }
                        write_rows("purchases", pd.DataFrame([row]))
                        recompute_master_snapshot_and_save()
                        st.success(f"Added new material {new_id} with {pieces} pcs at ₹{price_per_piece:.4f}/pc")
        st.markdown("---")
        st.subheader("Inventory Snapshot (derived)")
        st.dataframe(read_table("master"))
    elif sub == "Restock Planner":
        st.subheader("Restock Planner")
        master_snapshot = read_table("master")
        threshold = st.number_input("Show items with stock <= (pieces)", min_value=1, value=5)
        low = master_snapshot[pd.to_numeric(master_snapshot.get("available_pieces", master_snapshot.get("Available Pieces", pd.Series())), errors="coerce").fillna(0) <= threshold] if not master_snapshot.empty else pd.DataFrame([])
        st.dataframe(low[["id","name","available_pieces","current_price_per_piece"]] if not low.empty else pd.DataFrame([{"Status":"✅ All good"}]))
        with st.form("restock_form"):
            rid = st.selectbox("Material ID", [""] + sorted(master_snapshot.get("id", pd.Series()).astype(str).tolist()) if not master_snapshot.empty else [""])
            desired = st.number_input("Desired Level (pieces)", min_value=0, step=1, value=0)
            notes = st.text_input("Notes")
            submitted = st.form_submit_button("Add / Update Restock")
            if submitted:
                if not rid:
                    st.error("Choose material")
                else:
                    write_rows("restock", pd.DataFrame([{"material_id": rid, "desired_level": int(desired), "notes": notes}]))
                    st.success("Restock plan saved.")
        st.subheader("Restock Records")
        st.dataframe(read_table("restock"))
    elif sub == "Defective Items":
        st.subheader("Defective / Damaged Items (decreases available stock)")
        master_snapshot = read_table("master")
        mid = st.selectbox("Material ID", [""] + sorted(master_snapshot.get("id", pd.Series()).astype(str).tolist()) if not master_snapshot.empty else [""])
        qty = st.number_input("Qty (pieces)", min_value=1, step=1, value=1)
        reason = st.text_input("Reason")
        date_d = st.date_input("Date", value=date.today())
        if st.button("Log Defect & Deduct Stock"):
            if not mid:
                st.error("Choose material.")
            else:
                row = {"date": date_d.strftime("%Y-%m-%d"), "material_id": mid, "qty": int(qty), "reason": reason}
                write_rows("defective", pd.DataFrame([row]))
                recompute_master_snapshot_and_save()
                st.success("Logged defective item and updated snapshot.")
        st.subheader("Defective Log")
        st.dataframe(read_table("defective"))
    elif sub == "Purchase History":
        st.subheader("Purchase Ledger")
        purchases_df = read_table("purchases")
        filter_mat = st.selectbox("Filter by Material ID (optional)", [""] + sorted(purchases_df.get("material_id", pd.Series()).dropna().unique().astype(str).tolist()) if not purchases_df.empty else [""])
        view = purchases_df.copy()
        if filter_mat:
            view = view[view.get("material_id","") == filter_mat]
        view = view.sort_values("date", ascending=False) if "date" in view.columns else view
        st.dataframe(view)

# ---------------- Kits Management ----------------
elif main_section == "🧩 Kits Management":
    st.title("🧩 Kits Management")
    sub = st.sidebar.radio("Kits", ["BOM (Kit components)", "Create Kits", "Kits Inventory"])
    if sub == "BOM (Kit components)":
        st.subheader("Define Kit and add BOM rows")
        kit_id_input = st.text_input("Kit ID (e.g., KIT001)")
        kit_name_input = st.text_input("Kit Name (human readable)")
        st.markdown("**Add component to BOM**")
        col1, col2, col3 = st.columns(3)
        with col1:
            material_id = st.text_input("Material ID")
        with col2:
            qty_per = st.number_input("Qty Per Kit (pieces)", min_value=0.0, step=1.0, value=1.0)
        with col3:
            unit_override = st.number_input("Unit Price Override (₹) (optional)", min_value=0.0, step=0.01, value=0.0)
        if st.button("Add BOM Row"):
            if not kit_id_input or not kit_name_input or not material_id or qty_per <= 0:
                st.error("Provide Kit ID, Kit Name, Material, and qty > 0.")
            else:
                row = {
                    "kit_id": kit_id_input.strip(),
                    "kit_name": kit_name_input.strip(),
                    "material_id": material_id,
                    "material_name": "",
                    "qty_per_kit": float(qty_per),
                    "unit_price_ref": float(unit_override) if unit_override > 0 else None
                }
                write_rows("kits_bom", pd.DataFrame([row]))
                st.success(f"Added BOM row for {kit_id_input} ({kit_name_input}).")
        st.markdown("---")
        st.subheader("Current BOM")
        st.dataframe(read_table("kits_bom"))
    elif sub == "Create Kits":
        st.subheader("Produce / Assemble Kits (consume raw materials)")
        bom_df = read_table("kits_bom")
        kit_options = (sorted(bom_df.get("kit_id", pd.Series()).dropna().unique().tolist()) if not bom_df.empty else [])
        k_id = st.selectbox("Kit ID", [""] + kit_options)
        kit_name = ""
        if k_id:
            tmp = bom_df[bom_df.get("kit_id","") == k_id]
            if not tmp.empty:
                kit_name = tmp.iloc[0].get("kit_name","")
        qty_build = st.number_input("Qty to create", min_value=1, step=1, value=1)
        notes = st.text_input("Notes (optional)")
        if st.button("Check Feasibility"):
            ok = True
            msgs = []
            snapshot = read_table("master")
            if bom_df is None or bom_df.empty or k_id == "":
                ok = False
                msgs.append("Kit BOM not defined.")
            else:
                kit_rows = bom_df[bom_df.get("kit_id","") == k_id]
                for _, r in kit_rows.iterrows():
                    mid = r.get("material_id")
                    req_qty = float(r.get("qty_per_kit") or 0) * qty_build
                    stock_row = snapshot[snapshot.get("id","") == mid]
                    avail = float(stock_row["available_pieces"].iat[0]) if not stock_row.empty and "available_pieces" in stock_row.columns else (float(stock_row["Available Pieces"].iat[0]) if not stock_row.empty and "Available Pieces" in stock_row.columns else 0)
                    if avail < req_qty:
                        msgs.append(f"{mid} ({r.get('material_name','')}): need {req_qty}, available {avail}")
                        ok = False
            if ok:
                st.success("Enough materials available ✅")
            else:
                st.error("Insufficient materials:")
                for m in msgs:
                    st.write("- ", m)
        if st.button("Produce Kits (Consume Raw Materials)"):
            snapshot = read_table("master")
            bom_df = read_table("kits_bom")
            ok = True
            msgs = []
            if bom_df is None or bom_df.empty or k_id == "":
                ok = False
                msgs.append("BOM not defined")
            else:
                kit_rows = bom_df[bom_df.get("kit_id","") == k_id]
                for _, r in kit_rows.iterrows():
                    mid = r.get("material_id")
                    req_qty = float(r.get("qty_per_kit") or 0) * qty_build
                    stock_row = snapshot[snapshot.get("id","") == mid]
                    avail = float(stock_row["available_pieces"].iat[0]) if not stock_row.empty and "available_pieces" in stock_row.columns else (float(stock_row["Available Pieces"].iat[0]) if not stock_row.empty and "Available Pieces" in stock_row.columns else 0)
                    if avail < req_qty:
                        msgs.append(f"{mid}: need {req_qty}, available {avail}")
                        ok = False
            if not ok:
                st.error("Cannot create kits due to shortages.")
                for m in msgs:
                    st.write("- ", m)
            else:
                write_rows("created_kits", pd.DataFrame([{"date": datetime.today().strftime("%Y-%m-%d"), "kit_id": k_id, "kit_name": kit_name, "qty": int(qty_build), "notes": notes}]))
                recompute_master_snapshot_and_save()
                st.success(f"Created {qty_build} x {k_id} ({kit_name}).")
        st.subheader("Production Log (recent)")
        st.dataframe(read_table("created_kits").sort_values("date", ascending=False).head(200) if not read_table("created_kits").empty else pd.DataFrame([{"Status":"No production"}]))
    elif sub == "Kits Inventory":
        st.subheader("Kits Inventory (Produced - Sold)")
        created_df = read_table("created_kits")
        sold_df = read_table("sold_kits")
        inv_df = pd.DataFrame()
        if not created_df.empty:
            inv_df = created_df.groupby("kit_id", as_index=False)["qty"].sum().rename(columns={"qty":"Created"})
        sold_agg = pd.DataFrame()
        if not sold_df.empty:
            sold_agg = sold_df.groupby("kit_id", as_index=False)["qty"].sum().rename(columns={"qty":"Sold"})
        inv = inv_df.merge(sold_agg, on="kit_id", how="left").fillna({"Sold": 0}) if not inv_df.empty else pd.DataFrame()
        if not inv.empty:
            inv["Available"] = inv["Created"] - inv["Sold"]
            kit_names = created_df[["kit_id","kit_name"]].drop_duplicates("kit_id") if not created_df.empty else pd.DataFrame()
            bom_names = read_table("kits_bom")[["kit_id","kit_name"]].drop_duplicates("kit_id") if not read_table("kits_bom").empty else pd.DataFrame()
            names = pd.concat([kit_names, bom_names]).drop_duplicates("kit_id", keep="first") if not kit_names.empty or not bom_names.empty else pd.DataFrame()
            if not names.empty:
                inv = inv.merge(names, on="kit_id", how="left")
        st.dataframe(inv if not inv.empty else pd.DataFrame([{"Status":"No kit data"}]))

# ---------------- Sales ----------------
elif main_section == "🧾 Sales":
    st.title("🧾 Sales")
    sub = st.sidebar.radio("Sales", ["Record Sale", "Upload Marketplace CSV", "Sales Ledger"])
    if sub == "Upload Marketplace CSV":
        st.subheader("Upload marketplace CSV (Amazon / Meesho / Flipkart)")
        platform = st.selectbox("Platform", ["Amazon", "Meesho", "Flipkart", "Other"])
        uploaded = st.file_uploader("Select CSV file", type=["csv"])
        notes = st.text_input("Notes (optional)")
        if uploaded is not None and st.button("Process Upload"):
            bt = uploaded.read()
            write_rows("raw_uploads", pd.DataFrame([{"uploaded_at": datetime.utcnow(), "filename": uploaded.name, "platform": platform, "storage_path": None, "rows": 0, "notes": notes}]))
            try:
                df_raw = pd.read_csv(io.BytesIO(bt))
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")
                st.stop()
            if platform == "Amazon":
                df_norm = parse_amazon(df_raw)
            elif platform == "Flipkart":
                df_norm = parse_flipkart(df_raw)
            elif platform == "Meesho":
                df_norm = parse_meesho(df_raw)
            else:
                df_norm = parse_amazon(df_raw)
            if df_norm.empty:
                st.warning("No rows parsed from CSV.")
            else:
                # compute cost price per kit using current snapshot
                snapshot = read_table("master")
                bom = read_table("kits_bom")
                def kit_cost_from_snapshot(kit_id, qty):
                    if bom.empty or snapshot.empty or not kit_id:
                        return 0.0
                    parts = bom[bom.get("kit_id","") == kit_id] if "kit_id" in bom.columns else bom[bom.get("Kit ID","") == kit_id] if "Kit ID" in bom.columns else pd.DataFrame()
                    total = 0.0
                    for _, pr in parts.iterrows():
                        mid = pr.get("material_id") or pr.get("Material ID")
                        qpk = float(pr.get("qty_per_kit") or pr.get("Qty Per Kit") or 0)
                        row = snapshot[snapshot.get("id","") == mid] if "id" in snapshot.columns else snapshot[snapshot.get("ID","") == mid]
                        price = float(row["current_price_per_piece"].iat[0]) if not row.empty and "current_price_per_piece" in row.columns else (float(row["Current Price Per Piece"].iat[0]) if not row.empty and "Current Price Per Piece" in row.columns else 0.0)
                        unit_override = float(pr.get("unit_price_ref") or pr.get("Unit Price Ref") or 0)
                        unit = unit_override if unit_override > 0 else price
                        total += qpk * unit
                    return round(total,2)
                rows = []
                for r in df_norm.to_dict(orient="records"):
                    qty = int(r.get("Qty") or r.get("qty") or 1)
                    kit_id = r.get("Kit ID") or r.get("sku") or r.get("kit_id") or ""
                    cost_total = kit_cost_from_snapshot(kit_id, qty)
                    amount_received = _safe_float(r.get("Amount Received") or r.get("amount_received") or r.get("Amount") or r.get("amount") or 0)
                    profit = round(amount_received - cost_total, 2)
                    rows.append({
                        "date": r.get("Date") or "",
                        "order_id": r.get("Order ID") or "",
                        "kit_id": kit_id,
                        "kit_name": r.get("Kit Name") or "",
                        "qty": int(qty),
                        "platform": r.get("Platform") or platform,
                        "price": _safe_float(r.get("Price")),
                        "fees": _safe_float(r.get("Fees")),
                        "amount_received": amount_received,
                        "cost_price": cost_total,
                        "profit": profit,
                        "notes": notes or ""
                    })
                write_rows("sold_kits", pd.DataFrame(rows))
                st.success(f"Processed {len(rows)} rows and saved to sold_kits.")
    elif sub == "Record Sale":
        st.subheader("Record a sale (manual)")
        bom_df = read_table("kits_bom")
        kit_list = sorted(bom_df.get("kit_id", pd.Series()).dropna().unique().tolist()) if not bom_df.empty else []
        k_id = st.selectbox("Kit ID", [""] + kit_list)
        kit_name = st.text_input("Kit Name")
        platform = st.selectbox("Platform", ["Amazon", "Meesho", "Flipkart", "Offline", "Other"])
        qty = st.number_input("Quantity sold", min_value=1, step=1, value=1)
        price = st.number_input("Sale price per kit (₹)", min_value=0.0, step=0.01, value=0.0)
        fees = st.number_input("Fees / commission (total ₹)", min_value=0.0, step=0.01, value=0.0)
        amount_received = st.number_input("Amount received (total ₹)", min_value=0.0, step=0.01, value=float(price * qty))
        notes = st.text_input("Notes (optional)")
        sale_date = st.date_input("Sale date", value=date.today())
        if st.button("Save Sale"):
            snapshot = read_table("master")
            bom = read_table("kits_bom")
            cost_total = 0.0
            if not snapshot.empty and k_id:
                parts = bom[bom.get("kit_id","")==k_id] if not bom.empty and "kit_id" in bom.columns else bom[bom.get("Kit ID","")==k_id] if not bom.empty else pd.DataFrame()
                for _, pr in parts.iterrows():
                    mid = pr.get("material_id") or pr.get("Material ID")
                    qpk = float(pr.get("qty_per_kit") or pr.get("Qty Per Kit") or 0)
                    row = snapshot[snapshot.get("id","")==mid] if "id" in snapshot.columns else snapshot[snapshot.get("ID","")==mid]
                    price_piece = float(row["current_price_per_piece"].iat[0]) if not row.empty and "current_price_per_piece" in row.columns else (float(row["Current Price Per Piece"].iat[0]) if not row.empty and "Current Price Per Piece" in row.columns else 0.0)
                    unit_override = float(pr.get("unit_price_ref") or pr.get("Unit Price Ref") or 0)
                    unit = unit_override if unit_override>0 else price_piece
                    cost_total += qpk * unit
            profit = round(float(amount_received) - float(cost_total), 2)
            write_rows("sold_kits", pd.DataFrame([{
                "date": sale_date.strftime("%Y-%m-%d"),
                "order_id": "",
                "kit_id": k_id,
                "kit_name": kit_name,
                "qty": int(qty),
                "platform": platform,
                "price": float(price),
                "fees": float(fees),
                "amount_received": float(amount_received),
                "cost_price": float(cost_total),
                "profit": float(profit),
                "notes": notes
            }]))
            st.success(f"Sale recorded. Profit: ₹{profit:,.2f}")
    else:
        st.subheader("Sales Ledger")
        st.dataframe(read_table("sold_kits").sort_values("date", ascending=False) if not read_table("sold_kits").empty else pd.DataFrame([{"Status":"No sales"}]))

# ---------------- Data / Admin ----------------
elif main_section == "⬇️ Data":
    st.title("⬇️ Import / ⬆️ Export Data & Templates")
    st.markdown("Download current datasets or templates. Upload CSV to replace dataset (must match template columns).")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Download purchases.csv", read_table("purchases").to_csv(index=False).encode("utf-8"), "purchases.csv")
        st.download_button("Download master_snapshot.csv", read_table("master").to_csv(index=False).encode("utf-8"), "master_snapshot.csv")
    with e2:
        st.download_button("Download kits_bom.csv", read_table("kits_bom").to_csv(index=False).encode("utf-8"), "kits_bom.csv")
        st.download_button("Download created_kits.csv", read_table("created_kits").to_csv(index=False).encode("utf-8"), "created_kits.csv")
    with e3:
        st.download_button("Download sold_kits.csv", read_table("sold_kits").to_csv(index=False).encode("utf-8"), "sold_kits.csv")
        st.download_button("Download defective_items.csv", read_table("defective").to_csv(index=False).encode("utf-8"), "defective_items.csv")
    st.markdown("---")
    st.subheader("📄 Download CSV templates (headers only)")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.download_button("Template: purchases.csv", DEFAULTS["purchases"].to_csv(index=False).encode("utf-8"), "template_purchases.csv")
        st.download_button("Template: master_snapshot.csv", DEFAULTS["master"].to_csv(index=False).encode("utf-8"), "template_master_snapshot.csv")
    with t2:
        st.download_button("Template: kits_bom.csv", DEFAULTS["kits_bom"].to_csv(index=False).encode("utf-8"), "template_kits_bom.csv")
        st.download_button("Template: created_kits.csv", DEFAULTS["created_kits"].to_csv(index=False).encode("utf-8"), "template_created_kits.csv")
    with t3:
        st.download_button("Template: sold_kits.csv", DEFAULTS["sold_kits"].to_csv(index=False).encode("utf-8"), "template_sold_kits.csv")
        st.download_button("Template: defective.csv", DEFAULTS["defective"].to_csv(index=False).encode("utf-8"), "template_defective.csv")
    st.markdown("---")
    st.subheader("Upload CSV to replace a dataset (must match template columns)")
    which_map = {
        "purchases": "purchases",
        "kits_bom": "kits_bom",
        "created_kits": "created_kits",
        "sold_kits": "sold_kits",
        "defective": "defective",
        "restock": "restock"
    }
    which_choice = st.selectbox("Choose dataset to replace", list(which_map.keys()))
    up = st.file_uploader(f"Upload CSV for {which_choice}", type=["csv"], key="up_"+which_choice)
    if up is not None and st.button("Replace dataset"):
        try:
            new_df = pd.read_csv(up)
            expected = DEFAULTS[which_map[which_choice]].columns.tolist()
            if list(new_df.columns) != expected:
                st.warning(f"Uploaded headers differ from template. Attempting case-insensitive align...")
                mapping = {}
                lower_expected = {c.lower(): c for c in expected}
                for col in new_df.columns:
                    if col.lower() in lower_expected:
                        mapping[col] = lower_expected[col.lower()]
                new_df = new_df.rename(columns=mapping)
            if list(new_df.columns) != expected:
                st.error(f"Invalid columns. Expected: {expected}. Uploaded: {list(new_df.columns)}")
            else:
                overwrite_table(which_map[which_choice], new_df)
                if which_choice in ("purchases","created_kits","defective","kits_bom"):
                    recompute_master_snapshot_and_save()
                st.success(f"Replaced {which_choice} with uploaded CSV ({len(new_df)} rows).")
        except Exception as e:
            st.error(f"Failed to import: {e}")
    st.markdown("---")
    st.subheader("Admin: Manage Data (view / delete rows)")
    try:
        insp = inspect(engine)
        all_tables = insp.get_table_names(schema="public")
    except Exception:
        all_tables = list(DEFAULTS.keys()) + ["sold_kits", "master", "raw_uploads"]
    known = ["purchases","kits_bom","created_kits","sold_kits","defective","restock","master","raw_uploads"]
    ordered = [t for t in known if t in all_tables] + [t for t in all_tables if t not in known]
    sel_table = st.selectbox("Choose table to manage", ordered)
    if sel_table:
        df_view = read_table(sel_table)
        if df_view.empty:
            st.info("No rows in table.")
        else:
            st.write(f"Showing up to 200 rows from `{sel_table}`")
            st.dataframe(df_view.head(200))
            pk = get_primary_key_column(sel_table)
            st.write(f"Detected primary key: **{pk}**")
            options = df_view[pk].astype(str).tolist()
            to_delete = st.multiselect("Select rows (by PK) to delete", options=options)
            if to_delete:
                st.warning(f"Selected {len(to_delete)} row(s) for deletion.")
                if st.checkbox("Confirm deletion? This action cannot be undone."):
                    if st.button("Delete selected rows"):
                        prepared = []
                        for v in to_delete:
                            try:
                                prepared.append(int(v))
                            except Exception:
                                prepared.append(v)
                        deleted = delete_rows_by_pk(sel_table, prepared)
                        st.success(f"Deleted {deleted} row(s) from `{sel_table}`.")
                        st.experimental_rerun()

# ---------------- Footer ----------------
st.sidebar.markdown("---")
if st.sidebar.button("Refresh data (clear cache)"):
    try:
        read_table.clear()
    except Exception:
        pass
    st.experimental_rerun()
