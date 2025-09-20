# inventory_app_supabase.py
"""
Inventory & Kits Manager using Supabase (Postgres) as persistent store.

Pre-req:
- Put DATABASE_URL in st.secrets (Supabase connection string, include ?sslmode=require)
- Install packages: streamlit,pandas,sqlalchemy,psycopg2-binary,python-dateutil

Usage:
streamlit run inventory_app_supabase.py
"""

import io
from datetime import datetime, date
from typing import Dict, List

import streamlit as st
import pandas as pd
from dateutil import parser as dateparser
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

st.set_page_config(page_title="Inventory (Supabase)", page_icon="📦", layout="wide")

# ----------------------
# Config / secrets
# ----------------------
if "DATABASE_URL" not in st.secrets:
    st.error("Please add DATABASE_URL to Streamlit secrets (Supabase Postgres connection string).")
    st.stop()

DATABASE_URL = st.secrets["DATABASE_URL"]

# create SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10, future=True)

# ----------------------
# Default templates
# ----------------------
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
}

# ----------------------
# DB helpers
# ----------------------
def init_db():
    """Create tables if they do not exist."""
    ddl = f"""
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
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

@st.cache_data(ttl=60)
def read_table(name: str) -> pd.DataFrame:
    """Read full table into DataFrame (cached)."""
    try:
        df = pd.read_sql(f"SELECT * FROM {name} ORDER BY id NULLS LAST", engine)
        if df.empty:
            return DEFAULTS.get(name, pd.DataFrame()).copy()
        return df
    except Exception:
        # if table doesn't have id (like master uses id as primary key), read differently
        try:
            df = pd.read_sql(f"SELECT * FROM {name}", engine)
            if df.empty:
                return DEFAULTS.get(name, pd.DataFrame()).copy()
            return df
        except Exception:
            return DEFAULTS.get(name, pd.DataFrame()).copy()

def write_rows(name: str, df: pd.DataFrame):
    """Append rows from df into table name using COPY via to_sql is fine for small datasets."""
    try:
        # use pandas to_sql append
        df.to_sql(name, engine, if_exists="append", index=False, method="multi", chunksize=500)
    except Exception as e:
        # fallback: insert row-by-row
        with engine.begin() as conn:
            for r in df.fillna("").to_dict(orient="records"):
                keys = ", ".join(r.keys())
                vals = ", ".join([f":{k}" for k in r.keys()])
                stmt = text(f"INSERT INTO {name} ({keys}) VALUES ({vals})")
                conn.execute(stmt, r)
    # clear cache
    try:
        read_table.clear()
    except Exception:
        pass

def overwrite_table(name: str, df: pd.DataFrame):
    """Replace table contents (used for master snapshot)."""
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {name}"))
    if not df.empty:
        # write with to_sql (replace would drop schema) so append after delete
        write_rows(name, df)

# ----------------------
# Utility functions
# ----------------------
def _safe_float(x):
    try:
        if x in (None, ""):
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ----------------------
# Snapshot logic (weighted average)
# ----------------------
def build_material_events(purchases: pd.DataFrame,
                          created_kits: pd.DataFrame,
                          defective: pd.DataFrame,
                          bom: pd.DataFrame) -> Dict[str, List[dict]]:
    events_by_mat = {}
    # purchases
    if purchases is not None and not purchases.empty:
        for r in purchases.to_dict(orient="records"):
            d = pd.to_datetime(r.get("date", r.get("Date")), errors="coerce")
            mid = str(r.get("material_id") or r.get("Material ID") or "")
            pieces = _safe_float(r.get("pieces") or r.get("Pieces") or 0)
            if pieces == 0:
                packs = _safe_float(r.get("packs") or r.get("Packs") or 0)
                qpp = _safe_float(r.get("qty_per_pack") or r.get("Qty Per Pack") or 0)
                pieces = packs * qpp
            ppp = _safe_float(r.get("price_per_piece") or r.get("Price Per Piece") or r.get("price") or 0)
            ev = {"date": d, "delta": pieces, "price_per_piece": ppp, "type": "purchase", "vendor": r.get("vendor") or r.get("Vendor", ""), "name": r.get("material_name") or r.get("Material Name", "")}
            events_by_mat.setdefault(mid, []).append(ev)
    # created kits -> expand via BOM
    if bom is not None and not bom.empty and created_kits is not None and not created_kits.empty:
        bom_map = {}
        for br in bom.to_dict(orient="records"):
            k = str(br.get("kit_id") or br.get("Kit ID") or "")
            bom_map.setdefault(k, []).append(br)
        for cr in created_kits.to_dict(orient="records"):
            d = pd.to_datetime(cr.get("date", cr.get("Date")), errors="coerce")
            kit_id = str(cr.get("kit_id") or cr.get("Kit ID") or "")
            qty_kits = _safe_float(cr.get("qty") or cr.get("Qty") or 0)
            if kit_id in bom_map:
                for br in bom_map[kit_id]:
                    mid = str(br.get("material_id") or br.get("Material ID") or "")
                    qty_per_kit = _safe_float(br.get("qty_per_kit") or br.get("Qty Per Kit") or 0)
                    total_consume = qty_per_kit * qty_kits
                    ev = {"date": d, "delta": -total_consume, "price_per_piece": None, "type": "consume"}
                    events_by_mat.setdefault(mid, []).append(ev)
    # defective
    if defective is not None and not defective.empty:
        for dr in defective.to_dict(orient="records"):
            d = pd.to_datetime(dr.get("date", dr.get("Date")), errors="coerce")
            mid = str(dr.get("material_id") or dr.get("Material ID") or "")
            qty = _safe_float(dr.get("qty") or dr.get("Qty") or 0)
            ev = {"date": d, "delta": -qty, "price_per_piece": None, "type": "defect"}
            events_by_mat.setdefault(mid, []).append(ev)
    # sort
    for mid, evs in events_by_mat.items():
        evs_sorted = sorted(evs, key=lambda x: pd.to_datetime(x.get("date")))
        events_by_mat[mid] = evs_sorted
    return events_by_mat

def compute_snapshot_from_events(events_by_mat: Dict[str, List[dict]]) -> pd.DataFrame:
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
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return df

def recompute_and_save_master():
    purchases = read_table("purchases")
    created = read_table("created_kits")
    defective = read_table("defective")
    bom = read_table("kits_bom")
    events = build_material_events(purchases, created, defective, bom)
    snap = compute_snapshot_from_events(events)
    # overwrite master table (use id column as PK)
    if not snap.empty:
        # rename columns to match master schema names
        snap2 = snap.rename(columns={"ID": "id", "Name": "name", "Vendor": "vendor",
                                     "Available Pieces": "available_pieces",
                                     "Current Price Per Piece": "current_price_per_piece",
                                     "Total Value": "total_value",
                                     "Total Purchased": "total_purchased",
                                     "Total Consumed": "total_consumed"})
        overwrite_table("master", snap2)
    else:
        overwrite_table("master", pd.DataFrame())
    # clear cache
    try:
        read_table.clear()
    except Exception:
        pass
    return snap

# ----------------------
# Marketplace parsers
# ----------------------
def normalize_header(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "_")

def parse_amazon(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame([])
    cols = {c: normalize_header(c) for c in df_raw.columns}
    df = df_raw.rename(columns=cols)
    # mapping heuristics
    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None
    date_k = pick(["purchase_date","order_date","date","sale_date"])
    sku_k = pick(["sku","seller_sku","product_sku","asin","item_sku"])
    title_k = pick(["title","product_title","item_name"])
    qty_k = pick(["quantity","qty_shipped","qty"])
    price_k = pick(["item_price","price","item_subtotal","sale_price"])
    fees_k = pick(["fees","fee","amazon_fees","commission"])
    amount_k = pick(["amount","total","amount_received","net_amount"])
    out = []
    for r in df.to_dict(orient="records"):
        d = r.get(date_k,"")
        try:
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
    df = df_raw.copy()
    df_parsed = parse_amazon(df)
    if not df_parsed.empty:
        df_parsed["Platform"] = "Flipkart"
    return df_parsed

def parse_meesho(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df_parsed = parse_amazon(df)
    if not df_parsed.empty:
        df_parsed["Platform"] = "Meesho"
    return df_parsed

# ----------------------
# UI & Controls
# ----------------------
init_db()  # ensure tables exist

st.title("📦 Inventory & Kits Manager (Supabase/Postgres)")

MAIN_SECTIONS = ["📊 Dashboards","📦 Inventory","🧩 Kits","🧾 Sales","⬇️ Data"]
main = st.sidebar.selectbox("Section", MAIN_SECTIONS)

# load tables (cached reads)
purchases_df = read_table("purchases")
bom_df = read_table("kits_bom")
created_df = read_table("created_kits")
sold_df = read_table("sold_kits")
defective_df = read_table("defective")
restock_df = read_table("restock")
master_df = read_table("master")

# Dashboards
if main == "📊 Dashboards":
    st.header("Inventory Snapshot")
    if master_df.empty:
        st.info("No master snapshot available. Recompute in Data or add purchases.")
    else:
        total_val = float(master_df["total_value"].sum()) if "total_value" in master_df.columns else float(master_df["Total Value"].sum())
        total_pieces = int(master_df["available_pieces"].sum()) if "available_pieces" in master_df.columns else int(master_df["Available Pieces"].sum())
        st.metric("Inventory value (₹)", f"₹{total_val:,.2f}")
        st.metric("Available pieces", f"{total_pieces}")
    st.subheader("Master Table")
    st.dataframe(master_df)

# Inventory
elif main == "📦 Inventory":
    st.header("Purchases")
    with st.expander("Add purchase"):
        col1, col2 = st.columns(2)
        with col1:
            mid = st.text_input("Material ID")
            mname = st.text_input("Material Name")
            vendor = st.text_input("Vendor")
            date_p = st.date_input("Purchase date", value=date.today())
        with col2:
            packs = st.number_input("Packs", min_value=0, step=1, value=0)
            qty_pack = st.number_input("Qty per pack", min_value=0, step=1, value=0)
            cost_pack = st.number_input("Cost per pack (₹)", min_value=0.0, step=0.01, value=0.0)
        pieces = int(packs * qty_pack)
        price_per_piece = float((cost_pack/qty_pack) if qty_pack>0 else 0.0)
        st.write(f"Pieces: {pieces} — Price/pc: ₹{price_per_piece:.4f}")
        if st.button("Save Purchase"):
            if not mid:
                st.error("Material ID required")
            else:
                df = pd.DataFrame([{
                    "date": date_p.strftime("%Y-%m-%d"),
                    "material_id": mid,
                    "material_name": mname,
                    "vendor": vendor,
                    "packs": int(packs),
                    "qty_per_pack": int(qty_pack),
                    "cost_per_pack": float(cost_pack),
                    "pieces": pieces,
                    "price_per_piece": float(price_per_piece)
                }])
                write_rows("purchases", df)
                st.success("Purchase saved")
                recompute_and_save_master()
    st.subheader("Purchase ledger")
    st.dataframe(purchases_df.sort_values("date", ascending=False) if not purchases_df.empty else pd.DataFrame([{"Status":"No purchases"}]))

# Kits
elif main == "🧩 Kits":
    st.header("BOM & Production")
    with st.expander("Add BOM row"):
        kit_id = st.text_input("Kit ID")
        kit_name = st.text_input("Kit Name")
        mat_id = st.text_input("Material ID")
        qty_per = st.number_input("Qty per kit", min_value=0.0, step=1.0, value=1.0)
        unit_override = st.number_input("Unit price override (₹)", min_value=0.0, step=0.01, value=0.0)
        if st.button("Add BOM row"):
            if not kit_id or not mat_id:
                st.error("Kit ID and Material ID required")
            else:
                df = pd.DataFrame([{
                    "kit_id": kit_id, "kit_name": kit_name, "material_id": mat_id,
                    "material_name": "", "qty_per_kit": float(qty_per),
                    "unit_price_ref": float(unit_override) if unit_override>0 else None
                }])
                write_rows("kits_bom", df)
                st.success("BOM row added")
    st.subheader("Produce kits")
    bom_table = bom_df
    kit_options = sorted(bom_table["kit_id"].dropna().unique().tolist()) if not bom_table.empty else []
    k = st.selectbox("Kit ID", [""]+kit_options)
    qty = st.number_input("Qty to produce", min_value=1, step=1, value=1)
    if st.button("Produce"):
        if not k:
            st.error("Choose kit")
        else:
            write_rows("created_kits", pd.DataFrame([{"date": datetime.today().strftime("%Y-%m-%d"), "kit_id": k, "kit_name": "", "qty": int(qty), "notes": ""}]))
            recompute_and_save_master()
            st.success("Produced and snapshot updated")
    st.dataframe(bom_table if not bom_table.empty else pd.DataFrame([{"Status":"No BOM"}]))

# Sales
elif main == "🧾 Sales":
    st.header("Sales")
    sub = st.sidebar.radio("Sales area", ["Upload marketplace CSV","Manual sale","Sales ledger"])
    if sub == "Upload marketplace CSV":
        st.subheader("Upload CSV (Amazon / Flipkart / Meesho)")
        platform = st.selectbox("Platform", ["Amazon","Flipkart","Meesho","Other"])
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None and st.button("Process file"):
            try:
                raw = pd.read_csv(io.BytesIO(uploaded.read()))
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
            if platform == "Amazon":
                parsed = parse_amazon(raw)
            elif platform == "Flipkart":
                parsed = parse_flipkart(raw)
            elif platform == "Meesho":
                parsed = parse_meesho(raw)
            else:
                parsed = parse_amazon(raw)
            if parsed.empty:
                st.warning("No rows parsed from the CSV.")
            else:
                # compute cost price and profit using snapshot (snapshot must exist)
                snapshot = read_table("master")
                # map current prices
                def kit_cost(kit_id, qty):
                    # look up bom rows
                    bom = read_table("kits_bom")
                    if bom.empty:
                        return 0.0
                    parts = bom[bom["kit_id"]==kit_id] if "kit_id" in bom.columns else bom[bom["Kit ID"]==kit_id]
                    total = 0.0
                    for _, pr in parts.iterrows():
                        mid = pr.get("material_id") or pr.get("Material ID")
                        qpk = float(pr.get("qty_per_kit") or pr.get("Qty Per Kit") or 0)
                        row = snapshot[snapshot["id"]==mid] if "id" in snapshot.columns else snapshot[snapshot["ID"]==mid]
                        price = float(row["current_price_per_piece"].iat[0]) if not row.empty and "current_price_per_piece" in row.columns else (float(row["Current Price Per Piece"].iat[0]) if not row.empty and "Current Price Per Piece" in row.columns else 0.0)
                        unit_override = float(pr.get("unit_price_ref") or 0)
                        unit = unit_override if unit_override>0 else price
                        total += qpk * unit
                    return round(total,2)
                # build rows for DB
                rows = []
                for r in parsed.to_dict(orient="records"):
                    qtyv = int(r.get("Qty") or r.get("qty") or 1)
                    cid = r.get("Kit ID") or r.get("sku") or r.get("Kit Id") or r.get("kit_id","")
                    cost_total = kit_cost(cid, qtyv)
                    amount_received = _safe_float(r.get("Amount Received") or r.get("amount_received") or r.get("Amount") or 0)
                    profit = round(amount_received - cost_total, 2)
                    rows.append({
                        "date": r.get("Date") or "",
                        "order_id": r.get("Order ID") or r.get("order_id",""),
                        "kit_id": cid,
                        "kit_name": r.get("Kit Name") or r.get("kit_name",""),
                        "qty": qtyv,
                        "platform": r.get("Platform") or platform,
                        "price": _safe_float(r.get("Price")),
                        "fees": _safe_float(r.get("Fees")),
                        "amount_received": amount_received,
                        "cost_price": cost_total,
                        "profit": profit,
                        "notes": r.get("Notes","")
                    })
                write_rows("sold_kits", pd.DataFrame(rows))
                st.success(f"Inserted {len(rows)} sales rows.")
    elif sub == "Manual sale":
        st.subheader("Record Sale")
        kit_list = sorted(bom_df["kit_id"].dropna().unique().tolist()) if not bom_df.empty else []
        k = st.selectbox("Kit ID", [""]+kit_list)
        kit_name = st.text_input("Kit Name")
        qty = st.number_input("Qty", min_value=1, step=1, value=1)
        platform = st.selectbox("Platform", ["Amazon","Flipkart","Meesho","Offline","Other"])
        price = st.number_input("Sale price per kit (₹)", min_value=0.0, step=0.01, value=0.0)
        fees = st.number_input("Fees total (₹)", min_value=0.0, step=0.01, value=0.0)
        amount_received = st.number_input("Amount received (total ₹)", min_value=0.0, step=0.01, value=float(price*qty))
        sale_date = st.date_input("Sale date", value=date.today())
        if st.button("Save Sale"):
            cost_total = 0.0
            # compute cost from snapshot if available
            snapshot = read_table("master")
            if not snapshot.empty and k:
                # compute using BOM (simple)
                bom = read_table("kits_bom")
                if not bom.empty:
                    parts = bom[bom["kit_id"]==k] if "kit_id" in bom.columns else bom[bom["Kit ID"]==k]
                    for _, pr in parts.iterrows():
                        mid = pr.get("material_id") or pr.get("Material ID")
                        qpk = float(pr.get("qty_per_kit") or pr.get("Qty Per Kit") or 0)
                        row = snapshot[snapshot["id"]==mid] if "id" in snapshot.columns else snapshot[snapshot["ID"]==mid]
                        price_piece = float(row["current_price_per_piece"].iat[0]) if not row.empty and "current_price_per_piece" in row.columns else (float(row["Current Price Per Piece"].iat[0]) if not row.empty and "Current Price Per Piece" in row.columns else 0.0)
                        unit_override = float(pr.get("unit_price_ref") or 0)
                        unit = unit_override if unit_override>0 else price_piece
                        cost_total += qpk * unit
            profit = round(float(amount_received) - float(cost_total), 2)
            write_rows("sold_kits", pd.DataFrame([{
                "date": sale_date.strftime("%Y-%m-%d"), "order_id": "", "kit_id": k,
                "kit_name": kit_name, "qty": int(qty), "platform": platform, "price": float(price),
                "fees": float(fees), "amount_received": float(amount_received),
                "cost_price": float(cost_total), "profit": float(profit), "notes": ""
            }]))
            st.success("Sale recorded.")
    elif sub == "Sales ledger":
        st.subheader("Sales ledger")
        st.dataframe(sold_df.sort_values("date", ascending=False) if not sold_df.empty else pd.DataFrame([{"Status":"No sales"}]))

# Data / exports
elif main == "⬇️ Data":
    st.header("Data Export & Admin")
    choice = st.selectbox("Choose table to export", list(DEFAULTS.keys())+["sold_kits","master"])
    table_map = {
        **{k:k for k in DEFAULTS.keys()},
        "sold_kits":"sold_kits",
        "master":"master"
    }
    if st.button("Export CSV"):
        tbl = table_map[choice]
        df = read_table(tbl)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(f"Download {tbl}.csv", csv, f"{tbl}.csv")
    if st.button("Recompute master snapshot"):
        snap = recompute_and_save_master()
        st.success("Snapshot recomputed")
        st.dataframe(snap)
    st.markdown("**Raw tables preview**")
    tab = st.selectbox("Preview table", ["purchases","kits_bom","created_kits","sold_kits","defective","restock","master"])
    st.dataframe(read_table(tab))

# Footer: refresh
st.sidebar.markdown("---")
if st.sidebar.button("Refresh"):
    try:
        read_table.clear()
    except Exception:
        pass
    st.experimental_rerun()
