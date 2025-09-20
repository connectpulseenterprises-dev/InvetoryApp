# test_db_conn.py
import os, sys
from urllib.parse import urlparse
import psycopg2
import streamlit as st

# If running locally, export env var or paste your URL here:
DB_URL = None
if "DATABASE_URL" in os.environ:
    DB_URL = os.environ["DATABASE_URL"]
# Optionally override:
# DB_URL = "postgresql://postgres:password@db.xxxxxx.supabase.co:5432/postgres?sslmode=require"

if not DB_URL:
    print("DATABASE_URL not set in environment. If running in Streamlit, add to st.secrets['DATABASE_URL'] or set env var.")
    sys.exit(1)

print("Trying to connect to DB (masked):", DB_URL[:40] + "...")

try:
    # simple connect via psycopg2
    # ensure sslmode is present; if not, psycopg2.connect will accept parameters
    conn = psycopg2.connect(DB_URL, connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT version()")
    print("Connected. Postgres version:", cur.fetchone())
    cur.close()
    conn.close()
except Exception as e:
    # print full error (not redacted)
    import traceback
    traceback.print_exc()
    print("Exception repr:", repr(e))
