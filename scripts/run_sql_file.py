"""
Run a SQL file against Snowflake.
Context (warehouse, database) is set from .env — no hardcoded names in SQL files.

Usage:
    python scripts/run_sql_file.py sql/01_create_schema.sql
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.sf_connect import get_conn


def run_sql_file(filepath: str, passcode: str = ""):
    with open(filepath, "r") as f:
        sql = f.read()

    statements = [s.strip() for s in sql.split(";") if s.strip()]

    conn = get_conn(passcode=passcode)
    cur = conn.cursor()

    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    print(f"Using warehouse: {warehouse}")
    print(f"Using database:  {database}")
    print("-" * 50)

    cur.execute(f"USE WAREHOUSE {warehouse}")
    cur.execute(f"USE DATABASE {database}")

    for i, stmt in enumerate(statements, 1):
        preview = stmt[:80].replace("\n", " ")
        print(f"[{i}/{len(statements)}] {preview}...")
        try:
            cur.execute(stmt)
            try:
                rows = cur.fetchall()
                if rows:
                    for row in rows:
                        print("  →", row)
            except Exception:
                pass
        except Exception as e:
            print(f"  Error: {e}")

    conn.close()
    print("-" * 50)
    print("Done")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_sql_file.py <path/to/file.sql>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    mfa = input("MFA code (or press Enter to skip): ").strip()
    run_sql_file(filepath, passcode=mfa)