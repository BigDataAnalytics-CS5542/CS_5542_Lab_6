"""
Snowflake connection helper.
Reads config from .env (account, user, password, warehouse, database, schema).
Provides get_conn() for use by other scripts.

Usage:
    from scripts.sf_connect import get_conn
    conn = get_conn()

    # Or run directly to test connection:
    python scripts/sf_connect.py
"""
import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def get_conn(passcode=""):
    required = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}. Fill .env from .env.example")

    conn_kwargs = dict(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        role=os.getenv("SNOWFLAKE_ROLE") or None,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR")

    if authenticator:
        conn_kwargs["authenticator"] = authenticator
        conn_kwargs.pop("password", None)
    elif passcode:
        conn_kwargs["authenticator"] = "username_password_mfa"
        conn_kwargs["passcode"] = passcode

    conn_kwargs = {k: v for k, v in conn_kwargs.items() if v}
    return snowflake.connector.connect(**conn_kwargs)


if __name__ == "__main__":
    mfa = input("Enter MFA code (or press Enter if no MFA): ").strip()
    try:
        conn = get_conn(passcode=mfa)
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_WAREHOUSE(), CURRENT_DATABASE()")
        print("Connected:", cur.fetchone())
        conn.close()
    except Exception as e:
        print("Connection failed:", e)