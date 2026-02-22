import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def get_conn():
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
    mfa_code = (os.getenv("SNOWFLAKE_MFA_CODE") or "").strip()
    if authenticator:
        conn_kwargs["authenticator"] = authenticator
        conn_kwargs.pop("password", None)
    elif mfa_code:
        conn_kwargs["authenticator"] = "username_password_mfa"
        conn_kwargs["passcode"] = mfa_code

    # remove None values
    conn_kwargs = {k: v for k, v in conn_kwargs.items() if v}

    return snowflake.connector.connect(**conn_kwargs)