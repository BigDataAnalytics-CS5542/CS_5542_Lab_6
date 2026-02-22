from sf_connect import get_conn

with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA();")
        print("Connection successful.")
        print(cur.fetchall())