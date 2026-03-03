from sf_connect import get_conn

totp = input("Enter TOTP code: ").strip()

with get_conn(totp) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM RAW.PAPERS;")
        print("Connection successful.")
        print(cur.fetchall())