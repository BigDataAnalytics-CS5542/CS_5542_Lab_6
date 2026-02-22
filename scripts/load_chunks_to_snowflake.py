import os
import time
from pathlib import Path
from sf_connect import get_conn

STAGE_NAME = "ROHAN_BLAKE_KENNETH_STAGE"
FILE_FORMAT = "ROHAN_BLAKE_KENNETH_CSV_FMT"
TARGET_TABLE = "RAW.CHUNKS"

LOCAL_PATH = Path("data/chunks.csv")

def run(cur, sql: str):
    cur.execute(sql)
    try:
        return cur.fetchall()
    except Exception:
        return None

def main():
    if not LOCAL_PATH.exists():
        raise FileNotFoundError(f"Missing {LOCAL_PATH}. Run export step first.")

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Ensure schemas/table exist (idempotent)
            run(cur, "CREATE SCHEMA IF NOT EXISTS RAW")
            run(cur, "CREATE SCHEMA IF NOT EXISTS APP")
            run(cur, f"""
                CREATE OR REPLACE TABLE {TARGET_TABLE} (
                  EVIDENCE_ID STRING,
                  DOC_ID STRING,
                  SOURCE_FILE STRING,
                  PAGE NUMBER,
                  CHUNK_INDEX NUMBER,
                  CHUNK_TEXT STRING
                )
            """)

            # File format: safe for quoted CSV (we wrote QUOTE_ALL)
            run(cur, f"""
                CREATE OR REPLACE FILE FORMAT {FILE_FORMAT}
                  TYPE = CSV
                  SKIP_HEADER = 1
                  FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
                  ESCAPE = '\\\\'
                  NULL_IF = ('', 'NULL', 'null');
            """)

            run(cur, f"CREATE OR REPLACE STAGE {STAGE_NAME} FILE_FORMAT = {FILE_FORMAT}")

            # PUT upload
            put_sql = f"PUT file://{LOCAL_PATH.resolve()} @{STAGE_NAME} AUTO_COMPRESS=TRUE OVERWRITE=TRUE;"
            print(put_sql)
            cur.execute(put_sql)
            print("PUT result:", cur.fetchall())

            # COPY INTO (explicit column order)
            filename = LOCAL_PATH.name
            copy_sql = f"""
            COPY INTO {TARGET_TABLE} (EVIDENCE_ID, DOC_ID, SOURCE_FILE, PAGE, CHUNK_INDEX, CHUNK_TEXT)
            FROM (
              SELECT
                $1, $2, $3,
                TRY_TO_NUMBER($4),
                TRY_TO_NUMBER($5),
                $6
              FROM @{STAGE_NAME}/{filename}.gz
            )
            ON_ERROR = 'ABORT_STATEMENT';
            """

            t0 = time.time()
            cur.execute(copy_sql)
            res = cur.fetchall()
            dt_ms = int((time.time() - t0) * 1000)

            print("COPY result:", res)
            print(f"Load latency: {dt_ms} ms")

            # Validate counts
            cur.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
            print("Row count:", cur.fetchall())

if __name__ == "__main__":
    main()