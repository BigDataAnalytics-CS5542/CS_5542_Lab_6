import sys
from pathlib import Path
from sf_connect import get_conn

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_sql_file.py <sql_file_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    sql = path.read_text(encoding="utf-8")

    totp = input("Enter TOTP code: ").strip()

    with get_conn(totp) as conn:
        with conn.cursor() as cur:
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                cur.execute(stmt)
                try:
                    rows = cur.fetchall()
                    if rows:
                        print("Result:", rows)
                except Exception:
                    pass

    print(f"Successfully ran {path}")

if __name__ == "__main__":
    main()