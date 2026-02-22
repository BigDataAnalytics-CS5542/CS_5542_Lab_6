import csv
import json
from pathlib import Path

KB_PATH = Path("data/processed/kb.jsonl")
OUT_PATH = Path("data/chunks.csv")

def infer_doc_id(evidence_id: str, source_file: str) -> str:
    # Prefer file stem, fallback to evidence prefix
    if source_file:
        return Path(source_file).stem
    # evidence_id like "bm25_prf_p01_c003" -> "bm25_prf"
    return evidence_id.split("_p", 1)[0] if "_p" in evidence_id else evidence_id

def main():
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Missing {KB_PATH}. Run build_kb() first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with KB_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "evidence_id",
                "doc_id",
                "source_file",
                "page",
                "chunk_index",
                "chunk_text",
            ],
            quoting=csv.QUOTE_ALL,  # important: prevents COPY breakage from commas/newlines
            escapechar="\\",
        )
        writer.writeheader()

        for line in fin:
            obj = json.loads(line)
            evidence_id = obj.get("evidence_id", "")
            source_file = obj.get("source_file", "")
            doc_id = infer_doc_id(evidence_id, source_file)

            writer.writerow(
                {
                    "evidence_id": evidence_id,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "page": obj.get("page", 0),
                    "chunk_index": obj.get("chunk_index", 0),
                    "chunk_text": obj.get("text", ""),
                }
            )

    print(f"Successfully wrote {OUT_PATH} from {KB_PATH}")

if __name__ == "__main__":
    main()