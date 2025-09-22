# scripts/data_prep.py
from backend.app.summarizer import run_full_summary
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Optional row limit per table", default=None)
    args = parser.parse_args()
    run_full_summary(limit_per_table=args.limit)
    print("Summaries created/updated.")
