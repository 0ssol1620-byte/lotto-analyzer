# update_data.py
# -*- coding: utf-8 -*-
import argparse, sys
from lotto_data_fixed import incremental_update

def main():
    p = argparse.ArgumentParser(description="Lotto 6/45 incremental updater")
    p.add_argument("--data-path", default="data/lotto_draws.csv", help="CSV 저장 경로")
    args = p.parse_args()
    try:
        df, prev, latest = incremental_update(args.data_path)
    except Exception as e:
        print(f"[ERROR] Update failed: {e}")
        return 2
    if prev >= latest:
        print(f"[INFO] Already up-to-date. latest={latest}, rows={len(df)}")
    else:
        print(f"[INFO] Updated: {prev+1}..{latest} appended. rows={len(df)}")

if __name__ == "__main__":
    sys.exit(main() or 0)
