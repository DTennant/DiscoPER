#!/usr/bin/env python3
"""Download and prepare datasets referenced by INQUIRE for the discovery pipeline.

Usage:
    # Download iNat24 metadata and build data/inquire_benchmarks/inat24_50k.parquet
    python scripts/download_inquire_datasets.py --datasets inat24 --output data/inquire_benchmarks

    # Download all external datasets
    python scripts/download_inquire_datasets.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


DATASETS = {
    "inat24": {
        "description": "iNaturalist 2024 metadata (5M observations, species+GPS+dates+taxonomy)",
        "files": [
            {
                "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2024/train.json.tar.gz",
                "filename": "train.json.tar.gz",
                "size_mb": 438,
                "extract": True,
            },
        ],
    },
    "newt": {
        "description": "NeWT labels (164 binary classification tasks, 36K images)",
        "files": [
            {
                "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_labels.csv.tar.gz",
                "filename": "newt_labels.tar.gz",
                "size_mb": 1,
                "extract": True,
            },
        ],
    },
    "inquire": {
        "description": "INQUIRE queries (250 expert ecological queries + 33K annotations)",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/main/data/inquire/inquire_queries_val.csv",
                "filename": "inquire_queries_val.csv",
                "size_mb": 0.01,
            },
            {
                "url": "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/main/data/inquire/inquire_queries_test.csv",
                "filename": "inquire_queries_test.csv",
                "size_mb": 0.02,
            },
            {
                "url": "https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/main/data/inquire/inquire_annotations.csv",
                "filename": "inquire_annotations.csv",
                "size_mb": 4.3,
            },
        ],
    },
    "wildclip": {
        "description": "WildCLIP camera trap attributes (Snapshot Serengeti, structured attributes)",
        "files": [
            {
                "url": "https://zenodo.org/records/10479317/files/val_dataset_crops_single_animal_template_captions_T1T7_ID.csv?download=1",
                "filename": "wildclip_val.csv",
                "size_mb": 96,
            },
        ],
    },
    "enc_vqa": {
        "description": "Encyclopedic-VQA (220K Q/A pairs, biology/ecology subset)",
        "files": [
            {
                "url": "https://storage.googleapis.com/encyclopedic-vqa/val.csv",
                "filename": "enc_vqa_val.csv",
                "size_mb": 1,
            },
            {
                "url": "https://storage.googleapis.com/encyclopedic-vqa/train.csv",
                "filename": "enc_vqa_train.csv",
                "size_mb": 74,
            },
        ],
    },
    "treeoflife": {
        "description": "TreeOfLife-10M taxonomy (8.4M images, full taxonomic labels)",
        "files": [
            {
                "url": "https://huggingface.co/datasets/imageomics/TreeOfLife-10M/resolve/main/metadata/species_level_taxonomy_chains.csv",
                "filename": "treeoflife_taxonomy.csv",
                "size_mb": 10,
            },
        ],
    },
}


def download_file(url: str, output_path: str, size_mb: float) -> bool:
    if os.path.exists(output_path):
        print(f"  SKIP (exists): {output_path}")
        return True

    print(f"  Downloading {os.path.basename(output_path)} ({size_mb:.0f}MB)...")
    result = subprocess.run(
        ["curl", "-sL", url, "-o", output_path],
        timeout=1800,
    )
    return result.returncode == 0


def build_inat24_50k_parquet(output_dir: str, n_samples: int = 50000, seed: int = 42) -> None:
    """Parse extracted train.json and write inat24_full_metadata.parquet + inat24_50k.parquet."""
    train_json = os.path.join(output_dir, "train.json")
    out_50k = os.path.join(output_dir, "inat24_50k.parquet")
    out_full = os.path.join(output_dir, "inat24_full_metadata.parquet")
    if os.path.exists(out_50k) or not os.path.exists(train_json):
        return

    print("  Building iNatDisco-50K parquet from train.json...")
    with open(train_json, "r") as f:
        raw = json.load(f)

    cats = {c["id"]: c for c in raw.get("categories", [])}
    img_to_cat = {ann["image_id"]: ann["category_id"] for ann in raw.get("annotations", [])}

    rows = []
    for img in raw.get("images", []):
        lat = img.get("latitude")
        lon = img.get("longitude")
        date_str = img.get("date", "")
        if lat is None or lon is None or not date_str:
            continue
        cat = cats.get(img_to_cat.get(img["id"]), {})
        rows.append(
            {
                "observation_id": int(img["id"]),
                "image_id": int(img["id"]),
                "file_name": img.get("file_name", ""),
                "species_name": cat.get("name", ""),
                "common_name": cat.get("common_name", ""),
                "kingdom": cat.get("kingdom", ""),
                "phylum": cat.get("phylum", ""),
                "class": cat.get("class", ""),
                "order": cat.get("order", ""),
                "family": cat.get("family", ""),
                "genus": cat.get("genus", ""),
                "latitude": float(lat),
                "longitude": float(lon),
                "location_uncertainty": float(img.get("location_uncertainty") or 0.0),
                "date": str(date_str)[:10],
            }
        )

    df = pd.DataFrame(rows)
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = dt.dt.month.fillna(1).astype(int)
    df["year"] = dt.dt.year.fillna(2024).astype(int)
    df["day_of_year"] = dt.dt.dayofyear.fillna(1).astype(int)
    df["hemisphere"] = np.where(df["latitude"] >= 0, "N", "S")
    df["abs_latitude"] = df["latitude"].abs()
    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    df["season_nh"] = df["month"].map(season_map)
    df["local_path"] = ""

    df.to_parquet(out_full, index=False)
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    sample_df.to_parquet(out_50k, index=False)
    # Also write data.parquet alias for image download scripts
    sample_df.to_parquet(os.path.join(output_dir, "data.parquet"), index=False)
    print(f"  Saved {len(sample_df):,} observations to {out_50k}")


def main():
    parser = argparse.ArgumentParser(description="Download INQUIRE-referenced datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Specific datasets to download",
    )
    parser.add_argument(
        "--output",
        default="data/inquire_benchmarks",
        help="Output directory (default: data/inquire_benchmarks)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        for name, info in DATASETS.items():
            total_mb = sum(f["size_mb"] for f in info["files"])
            print(f"  {name:12s} ({total_mb:>6.0f}MB): {info['description']}")
        return

    to_download = list(DATASETS.keys()) if args.all else (args.datasets or [])
    if not to_download:
        parser.print_help()
        return

    output_dir = os.path.join(str(ROOT), args.output)
    os.makedirs(output_dir, exist_ok=True)

    total_mb = sum(
        sum(f["size_mb"] for f in DATASETS[name]["files"]) for name in to_download
    )
    print(
        f"Downloading {len(to_download)} datasets ({total_mb:.0f}MB total) to {output_dir}"
    )

    for name in to_download:
        info = DATASETS[name]
        print(f"\n=== {name}: {info['description']} ===")
        for file_info in info["files"]:
            output_path = os.path.join(output_dir, file_info["filename"])
            success = download_file(file_info["url"], output_path, file_info["size_mb"])

            if success and file_info.get("extract"):
                print(f"  Extracting {file_info['filename']}...")
                with tarfile.open(output_path, "r:gz") as tar:
                    tar.extractall(path=output_dir)

        if name == "inat24":
            build_inat24_50k_parquet(output_dir)

    print(f"\nDone. Files in {output_dir}/")


if __name__ == "__main__":
    main()
