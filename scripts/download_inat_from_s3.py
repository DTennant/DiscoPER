#!/usr/bin/env python3
"""Download iNat 50K images directly from the iNat24 S3 bucket.

Uses train.json to map image_id -> file_name, then downloads from S3.
No API calls needed - avoids rate limiting entirely.

Usage:
    python scripts/download_inat_from_s3.py \
        --data-dir data/inat24_50k \
        --train-json ~/DataExplorer_results/inquire_benchmarks/train.json \
        --full-metadata ~/DataExplorer_results/inquire_benchmarks/inat24_full_metadata.parquet \
        --image-dir ~/DataExplorer_results/inat_verified_images
"""

import argparse
import concurrent.futures
import json
import os
import sys
import urllib.request

import pandas as pd

S3_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2024/"


def build_image_id_to_filename(train_json_path, needed_ids):
    """Stream-parse train.json to get file_name for each needed image_id."""
    id_to_filename = {}
    print(f"Parsing train.json for {len(needed_ids)} image IDs...")

    with open(train_json_path) as f:
        chunk_size = 10_000_000
        buffer = ""
        in_images = False
        found = 0

        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            if not in_images:
                idx = buffer.find('"images"')
                if idx >= 0:
                    in_images = True
                    buffer = buffer[idx:]
                else:
                    buffer = buffer[-100:]
                    continue

            while True:
                start = buffer.find('{"id"')
                if start < 0:
                    buffer = buffer[-200:]
                    break

                end = buffer.find("}", start)
                if end < 0:
                    break

                obj_str = buffer[start : end + 1]
                buffer = buffer[end + 1 :]

                try:
                    obj = json.loads(obj_str)
                    img_id = obj.get("id")
                    if img_id in needed_ids:
                        id_to_filename[img_id] = obj["file_name"]
                        found += 1
                        if found % 5000 == 0:
                            print(f"  Found {found}/{len(needed_ids)}")
                        if found >= len(needed_ids):
                            return id_to_filename
                except json.JSONDecodeError:
                    continue

    return id_to_filename


def download_image(args):
    obs_id, s3_url, species, image_dir = args
    sp_dir = os.path.join(image_dir, species.replace(" ", "_"))
    os.makedirs(sp_dir, exist_ok=True)
    fpath = os.path.join(sp_dir, f"{obs_id}.jpg")
    if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
        return obs_id, os.path.join(
            "images", species.replace(" ", "_"), f"{obs_id}.jpg"
        )
    try:
        urllib.request.urlretrieve(s3_url, fpath)
        if os.path.getsize(fpath) > 500:
            return obs_id, os.path.join(
                "images", species.replace(" ", "_"), f"{obs_id}.jpg"
            )
    except Exception:
        pass
    return obs_id, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/inquire_benchmarks")
    parser.add_argument(
        "--train-json",
        default="data/inquire_benchmarks/train.json",
    )
    parser.add_argument(
        "--full-metadata",
        default="data/inquire_benchmarks/inat24_full_metadata.parquet",
    )
    parser.add_argument(
        "--image-dir",
        default="data/inquire_benchmarks/images",
    )
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)

    # Load our 50K subset
    df = pd.read_parquet(os.path.join(args.data_dir, "data.parquet"))
    print(f"Dataset: {len(df)} rows, {df['species_name'].nunique()} species")

    # Load full metadata to get original image_ids
    full = pd.read_parquet(args.full_metadata)
    print(f"Full metadata: {len(full)} rows")

    species_counts = df["species_name"].value_counts()

    print("Grouping full metadata by species...")
    grouped = full.groupby("species_name")

    image_ids_needed = set()
    id_to_species = {}

    for species, n_needed in species_counts.items():
        if species not in grouped.groups:
            continue
        sp_full = grouped.get_group(species).head(n_needed)
        for img_id in sp_full["image_id"].values:
            img_id = int(img_id)
            image_ids_needed.add(img_id)
            id_to_species[img_id] = species

    print(f"Need filenames for {len(image_ids_needed)} images")

    # Parse train.json to get file_names
    id_to_filename = build_image_id_to_filename(args.train_json, image_ids_needed)
    print(f"Found filenames for {len(id_to_filename)}/{len(image_ids_needed)}")

    # Build download tasks
    tasks = []
    for img_id, filename in id_to_filename.items():
        s3_url = S3_BASE + filename
        species = id_to_species[img_id]
        tasks.append((img_id, s3_url, species, args.image_dir))

    print(f"\nDownloading {len(tasks)} images from S3...")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for f in concurrent.futures.as_completed(
            {ex.submit(download_image, t): t for t in tasks}
        ):
            obs_id, path = f.result()
            if path:
                results[obs_id] = path
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(tasks)}, {len(results)} downloaded")

    print(f"Downloaded {len(results)}/{len(tasks)} images")

    # Update dataframe with correct image_ids and local_paths
    local_images = os.path.join(args.data_dir, "images")
    if os.path.islink(local_images):
        os.remove(local_images)
    if not os.path.isdir(local_images):
        os.symlink(os.path.abspath(args.image_dir), local_images)

    df["local_path"] = ""
    for species, n_needed in species_counts.items():
        if species not in grouped.groups:
            continue
        sp_full = grouped.get_group(species).head(n_needed)
        sp_idx = df[df["species_name"] == species].index
        img_ids = sp_full["image_id"].values
        lats = sp_full["latitude"].values
        lons = sp_full["longitude"].values
        for j in range(min(len(img_ids), len(sp_idx))):
            idx = sp_idx[j]
            img_id = int(img_ids[j])
            df.at[idx, "observation_id"] = img_id
            df.at[idx, "latitude"] = float(lats[j])
            df.at[idx, "longitude"] = float(lons[j])
            if img_id in results:
                df.at[idx, "local_path"] = results[img_id]

    total = (df["local_path"].astype(str).str.len() > 0).sum()
    df.to_parquet(os.path.join(args.data_dir, "data.parquet"), index=False)
    print(f"\nSaved: {total}/{len(df)} with images ({100 * total / len(df):.1f}%)")


if __name__ == "__main__":
    main()
