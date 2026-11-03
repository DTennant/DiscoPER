#!/usr/bin/env python3
"""Download verified iNaturalist images by searching the API for each species.

Saves images to a shared location (~/DataExplorer_results/inat_verified_images/)
so they can be reused across workspaces. Also updates the local data.parquet
with correct observation_ids and local_paths.

Usage:
    python scripts/download_verified_inat_images.py --data-dir data/inat24_50k \
        --image-dir ~/DataExplorer_results/inat_verified_images \
        --max-per-species 200
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

import pandas as pd


def fetch_species_observations(species, n_needed, max_pages=10):
    """Search iNat API for research-grade observations of a species."""
    search = species.replace(" ", "+")
    all_obs = []
    per_page = min(200, n_needed)
    for page in range(1, max_pages + 1):
        if len(all_obs) >= n_needed:
            break
        url = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_name={search}&quality_grade=research&photos=true"
            f"&per_page={per_page}&page={page}&order=desc&order_by=id"
            f"&fields=id,photos,location"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "DataExplorer/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                if r.get("photos"):
                    loc = r.get("location", "")
                    lat, lon = 0.0, 0.0
                    if loc and "," in loc:
                        lat, lon = float(loc.split(",")[0]), float(loc.split(",")[1])
                    all_obs.append(
                        {
                            "obs_id": r["id"],
                            "photo_url": r["photos"][0]["url"].replace(
                                "square", "medium"
                            ),
                            "lat": lat,
                            "lon": lon,
                        }
                    )
        except Exception:
            break
        time.sleep(0.1)
    return all_obs


def download_photo(obs_id, photo_url, species, image_dir):
    """Download a single photo to image_dir/Species_name/obs_id.jpg."""
    sp_dir = os.path.join(image_dir, species.replace(" ", "_"))
    os.makedirs(sp_dir, exist_ok=True)
    fpath = os.path.join(sp_dir, f"{obs_id}.jpg")
    if os.path.exists(fpath) and os.path.getsize(fpath) > 500:
        return fpath
    try:
        urllib.request.urlretrieve(photo_url, fpath)
        if os.path.getsize(fpath) > 500:
            return fpath
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Download verified iNat images")
    parser.add_argument(
        "--data-dir",
        default="data/inat_subset",
        help="Path to dataset dir with data.parquet (default: data/inat_subset)",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory to store downloaded images (default: <data-dir>/images)",
    )
    parser.add_argument("--max-per-species", type=int, default=200)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()

    if args.image_dir is None:
        args.image_dir = os.path.join(args.data_dir, "images")
    os.makedirs(args.image_dir, exist_ok=True)

    parquet_path = os.path.join(args.data_dir, "data.parquet")
    df = pd.read_parquet(parquet_path)
    species_counts = df["species_name"].value_counts()
    total_species = len(species_counts)
    print(f"Dataset: {len(df)} rows, {total_species} species")

    # Step 1: Fetch correct observation IDs from iNat API
    print("\n=== Step 1: Fetching observation IDs from iNat API ===")
    species_obs = {}
    for i, (species, n_needed) in enumerate(species_counts.items()):
        n_fetch = min(args.max_per_species, n_needed)
        species_obs[species] = fetch_species_observations(species, n_fetch)
        if (i + 1) % 100 == 0:
            found = sum(len(v) for v in species_obs.values())
            pct = 100 * (i + 1) / total_species
            print(f"  [{pct:.0f}%] {i + 1}/{total_species} species, {found} obs found")

    found = sum(len(v) for v in species_obs.values())
    covered = sum(1 for v in species_obs.values() if v)
    print(f"  Done: {found} obs for {covered}/{total_species} species")

    # Step 2: Replace observation IDs in dataframe
    print("\n=== Step 2: Updating observation IDs ===")
    photo_tasks = []
    replaced = 0
    for species, obs_list in species_obs.items():
        if not obs_list:
            continue
        sp_idx = df[df["species_name"] == species].index
        for j, idx in enumerate(sp_idx):
            if j < len(obs_list):
                obs = obs_list[j]
                df.at[idx, "observation_id"] = obs["obs_id"]
                if obs["lat"] != 0:
                    df.at[idx, "latitude"] = obs["lat"]
                    df.at[idx, "longitude"] = obs["lon"]
                photo_tasks.append((obs["obs_id"], obs["photo_url"], species))
                replaced += 1
    print(f"  Replaced {replaced}/{len(df)} observation IDs")

    # Step 3: Download images
    print(
        f"\n=== Step 3: Downloading {len(photo_tasks)} images ({args.workers} workers) ==="
    )
    results = {}

    def dl_wrapper(task):
        obs_id, photo_url, species = task
        path = download_photo(obs_id, photo_url, species, args.image_dir)
        return obs_id, species, path

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        done = 0
        for f in concurrent.futures.as_completed(
            {ex.submit(dl_wrapper, t): t for t in photo_tasks}
        ):
            obs_id, species, path = f.result()
            if path:
                results[obs_id] = (path, species)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(photo_tasks)}, {len(results)} downloaded")

    print(f"  Downloaded {len(results)}/{len(photo_tasks)} images")

    # Step 4: Update local_path in dataframe (relative to data_dir)
    print("\n=== Step 4: Updating data.parquet ===")

    # Symlink shared images into data_dir
    local_images = os.path.join(args.data_dir, "images")
    if os.path.islink(local_images):
        os.remove(local_images)
    if not os.path.isdir(local_images):
        os.symlink(os.path.abspath(args.image_dir), local_images)
        print(f"  Symlinked {local_images} -> {args.image_dir}")

    df["local_path"] = ""
    for obs_id, (abs_path, species) in results.items():
        rel_path = os.path.join("images", species.replace(" ", "_"), f"{obs_id}.jpg")
        df.loc[df["observation_id"] == obs_id, "local_path"] = rel_path

    total_imgs = (df["local_path"].astype(str).str.len() > 0).sum()
    df.to_parquet(parquet_path, index=False)
    print(
        f"  Saved: {total_imgs}/{len(df)} with verified images ({100 * total_imgs / len(df):.1f}%)"
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
