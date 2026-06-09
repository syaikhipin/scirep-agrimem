#!/usr/bin/env python3
"""
Download PlantVillage and PlantDoc Datasets

Downloads the official datasets:
1. PlantVillage from Kaggle (54,305 images, 38 classes)
2. PlantDoc from GitHub (2,598 images, 27 classes)

Prerequisites:
- kaggle API token (~/.kaggle/kaggle.json)
- git installed

Usage:
    python download_datasets.py --output data/images
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def download_plantvillage_kaggle(output_dir: str):
    """
    Download PlantVillage dataset from Kaggle.

    Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
    Contains 54,305 images across 38 classes.
    """
    print("=" * 60)
    print("Downloading PlantVillage from Kaggle...")
    print("=" * 60)

    output_path = Path(output_dir) / "PlantVillage_full"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Check if kaggle is installed
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        print(f"Kaggle CLI: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: Kaggle CLI not installed. Install with: pip install kaggle")
        print("Also ensure you have ~/.kaggle/kaggle.json with your API credentials")
        return False

    # Download dataset
    try:
        print("Downloading dataset (this may take a while)...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "emmarex/plantdisease",
            "-p", str(output_path),
            "--unzip"
        ], check=True)
        print(f"PlantVillage downloaded to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR downloading PlantVillage: {e}")
        return False


def download_plantvillage_huggingface(output_dir: str):
    """
    Alternative: Download PlantVillage from HuggingFace.

    Dataset: https://huggingface.co/datasets/plantvillage
    """
    print("=" * 60)
    print("Downloading PlantVillage from HuggingFace...")
    print("=" * 60)

    try:
        from datasets import load_dataset

        output_path = Path(output_dir) / "PlantVillage_full"
        output_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset from HuggingFace (this may take a while)...")
        dataset = load_dataset("plantvillage", split="train")

        print(f"Dataset size: {len(dataset)} images")
        print(f"Features: {dataset.features}")

        # Save images to disk
        for i, item in enumerate(dataset):
            label = item['label']
            image = item['image']

            label_dir = output_path / str(label)
            label_dir.mkdir(exist_ok=True)

            image_path = label_dir / f"image_{i:06d}.jpg"
            image.save(image_path)

            if (i + 1) % 1000 == 0:
                print(f"Saved {i + 1}/{len(dataset)} images")

        print(f"PlantVillage saved to: {output_path}")
        return True

    except ImportError:
        print("ERROR: datasets library not installed. Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_plantdoc_github(output_dir: str):
    """
    Download PlantDoc dataset from GitHub.

    Repository: https://github.com/pratikkayal/PlantDoc-Dataset
    Contains 2,598 images across 27 classes.
    """
    print("=" * 60)
    print("Downloading PlantDoc from GitHub...")
    print("=" * 60)

    output_path = Path(output_dir) / "PlantDoc_full"
    output_path.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
    temp_dir = output_path / "temp_clone"

    try:
        # Clone the repository
        print(f"Cloning {repo_url}...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        subprocess.run([
            "git", "clone", "--depth", "1", repo_url, str(temp_dir)
        ], check=True)

        # Move train and test folders
        for split in ["train", "test"]:
            src = temp_dir / split
            dst = output_path / split
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
                print(f"Moved {split} folder to {dst}")

        # Clean up
        shutil.rmtree(temp_dir)

        print(f"PlantDoc downloaded to: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"ERROR cloning PlantDoc: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_plantdoc_direct(output_dir: str):
    """
    Alternative: Direct download of PlantDoc zip file.
    """
    print("=" * 60)
    print("Downloading PlantDoc directly...")
    print("=" * 60)

    import urllib.request

    output_path = Path(output_dir) / "PlantDoc_full"
    output_path.mkdir(parents=True, exist_ok=True)

    # Direct download URL (from GitHub releases)
    zip_url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    zip_path = output_path / "plantdoc.zip"

    try:
        print(f"Downloading from {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)

        # Move files from extracted folder
        extracted_dir = output_path / "PlantDoc-Dataset-master"
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                if item.is_dir() and item.name in ["train", "test"]:
                    dst = output_path / item.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(item), str(dst))

            shutil.rmtree(extracted_dir)

        zip_path.unlink()

        print(f"PlantDoc downloaded to: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def count_images(directory: str) -> dict:
    """Count images in directory by class."""
    counts = {}
    path = Path(directory)

    if not path.exists():
        return counts

    for class_dir in path.iterdir():
        if class_dir.is_dir():
            n_images = sum(1 for f in class_dir.iterdir()
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            counts[class_dir.name] = n_images

    return counts


def main():
    parser = argparse.ArgumentParser(description="Download PlantVillage and PlantDoc datasets")
    parser.add_argument("--output", type=str, default="data/images",
                        help="Output directory for datasets")
    parser.add_argument("--plantvillage-source", choices=["kaggle", "huggingface"],
                        default="kaggle", help="Source for PlantVillage")
    parser.add_argument("--skip-plantvillage", action="store_true",
                        help="Skip PlantVillage download")
    parser.add_argument("--skip-plantdoc", action="store_true",
                        help="Skip PlantDoc download")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AgriMemory Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Download PlantVillage
    if not args.skip_plantvillage:
        if args.plantvillage_source == "kaggle":
            success = download_plantvillage_kaggle(str(output_dir))
        else:
            success = download_plantvillage_huggingface(str(output_dir))

        if not success:
            print("PlantVillage download failed. Trying alternative source...")
            if args.plantvillage_source == "kaggle":
                download_plantvillage_huggingface(str(output_dir))
            else:
                download_plantvillage_kaggle(str(output_dir))
    else:
        print("Skipping PlantVillage download")

    print()

    # Download PlantDoc
    if not args.skip_plantdoc:
        success = download_plantdoc_github(str(output_dir))
        if not success:
            print("Git clone failed. Trying direct download...")
            download_plantdoc_direct(str(output_dir))
    else:
        print("Skipping PlantDoc download")

    print()

    # Summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)

    pv_path = output_dir / "PlantVillage_full"
    pd_path = output_dir / "PlantDoc_full"

    if pv_path.exists():
        print("\nPlantVillage:")
        # Check for nested structure
        for split in ["train", "val", "test", ""]:
            check_path = pv_path / split if split else pv_path
            if check_path.exists():
                counts = count_images(str(check_path))
                total = sum(counts.values())
                if total > 0:
                    print(f"  {split or 'root'}: {len(counts)} classes, {total} images")
    else:
        print("\nPlantVillage: Not downloaded")

    if pd_path.exists():
        print("\nPlantDoc:")
        for split in ["train", "test"]:
            split_path = pd_path / split
            if split_path.exists():
                counts = count_images(str(split_path))
                total = sum(counts.values())
                print(f"  {split}: {len(counts)} classes, {total} images")
    else:
        print("\nPlantDoc: Not downloaded")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
