#!/usr/bin/env python3
"""
Script to verify the dataset structure and compatibility with the model.
"""
import os
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

def load_json_file(file_path):
    """Load and return JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        return None

def check_dataset_structure(dataset_path):
    """Check if the dataset has the expected structure and files."""
    print("\n=== Checking Dataset Structure ===")
    
    # Check required directories
    required_dirs = ['data', 'meta', 'videos']
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.isdir(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        print(f"✅ Found directory: {dir_name}")
    
    # Check required metadata files
    meta_dir = os.path.join(dataset_path, 'meta')
    required_meta_files = ['info.json', 'modality.json', 'stats.json']
    for file_name in required_meta_files:
        file_path = os.path.join(meta_dir, file_name)
        if not os.path.isfile(file_path):
            print(f"❌ Missing metadata file: {file_name}")
            return False
        print(f"✅ Found metadata file: {file_name}")
    
    return True

def check_metadata(dataset_path):
    """Check the content of metadata files."""
    print("\n=== Checking Metadata ===")
    meta_dir = os.path.join(dataset_path, 'meta')
    
    # Check info.json
    info = load_json_file(os.path.join(meta_dir, 'info.json'))
    if not info:
        return False
    
    print("\n=== Dataset Info ===")
    print(f"Dataset name: {info.get('name', 'N/A')}")
    print(f"Total episodes: {info.get('total_episodes', 'N/A')}")
    print(f"Total frames: {info.get('total_frames', 'N/A')}")
    print(f"FPS: {info.get('fps', 'N/A')}")
    
    # Check modality.json
    modality = load_json_file(os.path.join(meta_dir, 'modality.json'))
    if not modality:
        return False
    
    print("\n=== Modalities ===")
    for mod, items in modality.items():
        print(f"{mod}:")
        for item in items:
            print(f"  - {item}")
    
    # Check stats.json
    stats = load_json_file(os.path.join(meta_dir, 'stats.json'))
    if not stats:
        return False
    
    print("\n=== Statistics ===")
    for key, values in stats.items():
        if isinstance(values, dict):
            print(f"{key}:")
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {values}")
    
    return True

def check_data_files(dataset_path):
    """Check the data files in the dataset."""
    print("\n=== Checking Data Files ===")
    data_dir = os.path.join(dataset_path, 'data')
    
    # List all parquet files
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not parquet_files:
        print("❌ No parquet files found in data directory")
        return False
    
    print(f"Found {len(parquet_files)} parquet files:")
    for f in sorted(parquet_files):
        print(f"- {f}")
    
    # Check first parquet file
    first_file = os.path.join(data_dir, parquet_files[0])
    try:
        table = pq.read_table(first_file)
        print(f"\nFirst parquet file schema: {table.schema}")
        print(f"Number of rows: {len(table)}")
        print("\nFirst few columns:")
        for i, field in enumerate(table.schema):
            if i >= 5:  # Only show first 5 columns
                print("...")
                break
            print(f"- {field.name}: {field.type}")
    except Exception as e:
        print(f"❌ Error reading parquet file: {str(e)}")
        return False
    
    return True

def check_video_files(dataset_path):
    """Check the video files in the dataset."""
    print("\n=== Checking Video Files ===")
    videos_dir = os.path.join(dataset_path, 'videos')
    
    if not os.path.isdir(videos_dir):
        print("❌ Videos directory not found")
        return False
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("❌ No video files found")
        return False
    
    print(f"Found {len(video_files)} video files")
    print("First few video files:")
    for f in sorted(video_files)[:5]:
        print(f"- {f}")
    
    return True

def main():
    # Path to the dataset
    dataset_path = "/scratch/cbjp404/Isaac-GR00T/demo_data/robot_sim.PickNPlace"
    
    print(f"Dataset path: {dataset_path}")
    
    # Run checks
    structure_ok = check_dataset_structure(dataset_path)
    metadata_ok = check_metadata(dataset_path)
    data_ok = check_data_files(dataset_path)
    videos_ok = check_video_files(dataset_path)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Dataset structure: {'✅' if structure_ok else '❌'}")
    print(f"Metadata: {'✅' if metadata_ok else '❌'}")
    print(f"Data files: {'✅' if data_ok else '❌'}")
    print(f"Video files: {'✅' if videos_ok else '❌'}")
    
    if all([structure_ok, metadata_ok, data_ok, videos_ok]):
        print("\n✅ Dataset appears to be valid and properly structured!")
    else:
        print("\n❌ There are issues with the dataset. Please check the messages above.")

if __name__ == "__main__":
    main()
