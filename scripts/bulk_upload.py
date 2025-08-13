#!/usr/bin/env python3
"""
Bulk upload pretrained_model checkpoints to Hugging Face Hub under unique subfolders.

Each checkpoint directory is uploaded to:
  runs/<date>/<run>/<step>

Notes:
- README.md files inside checkpoints are ignored so the Hub model card is not overwritten.
- Requires HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, upload_folder


def find_pretrained_model_dirs(base_dir: str) -> List[Path]:
    pattern = os.path.join(base_dir, "**", "pretrained_model")
    return [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk upload pretrained_model folders to HF Hub")
    parser.add_argument("--base", required=True, help="Base directory to scan (e.g., outputs/train/2025-08-12)")
    parser.add_argument("--repo_id", required=True, help="HF Hub repo id (e.g., user/repo)")
    parser.add_argument("--branch", default="main", help="Target branch on the Hub")
    parser.add_argument("--private", action="store_true", default=True, help="Create repo as private if needed")
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=["*.safetensors", "*.bin", "*.json", "*.txt"],
        help="Whitelist of file patterns to include",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per folder on failure")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token not found in environment (HF_TOKEN or HUGGINGFACE_HUB_TOKEN)")

    api = HfApi(token=token)

    # Ensure repo/branch exist
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    try:
        api.create_branch(repo_id=args.repo_id, repo_type="model", branch=args.branch, exist_ok=True)
    except Exception:
        pass

    paths = find_pretrained_model_dirs(args.base)
    print(f"Found {len(paths)} pretrained_model folders", flush=True)

    uploaded_any = False
    for p in paths:
        # Expect: .../<date>/<run>/checkpoints/<step>/pretrained_model
        try:
            step_dir = p.parent.name
            run_dir = p.parents[2].name
            date_dir = p.parents[3].name
        except Exception:
            print(f"SKIP {p}: unexpected directory structure", flush=True)
            continue

        path_in_repo = f"runs/{date_dir}/{run_dir}/{step_dir}"
        commit_message = f"Add {date_dir}/{run_dir}/{step_dir}"

        # Sanity checks
        has_config = (p / "config.json").exists()
        has_weights = any((p / fname).exists() for fname in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
        has_weights = any((p / fname).exists() for fname in WEIGHT_FILENAMES)
        if not (has_config and has_weights):
            print(f"SKIP {path_in_repo}: missing weights/config in {p}", flush=True)
            continue

        print(f"Uploading {p} -> {args.repo_id}:{path_in_repo}", flush=True)
        # Retry loop for robustness
        last_err: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                upload_folder(
                    folder_path=str(p),
                    repo_id=args.repo_id,
                    repo_type="model",
                    revision=args.branch,
                    path_in_repo=path_in_repo,
                    commit_message=commit_message,
                    allow_patterns=args.allow_patterns,
                    ignore_patterns=["__pycache__", "*.pyc", "README.md"],
                )
                print(f"Done: {path_in_repo}", flush=True)
                uploaded_any = True
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"Attempt {attempt}/{args.retries} failed for {path_in_repo}: {e}", flush=True)
        if last_err is not None:
            print(f"Failed {path_in_repo}: {last_err}", flush=True)

    if not uploaded_any:
        print("No uploads performed.", flush=True)
    else:
        print("All uploads attempted.", flush=True)


if __name__ == "__main__":
    main()


