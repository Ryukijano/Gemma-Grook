#!/usr/bin/env python3
"""
Script to upload the Gemma-GR00T model to Hugging Face Hub.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, ModelCard, upload_folder
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError

def upload_to_hub(
    model_path: str,
    repo_id: str = "Ryukijano/gemma-groot",
    private: bool = True,
    commit_message: str = "Add Gemma-GR00T model weights",
    path_in_repo: str | None = None,
):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to the directory containing model files
        repo_id: Repository ID on Hugging Face Hub (username/repo_name)
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    # Validate repository ID
    try:
        validate_repo_id(repo_id)
    except ValueError as e:
        print(f"Invalid repository ID: {e}")
        return
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Created repository: {repo_id}")
    except HfHubHTTPError as e:
        print(f"Repository already exists or error creating: {e}")
    
    # Upload all files in the model directory
    print(f"Uploading model files from {model_path}...")
    
    # Upload files using upload_folder
    # NOTE: We ignore README.md so weight uploads don't overwrite the Hub model card
    upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        path_in_repo=path_in_repo,
        ignore_patterns=["__pycache__", "*.pyc", "README.md"],
    )
    
    print(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
    parser.add_argument('--model_path', type=str, default='exported_weights',
                      help='Path to the directory containing model files')
    parser.add_argument('--repo_id', type=str, default='Ryukijano/gemma-groot',
                      help='Repository ID on Hugging Face Hub (username/repo_name)')
    parser.add_argument('--private', action='store_true', default=True,
                      help='Whether the repository should be private')
    parser.add_argument('--commit_message', type=str, 
                      default='Add Gemma-GR00T model weights',
                      help='Commit message for the upload')
    parser.add_argument('--path_in_repo', type=str, default=None,
                      help='Optional subfolder in the Hub repo to upload into')
    
    args = parser.parse_args()
    
    upload_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        path_in_repo=args.path_in_repo,
    )
