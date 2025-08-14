import os
import numpy as np
import torch
from tqdm import tqdm
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset

def test_model():
    # Set device - will automatically use available GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Configuration - use absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    MODEL_PATH = os.path.join(ROOT_DIR, "exported_weights")
    DATASET_PATH = os.path.join(ROOT_DIR, "demo_data", "robot_sim.PickNPlace")
    
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset path: {DATASET_PATH}")
    
    # Use string literal instead of EmbodimentTag enum
    EMBODIMENT_TAG = "gr1"  # Must match the metadata.json exactly
    
    # Load data configuration
    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Load dataset
    print("Loading dataset...")
    
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        embodiment_tag=EmbodimentTag.GR1,
    )
    
    # Load model
    print("Loading model...")
    try:
        policy = Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config,
            modality_transform=modality_transform,
            device=device,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model path exists: {os.path.exists(MODEL_PATH)}")
        print(f"Metadata path exists: {os.path.exists(os.path.join(MODEL_PATH, 'experiment_cfg', 'metadata.json'))}")
        raise
    
    # Set model to evaluation mode
    policy.model.eval()
    
    # Test on a few samples
    num_samples = min(5, len(dataset))  # Test on up to 5 samples
    print(f"Testing on {num_samples} samples...")
    
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}")
        
        # Get sample from dataset
        sample = dataset[i]
        
        # Prepare input (move to device)
        inputs = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Run inference
        with torch.no_grad():
            try:
                output = policy.model(**inputs)
                print("Inference successful!")
                print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
                
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                continue

if __name__ == "__main__":
    test_model()
