
import sys
import torch
import logging
from rrl.models import RRL
import shutil
from pathlib import Path

# Mock constants
MODELS_DIR = Path("outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Device
rrl_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {rrl_device}")

# Mock data dimensions
dim_list = [10, 16, 16, 8, 2] # input, layers..., output

def run_loop():
    for i in range(3):
        print(f"Iteration {i+1}")
        
        # Instantiate RRL (this triggers the logging wipe)
        model = RRL(
            dim_list=dim_list,
            device=rrl_device,
            use_not=True,
            is_rank0=True,
            save_best=False,
            distributed=False,
            save_path=MODELS_DIR / "test_model.pth",
            temperature=0.01
        )
        
        print("Model created.")
        
        # Verify logging handlers
        print(f"Root handlers: {logging.root.handlers}")
        
        # Cleanup
        del model
        if rrl_device.type == 'mps':
            torch.mps.empty_cache()

if __name__ == "__main__":
    try:
        run_loop()
        print("Finished successfully.")
    except Exception as e:
        print(f"Crashed: {e}")
