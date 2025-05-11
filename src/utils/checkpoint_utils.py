
import os
import torch

def save_checkpoint(model, output_dir, name="model.pt"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, name)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")

def load_checkpoint(model, checkpoint_path, device="cpu"):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"🔁 Loaded model from: {checkpoint_path}")
    return model
