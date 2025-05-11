
import torch
from torch.utils.data import Dataset
from .squad_loader import load_squad_dataset

class SQuADDataset(Dataset):
    def __init__(self, tokenizer_name="gpt2", max_length=512):
        self.samples = load_squad_dataset(tokenizer_name=tokenizer_name, max_length=max_length)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
