import torch
from datasets import load_dataset
import tiktoken
from data.dataset import MixedSequenceDataset
from torch.utils.data import random_split

def seq_collate_fn(batch):
    """
        batch: list of 1D LongTensors of various lengths [<= block_size].
        1) find max length
        2) pad with zeros
        3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded

class TextData:
    def __init__(self, config):
        self.config = config.data

        tinystories_seqs = []
        other_seqs = []

        if config.data.tinystories_weight > 0.0:
            print(f"Loading TinyStories from huggingface with weight={config.data.tinystories_weight}...")
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            dataset = dataset.select(range(config.data.train_subset_size))
        else:
            print("TinyStories weight=0 => skipping TinyStories.")
            dataset = None

        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab
        print(f"Vocab size: {vocab_size}")

        # only considering first block size tokens (not efficient)
        block_size = config.data.block_size
        if dataset is not None:
            for sample in dataset:
                text = sample['text']
                tokens = enc.encode(text)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    tinystories_seqs.append(tokens)
            print(f"TinyStories sequences: {len(tinystories_seqs)}")

        if config.data.input_files:
            for filepath in config.data.input_files:
                print(f"Reading custom text file: {filepath}")
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = enc.encode(line)
                    tokens = tokens[:block_size]
                    if len(tokens) > 0:
                        other_seqs.append(tokens)
            print(f"Custom input files: {len(other_seqs)} sequences loaded.")
        else:
            print("No custom input files provided.")

        p_tiny = config.data.tinystories_weight
        if len(tinystories_seqs) == 0 and p_tiny>0:
            print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
        combined_dataset = MixedSequenceDataset(
            tinystories_seqs=tinystories_seqs,
            other_seqs=other_seqs,
            p_tiny=p_tiny
        )

        test_split = config.data.test_split  # 20%? may need adjusting
        total_len = len(combined_dataset)
        test_len = int(total_len * test_split)
        train_len = total_len - test_len
        train_dataset, test_dataset = random_split(combined_dataset, [train_len, test_len])

        self.dataset = combined_dataset
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=seq_collate_fn
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=seq_collate_fn
        )
        self.enc = enc
        self.vocab_size = vocab_size