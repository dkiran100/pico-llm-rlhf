import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import os

from src.model.model import TransformerModel as Transformer
from src.model.model import LSTMSeqModel as LSTM
from src.model.kgram.kgram import KGramMLPSeqModel as KGram
from src.data.squad_preferences import generate_preference_pairs
import tiktoken

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        with torch.no_grad():
            _ = self.base_model(input_ids)  # forward pass to trigger internal state if needed

        last_tokens = input_ids[:, -1]  # (batch,)
        last_hidden = self.base_model.transformer.wte(last_tokens)  # (batch, n_embd)
        rewards = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return rewards


def get_model(model_type, vocab_size):
    if model_type == "transformer":
        model = Transformer(vocab_size=vocab_size)
    elif model_type == "lstm":
        model = LSTM(vocab_size=vocab_size)
    elif model_type == "kgram":
        model = KGram(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def collate_fn(batch, tokenizer, max_length=512):
    def encode(text):
        ids = tokenizer.encode(text)[:max_length]
        return torch.tensor(ids, dtype=torch.long)
    chosen = [encode(p + " " + c) for p, c, _ in batch]
    rejected = [encode(p + " " + r) for p, _, r in batch]

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq
        return padded
    return pad(chosen), pad(rejected)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["transformer", "lstm", "kgram"], default="transformer")
    args = parser.parse_args()

    wandb.init(project="pico-llm-final-run", name=f"reward-{args.model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    pairs = generate_preference_pairs(num_samples=1000)

    dataloader = DataLoader(
        pairs,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2,
        pin_memory=True
    )

    base_model = get_model(args.model_type, vocab_size=50257)

    # ✅ Hardcoded hidden dim based on model type
    if args.model_type == "transformer":
        hidden_dim = 1024
    elif args.model_type == "lstm":
        hidden_dim = 256
    elif args.model_type == "kgram":
        hidden_dim = 512

    reward_model = RewardModel(base_model, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)

    reward_model.train()
    for epoch in range(20):
        total_loss = 0.0
        for chosen, rejected in dataloader:
            chosen = chosen.to(device)
            rejected = rejected.to(device)

            r_chosen = reward_model(chosen)
            r_rejected = reward_model(rejected)

            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Avg Reward Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "reward/loss": avg_loss})

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/reward_model_{args.model_type}.pt"
    torch.save(reward_model.state_dict(), save_path)
    print(f"✅ Saved reward model to: {save_path}")
    wandb.finish()

if __name__ == "__main__":
    main()
