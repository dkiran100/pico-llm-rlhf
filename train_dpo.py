import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tiktoken

from src.model.model import TransformerModel as Transformer
from src.model.model import LSTMSeqModel as LSTM
from src.model.kgram.kgram import KGramMLPSeqModel as KGram
from src.data.squad_preferences import generate_preference_pairs

class DpoPreferenceDataset(Dataset):
    def __init__(self, tokenizer_name="gpt2", max_length=512, num_samples=1000):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_length = max_length
        self.pairs = generate_preference_pairs(num_samples=num_samples)
        self.data = []

        for prompt, chosen, rejected in self.pairs:
            chosen_ids = self.tokenizer.encode(prompt + " " + chosen)
            rejected_ids = self.tokenizer.encode(prompt + " " + rejected)
            if len(chosen_ids) < max_length and len(rejected_ids) < max_length:
                self.data.append((torch.tensor(chosen_ids), torch.tensor(rejected_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    def pad(seq, max_len):
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:len(seq)] = seq
        return padded

    max_len = max(max(len(c), len(r)) for c, r in batch)
    chosen_batch = torch.stack([pad(c, max_len) for c, _ in batch])
    rejected_batch = torch.stack([pad(r, max_len) for _, r in batch])
    return chosen_batch, rejected_batch

def compute_dpo_loss(model, chosen, rejected, model_type):
    logits_chosen = model(chosen)
    logits_rejected = model(rejected)

    if model_type == "kgram":
        logits_chosen = logits_chosen.transpose(0, 1)
        logits_rejected = logits_rejected.transpose(0, 1)

    log_probs_chosen = F.log_softmax(logits_chosen, dim=-1)
    log_probs_rejected = F.log_softmax(logits_rejected, dim=-1)

    log_prob_sum_chosen = torch.sum(log_probs_chosen.gather(-1, chosen.unsqueeze(-1)).squeeze(-1), dim=1)
    log_prob_sum_rejected = torch.sum(log_probs_rejected.gather(-1, rejected.unsqueeze(-1)).squeeze(-1), dim=1)

    logits_diff = log_prob_sum_chosen - log_prob_sum_rejected
    loss = -torch.log(torch.sigmoid(logits_diff)).mean()
    return loss

def get_model(model_type, vocab_size):
    if model_type == "transformer":
        return Transformer(vocab_size=vocab_size)
    elif model_type == "lstm":
        return LSTM(vocab_size=vocab_size)
    elif model_type == "kgram":
        return KGram(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["transformer", "lstm", "kgram"], default="transformer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DpoPreferenceDataset(tokenizer_name="gpt2", max_length=512, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = get_model(args.model_type, vocab_size=50257)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for chosen, rejected in dataloader:
            chosen = chosen.to(device)
            rejected = rejected.to(device)

            loss = compute_dpo_loss(model, chosen, rejected, args.model_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} - Avg DPO Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    main()
