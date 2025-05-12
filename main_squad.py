import torch
from torch.utils.data import DataLoader
from src.data.squad_dataset import SQuADDataset
from src.model.model import TransformerModel as Transformer
import torch.nn.functional as F

def collate_fn(batch):
    max_len = max(seq.size(0) for seq in batch)
    batch_size = len(batch)
    padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :seq.size(0)] = seq
    return padded

def main():
    batch_size = 8
    max_length = 512
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the SQuAD dataset
    dataset = SQuADDataset(tokenizer_name="gpt2", max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize Transformer model
    model = Transformer(vocab_size=50257)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
