import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb
import os
from src.model.model import TransformerModel as Transformer
from src.data.squad_preferences import generate_preference_pairs
import tiktoken

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        with torch.no_grad():
            _ = self.base_model(input_ids)
        last_tokens = input_ids[:, -1]
        last_hidden = self.base_model.transformer.wte(last_tokens)
        rewards = self.reward_head(last_hidden).squeeze(-1)
        return rewards

def generate_response(model, input_ids, max_new_tokens=20, top_p=0.9):
    model.eval()
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated)
        logits = logits[:, -1, :]  # (batch, vocab)
        probs = F.softmax(logits, dim=-1)

        # Top-p filtering
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = torch.clamp(sorted_probs, min=0.0)

            sum_probs = sorted_probs.sum(dim=-1, keepdim=True)
            sum_probs[sum_probs == 0] = 1e-8  # avoid divide-by-zero
            sorted_probs = sorted_probs / sum_probs

            # Fallback if probs are invalid
            if torch.any(torch.isnan(sorted_probs)) or torch.any(sorted_probs.sum(dim=-1) <= 0):
                print("⚠️ Invalid distribution detected! Using fallback argmax.")
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_token)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # Append next token
        generated = torch.cat([generated, next_token], dim=1)

        # Trim to block size
        if generated.size(1) > model.block_size:
            generated = generated[:, -model.block_size:]

    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["transformer"], default="transformer")
    args = parser.parse_args()

    wandb.init(project="pico-llm-final-run", name="ppo-transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model = Transformer(vocab_size=50257).to(device)
    reward_head = RewardModel(Transformer(vocab_size=50257), hidden_dim=1024).to(device)
    reward_ckpt = "checkpoints/reward_model_transformer.pt"
    reward_head.load_state_dict(torch.load(reward_ckpt))

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    pref_pairs = generate_preference_pairs(num_samples=1000)
    prompts = [p for p, _, _ in pref_pairs]
    dataloader = DataLoader(prompts, batch_size=4, shuffle=True)

    for epoch in range(1, 21):
        model.train()
        total_ppo_loss = 0.0
        total_reward = 0.0
        count = 0

        for batch_prompts in dataloader:
            batch = [torch.tensor(tokenizer.encode(p), dtype=torch.long) for p in batch_prompts]
            max_len = max(len(x) for x in batch)
            input_ids = torch.stack([F.pad(x, (0, max_len - len(x))) for x in batch]).to(device)

            generated_ids = generate_response(model, input_ids, max_new_tokens=20)

            rewards = reward_head(generated_ids).detach()
            baseline = rewards.mean()
            advantages = rewards - baseline

            logits = model(generated_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_act = log_probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
            log_probs_act = log_probs_act.sum(dim=1)

            ppo_loss = -(advantages * log_probs_act).mean()

            optimizer.zero_grad()
            ppo_loss.backward()
            optimizer.step()

            total_ppo_loss += ppo_loss.item()
            total_reward += rewards.mean().item()
            count += 1

        avg_ppo_loss = total_ppo_loss / count
        avg_reward = total_reward / count
        print(f"Epoch {epoch} - PPO Loss: {avg_ppo_loss:.4f} - Avg Reward: {avg_reward:.4f}")
        wandb.log({
            "epoch": epoch,
            "ppo/loss": avg_ppo_loss,
            "ppo/reward": avg_reward
        })

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/transformer_ppo_finetuned.pt")
    print("✅ Saved PPO fine-tuned model to checkpoints/transformer_ppo_finetuned.pt")
    wandb.finish()

if __name__ == "__main__":
    main()

