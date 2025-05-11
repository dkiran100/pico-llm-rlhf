import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from tqdm import tqdm

import wandb  # WandB for logging and monitoring
import os
import matplotlib.pyplot as plt
import pandas as pd
from wandb import Api


def generate_text(model, enc, init_text, max_new_tokens=20, device='cpu', top_p=None):
    """
        A single code path for all models:
        - We keep a growing list 'context_tokens'.
        - At each step, we feed the entire context as (seq_len,1) to model(...).
        - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
        - We pick next token (greedy or top-p), append to context_tokens.
        - Optionally do monosemantic analysis on that newly generated token.
    """
    initial_text = enc.encode(init_text)
    tokens_seq = torch.tensor(initial_text, dtype=torch.long, device=device).unsqueeze(1)
    tokens_seq_generated = model.generate(tokens_seq, max_new_tokens, top_p)[:, 0].tolist()
    final_text = enc.decode(tokens_seq_generated)
    return final_text

def compute_next_token_loss(logits, tokens):
    """
        logits: (seq_len, batch, vocab_size)
        tokens: (seq_len, batch)
        Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    # boundary condition
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)
    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


def plot_loss_vs_steps(run_id, out_dir):
    api = Api()
    run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{run_id}")
    history = run.history(samples=10000)  # or a larger value if needed
    steps = history['_step'].tolist()
    losses = history['loss'].tolist()
    epochs = history['epoch'].tolist()
    model_name = run.config.get("model", "unknown")
    plt.figure(figsize=(8, 6))
    plt.plot(steps, losses, label="Training Loss", color="blue")
    last_epoch = -1
    for s, e in zip(steps, epochs):
        if e != last_epoch:
            plt.axvline(x=s, color='red', linestyle='--', alpha=0.5)
            last_epoch = e
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Step ({model_name}) with Epoch Boundaries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "loss_plot.png")
    plt.savefig(save_path)
    plt.close()
    wandb.run.log({f"Loss Curve": wandb.Image(save_path)})


class Trainer:
    def __init__(self, data, model, config, device):
        self.device = device
        self.data = data
        self.model = model
        self.config = config.algorithm
        self.config_details = config

        self.optimizer = optim.Adam(self.model.model.parameters(), lr=self.config.lr)

    def evaluate(self):
        """ (with a split of training set and test set) evaluate the average loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_idx, batch_tokens in enumerate(self.data.test_dataloader, start=1):
                batch_tokens = batch_tokens.to(self.device)
                logits = self.model.predict(batch_tokens)
                loss = compute_next_token_loss(logits, batch_tokens)
                total_loss += loss.item()
                count += 1
        if count > 0:
            avg_loss = total_loss / count
        else:
            avg_loss = float('inf')
        return avg_loss

    def run(self):
        model_name = self.model.config.type
        epochs = self.config.epochs
        max_steps_per_epoch = self.config.max_steps_per_epoch
        sample_interval = self.config.sample_interval
        enc = self.data.enc
        prompt="Once upon a"

        # === Create output path to save generation logs ===
        run_name = wandb.run.name or wandb.run.id
        out_dir = os.path.join("wandb_output", run_name)
        os.makedirs(out_dir, exist_ok=True)
        text_log_path = os.path.join(out_dir, "all_generations.txt")

        # Logs table per epoch (start & end)
        generation_table = wandb.Table(columns=["epoch", "stage", "method", "text"])

        # For saving to .txt at the end
        all_generated_texts = []

        epoch_train_losses = []
        epoch_test_losses = []

        start_time = time.time()
        next_sample_time = start_time
        global_step = 0

        for epoch in range(1, epochs + 1):
            # === Generate & log text at epoch START ===
            for stage in ["start"]:
                for method, top_p in [("greedy", None), ("top_p_0.95", 0.95), ("top_p_1.0", 1.0)]:
                    text = generate_text(self.model, self.data.enc, prompt, max_new_tokens=20, device=self.device, top_p=top_p)
                    generation_table.add_data(epoch, stage, method, text)
    
            self.model.train()
            total_loss = 0.0
            partial_loss = 0.0
            partial_count = 0
            step_in_epoch = 0
            pbar = tqdm(self.data.train_dataloader)
            for batch_idx, batch_tokens in enumerate(pbar, start=1):
                step_in_epoch += 1
                global_step += 1
                batch_tokens = batch_tokens.to(self.device)  # (seq_len, batch)

                logits = self.model.predict(batch_tokens)  # (seq_len, batch, vocab_size)
                loss = compute_next_token_loss(logits, batch_tokens)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                partial_loss += loss.item()
                partial_count += 1
                # WandB log loss
                wandb.log({
                    "loss": loss.item(),
                    "step": global_step,
                    "epoch": epoch
                })
                
                if batch_idx % self.config.log_steps == 0:
                    avg_part_loss = partial_loss / partial_count
                    tqdm.write(f"[{model_name}] Epoch {epoch}/{epochs}, "
                          f"Step {batch_idx}/{len(self.data.train_dataloader)} (global step: {global_step}) "
                          f"Partial Avg Loss: {avg_part_loss:.4f}")
                    partial_loss = 0.0
                    partial_count = 0
                current_time = time.time()
                if current_time >= next_sample_time and enc is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for method, top_p in [("greedy", None), ("top_p_0.95", 0.95), ("top_p_1.0", 1.0)]:
                            text = generate_text(self.model, self.data.enc, prompt, max_new_tokens=20, device=self.device, top_p=top_p)
                            log_line = f"[{model_name}] Generating Sample text ({method}) at epoch={epoch}, step={batch_idx}: {text}"
                            all_generated_texts.append(log_line+"\n")
                            tqdm.write(log_line)
                        all_generated_texts.append("------------------------------\n")
                    next_sample_time = current_time + sample_interval
                    self.model.train()
                if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                    print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                    break
            pbar.close()
            avg_train_loss = total_loss / step_in_epoch
            print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Train Loss: {avg_train_loss:.4f}")

            # eval on test data after each epoch
            test_loss = self.evaluate()
            print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Test Loss: {test_loss:.4f}")
            wandb.log({"test_loss": test_loss, "train_loss": avg_train_loss})

            epoch_train_losses.append(avg_train_loss)
            epoch_test_losses.append(test_loss)

            # === Generate & log text at epoch END ===
            for stage in ["end"]:
                for method, top_p in [("greedy", None), ("top_p_0.95", 0.95), ("top_p_1.0", 1.0)]:
                    text = generate_text(self.model, enc, prompt, top_p=top_p, device=self.device)
                    generation_table.add_data(epoch, stage, method, text)

        # Final WandB logging
        wandb.log({"epoch_generations": generation_table})
        # Save the generation log to a text file
        with open(text_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_generated_texts))
        artifact = wandb.Artifact("generated_text_log", type="text")
        artifact.add_file(text_log_path)
        wandb.log_artifact(artifact)

        loss_table = wandb.Table(columns=["epoch", "train_loss", "test_loss"])
        for i in range(epochs):
            loss_table.add_data(i + 1, epoch_train_losses[i], epoch_test_losses[i])
        loss_chart = wandb.plot.line(loss_table, "epoch", ["train_loss", "test_loss"],
                                    title="Train-Test Loss Curve")
        wandb.run.log({"train_test_loss_curve": loss_chart})

        plot_loss_vs_steps(wandb.run.id, out_dir)
        wandb.finish()
        print(f"[{model_name}] Training completed. Check WandB for logs and generated text.")