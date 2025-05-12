import argparse
import torch
from src.model.model import TransformerModel as Transformer
from src.model.model import LSTMSeqModel as LSTM
from src.model.kgram.kgram import KGramMLPSeqModel as KGram
import tiktoken

# Sample SQuAD-style prompts
PROMPTS = [
    "Question: What is the capital of France? Context: France is a country in Western Europe. Its capital is Paris.",
    "Question: Who wrote the novel '1984'? Context: George Orwell is best known for writing the novel '1984'.",
    "Question: What is photosynthesis? Context: Photosynthesis is a process by which plants make food using sunlight."
]

def get_model(model_type, vocab_size):
    if model_type == "transformer":
        return Transformer(vocab_size=vocab_size)
    elif model_type == "lstm":
        return LSTM(vocab_size=vocab_size)
    elif model_type == "kgram":
        return KGram(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def generate_completion(model, prompt, tokenizer, model_type, max_new_tokens=30):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if model_type == "kgram":
            logits = model(input_tensor.transpose(0, 1)).transpose(0, 1)
        else:
            logits = model(input_tensor)

        next_token_logits = logits[0, -1, :]
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).item()
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

        if input_tensor.size(1) > 2 and input_tensor[0, -1] == input_tensor[0, -2]:
            break

        if next_token == tokenizer.eot_token:
            break

    return tokenizer.decode(input_tensor[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["transformer", "lstm", "kgram"], default="transformer")
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")
    model = get_model(args.model_type, vocab_size=50257).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Completions from {args.model_type.upper()} Model ===")
    for prompt in PROMPTS:
        output = generate_completion(model, prompt, tokenizer, args.model_type)
        print(f"\nPrompt: {prompt}\nModel Answer: {output}")

if __name__ == "__main__":
    main()

