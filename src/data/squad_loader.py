
import tiktoken
from datasets import load_dataset
import torch

def load_squad_dataset(tokenizer_name="gpt2", max_length=512):
    enc = tiktoken.get_encoding(tokenizer_name)
    dataset = load_dataset("squad", split="train")
    tokenized_samples = []

    for item in dataset:
        question = item["question"].strip().replace("\n", " ")
        context = item["context"].strip().replace("\n", " ")
        answer = item["answers"]["text"][0].strip().replace("\n", " ")
        prompt = f"Question: {question} Context: {context}"
        input_ids = enc.encode(prompt)
        target_ids = enc.encode(answer)
        if len(input_ids) + len(target_ids) + 1 <= max_length:
            sequence = input_ids + [enc.eot_token] + target_ids
            tokenized_samples.append(torch.tensor(sequence, dtype=torch.long))

    return tokenized_samples
