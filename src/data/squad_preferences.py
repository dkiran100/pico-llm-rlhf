
import random
from datasets import load_dataset

def generate_preference_pairs(split="train", num_samples=1000, seed=42):
    random.seed(seed)
    dataset = load_dataset("squad", split=split)
    preference_pairs = []

    for item in dataset.select(range(num_samples)):
        question = item["question"].strip().replace("\n", " ")
        context = item["context"].strip().replace("\n", " ")
        correct = item["answers"]["text"][0].strip().replace("\n", " ")
        corrupt_type = random.choice(["shuffle", "truncate", "random_span", "garbage"])
        if corrupt_type == "shuffle":
            tokens = correct.split()
            random.shuffle(tokens)
            bad = " ".join(tokens)
        elif corrupt_type == "truncate":
            bad = correct[: max(1, len(correct) // 2)]
        elif corrupt_type == "random_span":
            words = context.split()
            if len(words) > 3:
                start = random.randint(0, len(words) - 2)
                end = min(len(words), start + random.randint(1, 3))
                bad = " ".join(words[start:end])
            else:
                bad = "unknown"
        else:
            bad = "lorem ipsum dolor"

        prompt = f"Question: {question} Context: {context}"
        preference_pairs.append((prompt, correct, bad))

    return preference_pairs
