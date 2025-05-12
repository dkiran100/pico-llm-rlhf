
# 🧠 Pico-LLM RLHF Pipeline

This project extends a custom decoder-only language model (Pico-LLM) with a full **Reinforcement Learning from Human Feedback (RLHF)** pipeline on the SQuAD dataset.

It includes:

- ✅ Supervised fine-tuning on question-answer pairs  
- ✅ Direct Preference Optimization (DPO)  
- ✅ Reward model training  
- ✅ PPO-style RLHF  
- ✅ Evaluation and checkpointed generation  
- ✅ Modular training across Transformer, LSTM, and KGram models

---

## 📦 Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install tiktoken datasets
```

---

## 🧪 Training Pipeline

> Run all commands from your project root. Use `PYTHONPATH=src` to access modules properly.

---

### 1. Fine-tune on SQuAD (supervised)

```bash
PYTHONPATH=src python main_squad.py
```

This will:
- Load SQuAD QA pairs
- Train a Transformer model to predict answer tokens
- Save intermediate results

---

### 2. Direct Preference Optimization (DPO)

```bash
PYTHONPATH=src python train_dpo.py --model_type transformer
```

Other options:
```bash
--model_type lstm
--model_type kgram
```

This trains the model using:
- `(prompt, chosen, rejected)` preference pairs
- A DPO loss that encourages preferring good completions

---

### 3. Train a Reward Model

```bash
PYTHONPATH=src python train_reward_model.py --model_type transformer
```

This wraps your model in a scalar output head and trains it to score preferred responses higher than rejected ones.

---

### 4. RLHF with PPO

```bash
PYTHONPATH=src python train_ppo.py --model_type transformer
```

This uses PPO to fine-tune the model:
- Generates completions
- Gets scores from reward model
- Adjusts policy with reward-weighted log-likelihood

---

## 🔍 Evaluation & Generation

### Evaluate DPO-trained model

```bash
PYTHONPATH=src python eval_dpo.py --model_type transformer
```

This runs the trained model on sample prompts and prints completions.

---

### Generate from checkpoint

```bash
PYTHONPATH=src python generate_from_checkpoint.py \
  --model_type transformer \
  --checkpoint checkpoints/transformer_train_dpo/final.pt
```

---

## 📁 Folder Structure

```
pico-llm-rlhf/
├── main_squad.py                # Supervised QA fine-tuning
├── train_dpo.py                 # DPO preference training
├── train_reward_model.py        # Scalar reward model
├── train_ppo.py                 # PPO fine-tuning
├── eval_dpo.py                  # Output evaluation
├── generate_from_checkpoint.py  # Checkpoint-based inference
├── checkpoints/                 # Saved models
├── requirements.txt
└── src/
    ├── data/
    │   ├── squad_loader.py
    │   ├── squad_dataset.py
    │   └── squad_preferences.py
    ├── model/
    │   ├── model.py             # Registry for model classes
    │   ├── transformer/
    │   ├── lstm/
    │   └── kgram/
    └── utils/
        └── checkpoint_utils.py
```

---

## ✅ Tips & Notes

- Always run with `PYTHONPATH=src` to avoid module import errors  
- Use `--model_type transformer` unless testing others  
- You can tweak SQuAD dataset size or corruptions in `squad_preferences.py`  
- All training scripts save checkpoints automatically  

---
