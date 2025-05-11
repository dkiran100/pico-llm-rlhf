import torch
import torch.nn.functional as F
from model.lstm import LSTMSeqModel
from model.transformer import TransformerModel
from model.kgram import KGramMLPSeqModel

def nucleus_sampling(logits, p=0.95):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    k = torch.sum(cum_probs < p).item() + 1
    top_probs = sorted_probs[:k]
    top_indices = sorted_indices[:k]
    top_probs = top_probs / top_probs.sum()
    sampled_index = torch.multinomial(top_probs, 1).item()

    return top_indices[sampled_index].item()


class SequenceModel:
    def __init__(self, config, vocab_size, device):
        self.config = config.model
        self.vocab_size = vocab_size
        self.device = device

        if self.config.type == 'lstm':
            self.model = LSTMSeqModel(
                vocab_size=vocab_size,
                embed_size=self.config.embed_size,
                hidden_size=self.config.embed_size
            )
        elif self.config.type == 'mlp':
            self.model = KGramMLPSeqModel(
                vocab_size=vocab_size,
                kgram_k=self.config.kgram_k,
                embed_size=self.config.embed_size,
                num_inner_layers=self.config.num_inner_layers,
                chunk_size=self.config.kgram_chunk_size
            )
        elif self.config.type == 'transformer':
            self.model = TransformerModel(
                vocab_size=vocab_size,
                block_size=self.config.block_size,
                n_layer=self.config.n_layer,
                n_embd=self.config.n_embd,
                n_head=self.config.n_head
            )

            # crop block size
            if config.data.block_size < config.model.block_size:
                print("Cropping block size.")
                self.config.block_size = config.data.block_size
                self.model.crop_block_size(config.data.block_size)

        self.model = self.model.to(device)

        if config.system.to_compile:
            self.model = torch.compile(self.model)
            print('Model Compiled.')

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def predict(self, tokens_seq):
        return self.model(tokens_seq)

    def generate(self, tokens_seq, max_new_tokens, top_p):
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self.model(tokens_seq) # (seq_len,1,vocab_size)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[-1, 0, :]
            # optionally crop the logits to only the top k options
            if top_p is not None:
                next_token_id = nucleus_sampling(logits, p=top_p)
            else:
                next_token_id = torch.argmax(logits).item()

            next_token = torch.tensor([[next_token_id]], device=tokens_seq.device)
            
            # append sampled index to the running sequence and continue
            tokens_seq = torch.cat((tokens_seq, next_token), dim=0)
        return tokens_seq



