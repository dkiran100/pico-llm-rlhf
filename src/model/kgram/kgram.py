import torch
import torch.nn as nn
import torch.nn.functional as F

class KGramMLPSeqModel(nn.Module):
    def __init__(self, vocab_size, kgram_k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.k = kgram_k
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        input_dim = self.k * self.vocab_size
        layers = []
        layers.append(nn.Linear(input_dim, self.embed_size))
        layers.append(nn.SiLU())
        
        for _ in range(self.num_inner_layers - 1):
            layers.append(nn.Linear(self.embed_size, self.embed_size))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(self.embed_size, self.vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        Args:
            tokens_seq: (seq_len, batch) containing token ids.
        Returns:
            (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        padded = torch.nn.functional.pad(tokens_seq, (0, 0, self.k - 1, 0), mode='constant', value=0)
        windows = padded.unfold(0, self.k, 1)
        windows_onehot = torch.nn.functional.one_hot(windows, num_classes=self.vocab_size)
        windows_flat = windows_onehot.view(seq_len, batch_size, self.k * self.vocab_size).float()
        windows_flat = windows_flat.view(-1, self.k * self.vocab_size)
        logits = self.net(windows_flat) # (seq_len * batch, vocab_size)
        logits = logits.view(seq_len, batch_size, self.vocab_size) # (seq_len, batch, vocab_size)
        return logits