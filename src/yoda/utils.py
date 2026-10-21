from pathlib import Path

import torch
from torch import nn

import numpy as np

import math

import tiktoken
import torch
from torch import nn


class Tokenizer:
    def __init__(
        self
    ) -> None:
        self.encoder = tiktoken.get_encoding("o200k_base")

    def encode(self, text: str):
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]):
        return self.encoder.decode(tokens)

class Embedding (nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding (nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        positions = self.encoding[:, :x.size(1), :].to(
            device=x.device,
            dtype=x.dtype,
        )
        return x + positions

class HeadAttenion (nn.Module):
    def __init__(self, embedding_dim, num_h):
        super().__init__()
        head_dim = embedding_dim // num_h
        self.Wq = nn.Linear(embedding_dim, head_dim)
        self.Wk = nn.Linear(embedding_dim, head_dim)
        self.Wv = nn.Linear(embedding_dim, head_dim)

    def forward(self, x, y_decode = None, masked = False):
        if y_decode is not None:
            Q = self.Wq(y_decode)
           
        else :
            Q = self.Wq(x)

        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        if masked: 
            mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool()
            mask = mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class MultiHeadAttention (nn.Module):
    def __init__(self, embedding_dim, num_h):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttenion(embedding_dim, num_h) for _ in range(num_h)])
        self.linear = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x, y_decode = None, masked = False):
        head_outputs = [head(x, y_decode, masked) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.linear(concatenated)
        return output

class AddandNorm (nn.Module):
    def __init__ (self, embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward (self, X, L):
        N = X + L
        N = self.norm(N)
        return N

class FeedForward (nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
