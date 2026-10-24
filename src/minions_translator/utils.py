from pathlib import Path

import torch
from torch import nn
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


class Tokenizer:
    SPECIAL_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[FR_TO_MIN]",
        "[MIN_TO_FR]",
        "[SOURCE]",
        "[TARGET]",
    ]

    def __init__(self, tokenizer_path=None):
        self.encoder = None

        if tokenizer_path is not None:
            self.encoder = HFTokenizer.from_file(str(tokenizer_path))

    def train(self, texts, vocab_size=1000):
        self.encoder = HFTokenizer(
            models.BPE(unk_token="[UNK]")
        )

        self.encoder.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        self.encoder.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        self.encoder.train_from_iterator(texts, trainer=trainer)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder.save(str(path))

    def encode(self, text):
        return self.encoder.encode(text).ids

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    @property
    def vocab_size(self):
        return self.encoder.get_vocab_size()


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

    def forward(self, x, y_decode = None, masked = False, key_padding_mask = None):
        if y_decode is not None:
            Q = self.Wq(y_decode)
           
        else :
            Q = self.Wq(x)

        K = self.Wk(x)
        V = self.Wv(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        if masked:
            seq_len = scores.size(-1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, :].to(
                device=scores.device, dtype=torch.bool
            )
            scores = scores.masked_fill(padding_mask, float("-inf"))
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

class MultiHeadAttention (nn.Module):
    def __init__(self, embedding_dim, num_h):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttenion(embedding_dim, num_h) for _ in range(num_h)])
        self.linear = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x, y_decode = None, masked = False, key_padding_mask = None):
        head_outputs = [head(x, y_decode, masked, key_padding_mask) for head in self.heads]
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
