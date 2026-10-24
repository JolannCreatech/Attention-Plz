from torch import nn
from minions_translator.utils import MultiHeadAttention, AddandNorm, FeedForward

class Encoder (nn.Module):
    def __init__(self, i_embedding_dim, num_h, hidden_dim):
        super().__init__()
        self.mha = MultiHeadAttention(i_embedding_dim, num_h)
        self.addnorm1 = AddandNorm(i_embedding_dim)
        self.ffn = FeedForward(i_embedding_dim, hidden_dim)
        self.addnorm2 = AddandNorm(i_embedding_dim)

    def forward(self, x, source_padding_mask = None):
        x1 = self.mha(x, key_padding_mask=source_padding_mask)
        x2 = self.addnorm1(x, x1)
        x3 = self.ffn(x2)
        x4 = self.addnorm2(x2, x3)
        return x4

class Decoder (nn.Module):
    def __init__(self, i_embedding_dim, num_h, hidden_dim):
        super().__init__()
        self.mha1 = MultiHeadAttention(i_embedding_dim, num_h)
        self.addnorm1 = AddandNorm(i_embedding_dim)
        self.mha2 = MultiHeadAttention(i_embedding_dim, num_h)
        self.addnorm2 = AddandNorm(i_embedding_dim)
        self.ffn = FeedForward(i_embedding_dim, hidden_dim)
        self.addnorm3 = AddandNorm(i_embedding_dim)

    def forward(self, x, encoder_output, source_padding_mask = None, target_padding_mask = None):
        x1 = self.mha1(x, masked=True, key_padding_mask=target_padding_mask)
        x2 = self.addnorm1(x, x1)
        x3 = self.mha2(encoder_output, y_decode = x2, key_padding_mask = source_padding_mask)
        x4 = self.addnorm2(x2, x3)
        x5 = self.ffn(x4)
        x6 = self.addnorm3(x4, x5)
        return x6

