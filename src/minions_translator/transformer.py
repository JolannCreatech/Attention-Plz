from torch import nn
from minions_translator.coders import Encoder, Decoder
from minions_translator.utils import Embedding, PositionalEncoding

class Transformer (nn.Module):
    def __init__(self, i_embedding_dim, num_h, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = Embedding(vocab_size, i_embedding_dim)
        self.positional_encoding = PositionalEncoding(i_embedding_dim)
        
        self.encoder = Encoder(i_embedding_dim, num_h, hidden_dim)
        self.decoder = Decoder(i_embedding_dim, num_h, hidden_dim)

        self.output_projection = nn.Linear(i_embedding_dim, vocab_size)
    
    def embed(self, token_ids):
        x = self.embedding(token_ids)
        return self.positional_encoding(x)
    
    def forward(self, x, y):
        source_padding_mask = (x == 0)
        target_padding_mask = (y == 0)
        
        source = self.embed(x)
        target = self.embed(y)

        encoder_output = self.encoder(source, source_padding_mask=source_padding_mask)
        decoder_output = self.decoder(target, encoder_output, source_padding_mask=source_padding_mask, target_padding_mask=target_padding_mask)
       
        return self.output_projection(decoder_output)

    
