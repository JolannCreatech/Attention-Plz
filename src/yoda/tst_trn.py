import torch 
from torch import nn
from yoda.transormer import Transformer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from torch.utils.data import Dataset
import torch


class TextWindowDataset(Dataset):
    def __init__(
        self,
        token_ids,
        context_length=64,
        target_length=32,
        stride=8,
    ):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.target_length = target_length
        self.stride = stride

        last_start = len(self.tokens) - context_length - target_length
        if last_start < 0:
            raise ValueError("Le texte est trop court pour ces longueurs.")

        self.sample_count = last_start // stride + 1

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        start = index * self.stride
        context_end = start + self.context_length
        target_end = context_end + self.target_length

        source_ids = self.tokens[start:context_end]
        target_ids = self.tokens[context_end:target_end]

        # Le décodeur reçoit le token précédent pour prédire chaque cible.
        decoder_input_ids = torch.cat(
            (source_ids[-1:], target_ids[:-1])
        )

        return source_ids, decoder_input_ids, target_ids



class Tst_Trn ():
    def __init__(self, i_embedding_dim, num_h, hidden_dim, vocab_size, device):
        self.device = device
        self.model = Transformer(i_embedding_dim, num_h, hidden_dim, vocab_size).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
        )

    def train_loop(self, dataloader, global_step=0):
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for x, y, z in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x, y)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                z.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            token_count = z.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count
            global_step += 1

        if total_tokens == 0:
            raise ValueError("Le DataLoader d'entraînement est vide.")

        return total_loss / total_tokens, global_step
    def test_loop(self, dataloader, verbose=True):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        with torch.inference_mode():
            for x,y,z in dataloader:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                logits = self.model(x, y)
                loss = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)), 
                    z.reshape(-1)
                )

                token_count = z.numel()
                total_loss += loss.item() * token_count
                total_tokens += token_count
                correct_tokens += (logits.argmax(dim=-1) == z).sum().item()

        return total_loss / total_tokens, correct_tokens / total_tokens
