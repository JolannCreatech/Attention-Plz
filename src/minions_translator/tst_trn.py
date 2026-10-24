import torch 
from torch import nn
from minions_translator.transformer import Transformer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from torch.utils.data import Dataset
import torch

import re
import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        pattern = re.compile(
            r"TÂCHE\s*:\s*(.*?)\s*\n"
            r"SOURCE\s*:\s*(.*?)\s*\n"
            r"CIBLE\s*:\s*(.*?)(?=\n\s*\n|\Z)",
            re.DOTALL,
        )

        for task, source, target in pattern.findall(text):
            task = task.strip()
            source = source.strip()
            target = target.strip()

            if not source or not target:
                continue

            if "Français -> Minionese" in task:
                direction = "[FR_TO_MIN]"
            elif "Minionese -> Français" in task:
                direction = "[MIN_TO_FR]"
            else:
                continue

            source_ids = (
                [tokenizer.encoder.token_to_id(direction),
                 tokenizer.encoder.token_to_id("[SOURCE]")]
                + tokenizer.encode(source)
            )

            target_ids = (
                [tokenizer.encoder.token_to_id("[BOS]")]
                + tokenizer.encode(target)
                + [tokenizer.encoder.token_to_id("[EOS]")]
            )

            if (
                len(source_ids) <= max_length
                and len(target_ids) <= max_length
            ):
                self.examples.append((source_ids, target_ids))


        print("Texte reçu par TranslationDataset :")
        print(repr(text[:500]))
        print("Nombre de blocs reconnus :", len(pattern.findall(text)))

        if not self.examples:
            raise ValueError(
                "Aucune paire trouvée. Vérifie le format du corpus."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        source_ids, target_ids = self.examples[index]

        decoder_input = target_ids[:-1]
        target_output = target_ids[1:]

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target_output, dtype=torch.long),
        )
    
def collate_fn(batch, pad_id=0):
    sources, decoder_inputs, targets = zip(*batch)

    sources = torch.nn.utils.rnn.pad_sequence(
        sources, batch_first=True, padding_value=pad_id
    )
    decoder_inputs = torch.nn.utils.rnn.pad_sequence(
        decoder_inputs, batch_first=True, padding_value=pad_id
    )
    targets = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=pad_id
    )

    return sources, decoder_inputs, targets




class Tst_Trn ():
    def __init__(self, i_embedding_dim, num_h, hidden_dim, vocab_size, device):
        self.device = device
        self.model = Transformer(i_embedding_dim, num_h, hidden_dim, vocab_size).to(device)
        pad_id = 0  # À condition que [PAD] ait bien l'identifiant 0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
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

            token_count = (z != 0).sum().item()
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

                token_count = (z != 0).sum().item()
                total_loss += loss.item() * token_count
                total_tokens += token_count
                mask = z != 0
                correct_tokens += (
                    (logits.argmax(dim=-1) == z) & mask
                ).sum().item()

        return total_loss / total_tokens, correct_tokens / total_tokens
