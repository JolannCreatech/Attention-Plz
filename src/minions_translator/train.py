
import click
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minions_translator.tst_trn import TranslationDataset, collate_fn
from minions_translator.tst_trn import Tst_Trn
from minions_translator.utils import Tokenizer, choose_device
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = PROJECT_ROOT / "data" / "corpus_francais_minionese.txt"


def train_transformer(text_path, epochs, batch_size, context_length, target_length):
    text = text_path.read_text(encoding="utf-8")
    if not Path(PROJECT_ROOT / "models" / "tokenizer.json").exists():
        click.secho("Le tokenizer n'existe pas. Veuillez d'abord entraîner le tokenizer.", fg = 'red')
        return
    tokenizer = Tokenizer("models/tokenizer.json")
    vocab_size = tokenizer.vocab_size

    blocks = re.split(r"\n\s*\n", text.strip())

    examples = []
    for block in blocks:
        if (
            "TÂCHE :" in block
            and "SOURCE :" in block
            and "CIBLE :" in block
        ):
            examples.append(block.strip())

    if len(examples) < 2:
        raise ValueError(f"Pas assez de paires dans le corpus : {len(examples)}")

    split = int(len(examples) * 0.9)
    train_text = "\n\n".join(examples[:split])
    test_text = "\n\n".join(examples[split:])
    train_dataset = TranslationDataset(
        train_text, tokenizer, max_length=64
    )
    test_dataset = TranslationDataset(
        test_text, tokenizer, max_length=64
    )

    print("Exemples entraînement :", len(train_dataset))
    print("Exemples test :", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    embedding_dim = 256
    num_heads = 8
    hidden_dim = 512
    device = choose_device()

    trainer = Tst_Trn(
        embedding_dim,
        num_heads,
        hidden_dim,
        vocab_size,
        device,
    )

    global_step = 0
    for epoch in range(epochs):
        train_loss, global_step = trainer.train_loop(train_loader, global_step)
        test_loss, test_accuracy = trainer.test_loop(test_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

    checkpoint_dir = PROJECT_ROOT / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "minionese_transformer.pt"

    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": {
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "hidden_dim": hidden_dim,
                "vocab_size": vocab_size,
                "context_length": context_length,
                "target_length": target_length,
            },
        },
        checkpoint_path,
    )

    print(f"Modèle sauvegardé dans : {checkpoint_path}")

def train_tokenizer(text_path, vocab_size = 2000):
    tokenizer = Tokenizer()

    tokenizer.train(
        texts= [text_path.read_text(encoding="utf-8")],
        vocab_size=vocab_size,
    )

    tokenizer.save("models/tokenizer.json")

    print("Vocabulaire :", tokenizer.vocab_size)