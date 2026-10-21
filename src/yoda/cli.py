import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yoda.tst_trn import TextWindowDataset
from yoda.tst_trn import Tst_Trn
from yoda.utils import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = PROJECT_ROOT / "data" / "texte_sage_syntaxe_inversee.txt"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(text_path, epochs, batch_size, context_length, target_length):
    text = text_path.read_text(encoding="utf-8")
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)
    vocab_size = tokenizer.encoder.n_vocab  

    split = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split]
    test_ids = token_ids[split:]

    train_dataset = TextWindowDataset(
        train_ids, context_length=context_length, target_length=target_length
    )
    test_dataset = TextWindowDataset(
        test_ids, context_length=context_length, target_length=target_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
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


def main():
    parser = argparse.ArgumentParser(description="Entraîner Yoda")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--target-length", type=int, default=32)
    args = parser.parse_args()

    train(
        args.text,
        args.epochs,
        args.batch_size,
        args.context_length,
        args.target_length,
    )


if __name__ == "__main__":
    main()