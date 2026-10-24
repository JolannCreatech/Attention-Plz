import click
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from minions_translator.train import train_transformer, train_tokenizer
from minions_translator.utils import Tokenizer
from minions_translator.inference import load_model, translate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = PROJECT_ROOT / "data" / "corpus_francais_minionese.txt"

@click.command()
@click.option("--mode", type=click.Choice(["train_transformer", "train_tokenizer", "inferance"]), default="train_transformer", help="Mode d'exécution : 'train_transformer' pour l'entraînement du modele, 'train_tokenizer' pour l'entrainement du tokenizer, 'inferance' pour l'inférence.")
@click.option("--text", type=click.Path(exists=True), default=DEFAULT_TEXT, help="Chemin vers le fichier texte.")
@click.option("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement.")
@click.option("--batch_size", type=int, default=32, help="Taille du lot pour l'entraînement.")
@click.option("--context_length", type=int, default=64, help="Longueur du contexte pour l'entrée du modèle.")
@click.option("--target_length", type=int, default=32, help="Longueur de la cible pour l'entrée du modèle.")
def main(mode, text, epochs, batch_size, context_length, target_length): 

    if mode == "train_transformer":
        train_transformer(
            Path(text),
            epochs,
            batch_size,
            context_length,
            target_length,
        )
    if mode == "train_tokenizer":
        train_tokenizer(
            Path(text),
            vocab_size=2000,
        )

    if mode == "inferance":
        model, tokenizer, device, config = load_model()

        while True:
            tominion = click.confirm("Traduire vers le minions ? (Sinon traduire vers le français)", default=True)
            text = click.prompt("Entrez le texte à traduire")
            translation = translate(model, tokenizer, device, tominion, text)
            click.echo(f"Traduction : {translation}")



if __name__ == "__main__":
    main()