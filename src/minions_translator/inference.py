from pathlib import Path

import torch

from minions_translator.transformer import Transformer
from minions_translator.utils import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "minionese_transformer.pt"


def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint["config"]

    model = Transformer(
        config["embedding_dim"],
        config["num_heads"],
        config["hidden_dim"],
        config["vocab_size"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = PROJECT_ROOT / "models" / "tokenizer.json"
    tokenizer = Tokenizer(tokenizer_path)

    return model, tokenizer, device, config




def translate(model, tokenizer, device, tominion: bool, text: str) -> str:
    model.eval()

    # 1. Choisir le sens de traduction
    direction = "[FR_TO_MIN]" if tominion else "[MIN_TO_FR]"

    # 2. Encoder la phrase source
    source_ids = [
        tokenizer.encoder.token_to_id(direction),
        tokenizer.encoder.token_to_id("[SOURCE]"),
    ] + tokenizer.encode(text)

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device,
    )

    # 3. Commencer la génération avec [BOS]
    bos_id = tokenizer.encoder.token_to_id("[BOS]")
    eos_id = tokenizer.encoder.token_to_id("[EOS]")

    generated = [bos_id]
    max_length = 64

    with torch.inference_mode():
        for _ in range(max_length - 1):
            decoder_input = torch.tensor(
                [generated],
                dtype=torch.long,
                device=device,
            )

            # Prédire le prochain token
            logits = model(source, decoder_input)
            next_token = logits[0, -1, :].argmax().item()

            # Arrêter à la fin de la phrase
            if next_token == eos_id:
                break

            generated.append(next_token)

    # 4. Décoder les tokens en texte
    return tokenizer.decode(generated[1:])
