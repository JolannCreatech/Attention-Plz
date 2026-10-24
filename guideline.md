# Guideline — faire marcher l'inférence

> Tout ce qui suit a été exécuté sur ce dépôt (macOS, MPS, 8 Go). Les chiffres sont des mesures, pas des estimations.
> Les numéros de ligne sont ceux des fichiers **avant** modification.

---

## 0. Ce que tu obtiens à la fin

```bash
uv run yoda --generate "Le chat cherche la sortie."
# Da meow-meow looka for da exit-o.

uv run yoda --generate "Da meow-meow looka for da exit-o." --sens minion-fr
# Le chat cherche la sortie.
```

Mesures faites avec le checkpoint produit par cette guideline :

| test | résultat |
|---|---|
| paires du corpus vues à l'entraînement | **50/50 exactes** |
| paires de validation (jamais vues) | 2/25 exactes |
| taille du checkpoint | 6,6 Mo |
| paramètres du modèle | 1,72 M |
| durée d'un appel CLI | ~1,5 s (import de torch ~0,9 s + chargement 0,32 s + génération 0,3 s) |

---

## 1. Pourquoi ça ne marche pas aujourd'hui

Trois blocages, dans l'ordre où ils se présentent :

**1. Rien n'est sauvegardé.** `cli.py` entraîne puis jette le modèle. Il n'y a aucun `checkpoints/` dans le dépôt : impossible de recharger quoi que ce soit.

**2. Le modèle ne sait pas produire une suite.** `Transformer` n'a que `forward(x, y)`, qui renvoie des logits pour une séquence `y` déjà connue. Il faut une méthode qui choisit un token, le rajoute, et recommence.

**3. Le jeu de données n'apprend pas la traduction.** `TextWindowDataset` découpe le texte en fenêtres de 64 tokens et apprend à prédire les 32 suivants. Sur un corpus de paires mis bout à bout, le modèle apprend « continuer le fichier », pas « lire SOURCE et écrire CIBLE ».

Le point 3 est le vrai piège, et c'est mesurable :

| jeu de données | epochs | paires exactes |
|---|---|---|
| corpus mis à plat (TÂCHE / SOURCE / CIBLE concaténés) | 60 | **7/58 (12 %)** |
| une paire = un exemple | 60 | **50/50 (100 %)** |

Même modèle, même boucle d'entraînement, même loss : seul le découpage des données change. C'est la modification minimale qui fait marcher l'inférence.

**Le coût du vocabulaire complet.** `o200k_base` fait 200 019 entrées pour un corpus qui en utilise 776. Mesures sur ce Mac :

| vocabulaire | par pas | par epoch | 60 epochs |
|---|---|---|---|
| 200 019 (tiktoken complet) | 4,8 s | **735 s (12 min)** | ~12 h |
| 778 (corpus + `<unk>` + `<pad>`) | 0,04 s | **1,2 s** | **77 s** |

On garde `tiktoken` (aucun BPE maison, aucune dépendance en plus), mais on ne garde que les tokens réellement présents dans le corpus, plus deux tokens spéciaux. Hors vocabulaire, un mot devient `<unk>` : mesuré à **0,4 %** des tokens sur des phrases françaises du domaine.

---

## 2. Ce qu'on touche, ce qu'on ne touche pas

| fichier | changement |
|---|---|
| `src/yoda/utils.py` | `Tokenizer` remplacé (lignes 15-25) |
| `src/yoda/transormer.py` | +1 import, +15 lignes (`generate`) |
| `src/yoda/cli.py` | réécrit (168 lignes) — hyperparamètres et boucle **identiques** |
| `src/yoda/tst_trn.py` | **aucun changement** (`TextWindowDataset` reste, simplement plus utilisé) |
| `src/yoda/coders.py` | **aucun changement** |
| `data/` | **aucun changement** |

Ce qui ne bouge pas dans `cli.py` : `choose_device()`, `embedding_dim=256`, `num_heads=8`, `hidden_dim=512`, `batch_size=16`, l'appel à `Tst_Trn`, la boucle d'epochs, les quatre `print`.

---

## 3. Étape 1 — `src/yoda/utils.py`, lignes 15 à 25

Remplace exactement ces 11 lignes :

```python
class Tokenizer:
    def __init__(
        self
    ) -> None:
        self.encoder = tiktoken.get_encoding("o200k_base")

    def encode(self, text: str):
        return self.encoder.encode(text)

    def decode(self, tokens: list[int]):
        return self.encoder.decode(tokens)
```

par :

```python
class Tokenizer:
    def __init__(
        self, vocab: list[int] | None = None
    ) -> None:
        self.encoder = tiktoken.get_encoding("o200k_base")
        self.vocab = list(vocab) if vocab is not None else None

        if self.vocab is None:
            self.n_vocab = self.encoder.n_vocab
            self.index = None
            self.unk = None
            self.pad = None
        else:
            self.index = {token: i for i, token in enumerate(self.vocab)}
            self.unk = len(self.vocab)
            self.pad = len(self.vocab) + 1
            self.n_vocab = len(self.vocab) + 2

    @classmethod
    def depuis_paires(cls, paires):
        encoder = tiktoken.get_encoding("o200k_base")
        ids = set()
        for paire in paires:
            prompt = f"TÂCHE : {paire['TÂCHE']}\nSOURCE : {paire['SOURCE']}\nCIBLE :"
            ids.update(encoder.encode(prompt))
            ids.update(encoder.encode(" " + paire["CIBLE"]))
        return cls(sorted(ids))

    def encode(self, text: str):
        ids = self.encoder.encode(text)
        if self.index is None:
            return ids
        return [self.index.get(i, self.unk) for i in ids]

    def decode(self, tokens: list[int]):
        if self.index is None:
            return self.encoder.decode(tokens)
        bruts = [self.vocab[t] for t in tokens if t < len(self.vocab)]
        return self.encoder.decode(bruts)
```

Ce que ça fait, en quatre idées :

1. **`depuis_paires`** encode tous les prompts et toutes les cibles du corpus, puis garde l'ensemble trié des ids tiktoken rencontrés. C'est la table de correspondance `indice local → id tiktoken`.
2. **`encode`** traduit les ids tiktoken en indices locaux ; un id absent de la table devient `<unk>`.
3. **`decode`** fait l'inverse et **jette** `<unk>` et `<pad>` (ils ne doivent jamais s'afficher).
4. **`pad`** sert à compléter les séquences à longueur fixe ; il sert aussi de fin de phrase (le modèle apprend à l'émettre après la traduction).

Le round-trip est exact : `decode(encode(x)) == x` sur les 536 prompts et les 536 cibles du corpus.

---

## 4. Étape 2 — `src/yoda/transormer.py`

**4.1** Ligne 1, ajoute l'import avant `from torch import nn` :

```python
import torch
```

**4.2** Après la ligne 28 (la ligne vide qui suit `return self.output_projection(decoder_output)`), insère :

```python
    @torch.inference_mode()
    def generate(self, source_ids, max_new_tokens=32):
        """Génère la suite token par token, en repartant du dernier token source.

        C'est exactement la convention de l'entraînement : le décodeur démarre
        sur `source_ids[-1]`, et chaque token produit est rajouté à son entrée.
        """
        decoder_input = source_ids[:, -1:]
        nouveaux = []
        for _ in range(max_new_tokens):
            logits = self(source_ids, decoder_input)
            suivant = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            nouveaux.append(int(suivant))
            decoder_input = torch.cat((decoder_input, suivant), dim=1)
        return nouveaux
```

Pourquoi `source_ids[:, -1:]` : à l'entraînement, la cible du décodeur est construite comme `[source[-1]] + cible[:-1]` (c'était déjà la convention de `TextWindowDataset`, c'est aussi celle de `PaireDataset`). Le premier token donné au décodeur est donc le dernier token de la source. `generate` fait pareil, sinon le modèle est décalé d'un cran et tout part en vrille.

Pas de cache KV : le décodeur est rejoué en entier à chaque token. À 16 tokens de sortie, ça coûte ~0,3 s (mesuré) : invisible à l'échelle d'une phrase.

---

## 5. Étape 3 — `src/yoda/cli.py`

Remplace **tout le fichier** par ceci (168 lignes) :

```python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from yoda.tst_trn import Tst_Trn
from yoda.transormer import Transformer
from yoda.utils import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "corpus_francais_minionese.txt"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "yoda.pt"

TACHES = {
    "fr-minion": "TÂCHE : Français -> Minionese",
    "minion-fr": "TÂCHE : Minionese -> Français",
}


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lire_paires(chemin):
    """Lit le corpus TÂCHE / SOURCE / CIBLE en une liste de dictionnaires."""
    paires, courant = [], {}
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        if not ligne.strip() or ligne.startswith("#"):
            continue
        cle, _, valeur = ligne.partition(" : ")
        courant[cle] = valeur
        if cle == "CIBLE":
            paires.append(courant)
            courant = {}
    return paires


class PaireDataset(Dataset):
    """Un exemple = une paire. La source est le prompt, la cible est la CIBLE."""

    def __init__(self, paires, tokenizer, context_length=64, target_length=16):
        self.paires = paires
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.target_length = target_length

    def __len__(self):
        return len(self.paires)

    def __getitem__(self, index):
        paire = self.paires[index]
        prompt = f"TÂCHE : {paire['TÂCHE']}\nSOURCE : {paire['SOURCE']}\nCIBLE :"

        source = self.tokenizer.encode(prompt)[-self.context_length:]
        source = [self.tokenizer.pad] * (self.context_length - len(source)) + source

        cible = self.tokenizer.encode(" " + paire["CIBLE"])[: self.target_length]
        cible = cible + [self.tokenizer.pad] * (self.target_length - len(cible))

        source_ids = torch.tensor(source, dtype=torch.long)
        target_ids = torch.tensor(cible, dtype=torch.long)
        decoder_input_ids = torch.cat((source_ids[-1:], target_ids[:-1]))
        return source_ids, decoder_input_ids, target_ids


def train(pairs_path, epochs, batch_size, context_length, target_length):
    paires = lire_paires(pairs_path)
    tokenizer = Tokenizer.depuis_paires(paires)
    vocab_size = tokenizer.n_vocab

    split = int(len(paires) * 0.9)
    train_dataset = PaireDataset(paires[:split], tokenizer, context_length, target_length)
    test_dataset = PaireDataset(paires[split:], tokenizer, context_length, target_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedding_dim = 256
    num_heads = 8
    hidden_dim = 512
    device = choose_device()

    trainer = Tst_Trn(embedding_dim, num_heads, hidden_dim, vocab_size, device)

    global_step = 0
    for epoch in range(epochs):
        train_loss, global_step = trainer.train_loop(train_loader, global_step)
        test_loss, test_accuracy = trainer.test_loop(test_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "hidden_dim": hidden_dim,
                "vocab_size": vocab_size,
                "vocab": tokenizer.vocab,
                "context_length": context_length,
                "target_length": target_length,
            },
            "model": trainer.model.state_dict(),
        },
        CHECKPOINT,
    )
    print(f"Checkpoint écrit dans {CHECKPOINT}")


def charger(checkpoint_path, device):
    sauv = torch.load(checkpoint_path, map_location=device)
    config = sauv["config"]
    tokenizer = Tokenizer(config["vocab"])
    model = Transformer(
        config["embedding_dim"],
        config["num_heads"],
        config["hidden_dim"],
        config["vocab_size"],
    )
    model.load_state_dict(sauv["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, config


def generer(checkpoint_path, sens, texte, device):
    model, tokenizer, config = charger(checkpoint_path, device)
    prompt = f"{TACHES[sens]}\nSOURCE : {texte}\nCIBLE :"

    source = tokenizer.encode(prompt)[-config["context_length"]:]
    source = [tokenizer.pad] * (config["context_length"] - len(source)) + source
    entree = torch.tensor([source], dtype=torch.long, device=device)

    ids = model.generate(entree, max_new_tokens=config["target_length"])
    return tokenizer.decode(ids).strip()


def main():
    parser = argparse.ArgumentParser(description="Entraîner Yoda ou traduire une phrase")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--target-length", type=int, default=16)
    parser.add_argument("--generate", metavar="TEXTE", default=None)
    parser.add_argument("--sens", choices=list(TACHES), default="fr-minion")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    if args.generate is not None:
        print(generer(args.checkpoint, args.sens, args.generate, choose_device()))
        return

    train(args.pairs, args.epochs, args.batch_size, args.context_length, args.target_length)


if __name__ == "__main__":
    main()
```

### Les six blocs, ligne par ligne

| bloc | rôle | à noter |
|---|---|---|
| `lire_paires` | lit le fichier en liste de dicts `{TÂCHE, SOURCE, CIBLE}` | une paire = 4 lignes ; les `#` et lignes vides sont sautés |
| `PaireDataset` | un exemple = un prompt + une cible | source **complétée à gauche** avec `<pad>`, cible **à droite** |
| `train` | identique à avant | seul le dataset change, plus le `torch.save` final |
| `charger` | reconstruit le modèle depuis le checkpoint | la `config` contient `vocab`, sinon les indices ne veulent rien dire |
| `generer` | prompt → ids → texte | **exactement** la même construction d'entrée que `PaireDataset` |
| `main` | `--generate` sinon entraînement | `--epochs` reste à 100 par défaut |

Trois détails qui comptent :

- **`encode(" " + paire["CIBLE"])`** : le corpus écrit `CIBLE : Da meow...`, donc la cible commence par une espace. Sans elle, le premier token de la réponse n'est pas celui vu à l'entraînement.
- **Le prompt finit par `CIBLE :`** (sans espace après). Vérifié avec tiktoken : `encode("CIBLE :" + " Da") == encode("CIBLE : Da")`. Le modèle génère donc bien ` Da...`.
- **`--target-length` passe de 32 à 16** : les cibles du corpus font 2 à 14 tokens (médiane 6). 16 suffit et ça double la vitesse.

---

## 6. Étape 4 — entraînement

```bash
uv run yoda --epochs 60
```

Sortie attendue (extrait d'un vrai run ; les chiffres bougent un peu d'un run à l'autre, il n'y a pas de seed) :

```
Epoch 1/60
Train loss: 2.8029
Test loss: 2.1416
Test accuracy: 0.6678
Epoch 2/60
Train loss: 1.9112
...
Epoch 60/60
Train loss: 0.0035
Test loss: 1.8248
Test accuracy: 0.7731
Checkpoint écrit dans /Users/.../Yoda/checkpoints/yoda.pt
```

- durée mesurée : **77 s** (1,2 s par epoch + le chargement)
- taille du fichier : **6,6 Mo**
- 536 paires, dont 482 pour l'entraînement et 54 pour la validation

`Train loss` à 0,003 signifie que le modèle reproduit parfaitement les paires apprises. `Test accuracy` à 0,77 (elle compte aussi les `<pad>`) mesure la partie validation : elle ne monte pas, c'est normal (voir §11).

Facultatif, pour ne pas versionner 6,6 Mo :

```bash
echo "checkpoints/" >> .gitignore
```

---

## 7. Étape 5 — inférence

```bash
uv run yoda --generate "Le chat cherche la sortie."
# Da meow-meow looka for da exit-o.

uv run yoda --generate "Mon ami a besoin d'aide."
# Mi amigo need-o help-y.

uv run yoda --generate "Le chien joue dans le jardin."
# Da woof-woof bap-bap in-da garden-o.

uv run yoda --generate "Da meow-meow looka for da exit-o." --sens minion-fr
# Le chat cherche la sortie.

uv run yoda --generate "Me sorri-sorri." --sens minion-fr
# Je suis désolé.
```

Options utiles :

```bash
uv run yoda --generate "..." --sens minion-fr      # autre sens
uv run yoda --generate "..." --checkpoint autre.pt # autre checkpoint
```

Compter ~1,5 s par appel : ~0,9 s d'import de torch, 0,32 s de chargement du checkpoint, ~0,3 s pour générer les 16 tokens.

---

## 8. Étape 6 — vérifier que ça marche vraiment

Ce script n'est pas à enregistrer, colle-le dans un heredoc :

```bash
uv run python - <<'PY'
import torch
from pathlib import Path
from yoda.cli import lire_paires, charger, choose_device, CHECKPOINT, TACHES

device = choose_device()
model, tokenizer, config = charger(CHECKPOINT, device)
paires = lire_paires(Path("data/corpus_francais_minionese.txt"))
split = int(len(paires) * 0.9)

def traduire(paire):
    sens = "fr-minion" if paire["TÂCHE"].startswith("Français") else "minion-fr"
    prompt = f"{TACHES[sens]}\nSOURCE : {paire['SOURCE']}\nCIBLE :"
    source = tokenizer.encode(prompt)[-config["context_length"]:]
    source = [tokenizer.pad] * (config["context_length"] - len(source)) + source
    ids = model.generate(
        torch.tensor([source], device=device),
        max_new_tokens=config["target_length"],
    )
    return tokenizer.decode(ids).strip()

for nom, lot in (("entraînement", paires[2:52]), ("validation", paires[split:split + 25])):
    justes = sum(traduire(p) == p["CIBLE"] for p in lot)
    print(f"{nom} : {justes}/{len(lot)} exactes")
PY
```

Attendu :

```
entraînement : 50/50 exactes
validation : 2/25 exactes
```

Si l'entraînement est à 50/50, la chaîne complète est correcte. Le 2/25 n'est pas un bug (voir §11).

---

## 9. Dépannage

| symptôme | cause | correction |
|---|---|---|
| `FileNotFoundError: .../checkpoints/yoda.pt` | rien n'a été entraîné | lancer `uv run yoda --epochs 60` |
| `KeyError: 'vocab'` | checkpoint écrit par l'ancien `cli.py` | supprimer `checkpoints/yoda.pt` et réentraîner |
| `KeyError: 'context_length'` | idem | idem |
| toujours la même sortie courte (ex. `Da.`) | modèle sous-entraîné : après 1 epoch il répond `Da.` à tout | relancer avec `--epochs 60` |
| le modèle recopie la source | le prompt ne finit pas par `CIBLE :` | vérifier `generer()`, pas d'espace finale |
| `RuntimeError: expected ... MPSFloatType` | tenseur créé sans dtype | `torch.tensor(..., dtype=torch.long)` |
| `ModuleNotFoundError: yoda` | venv désynchronisé | `uv sync` |
| la traduction part en vrille | phrase hors corpus (voir §11) | tester une phrase du corpus |

Vérification rapide de l'installation éditable (doit afficher `.../Yoda/src/yoda/cli.py`) :

```bash
uv run python -c "import yoda.cli; print(yoda.cli.__file__)"
```

---

## 10. Où est le checkpoint, et ce qu'il contient

```
checkpoints/yoda.pt
├── "config"
│   ├── embedding_dim   : 256
│   ├── num_heads       : 8
│   ├── hidden_dim      : 512
│   ├── vocab_size      : 778
│   ├── vocab           : [776 ids tiktoken triés]
│   ├── context_length  : 64
│   └── target_length   : 16
└── "model"             : state_dict de Transformer (1,72 M paramètres)
```

`vocab` est indispensable : sans lui, l'indice 42 ne désigne pas le même token, et le modèle produit du bruit. C'est pour ça que `charger()` reconstruit le `Tokenizer` **depuis le checkpoint** et pas depuis le corpus.

Le vocabulaire local vaut `776 + 1 (<unk>) + 1 (<pad>) = 778` entrées. `<unk>` = 776, `<pad>` = 777.

---

## 11. Limites, dites franchement

- **Le modèle mémorise, il ne généralise pas.** 536 paires, c'est trop peu pour apprendre une règle de traduction. Résultat : 100 % sur les paires apprises, 8 % (2/25) sur les paires de validation.
- **Les 54 dernières paires ne sont jamais apprises.** Le split 90/10 est conservé tel quel. Si tu testes une phrase du corpus qui tombe dans cette tranche, la sortie sera mauvaise : c'est attendu, c'est exactement ce que mesure `Test accuracy`.
- **Hors corpus, ça déraille.** `"La soupe est trop chaude."` donne `Da light-o es too strong-o.` (plus proche voisin : « la lumière est trop forte »). Le modèle n'a rien d'autre à proposer.
- **`<unk>`.** Un mot absent du corpus est ignoré silencieusement (`Bonjour` par exemple). Mesuré à 0,4 % des tokens sur des phrases du domaine, mais ça peut être bien plus sur un texte libre.
- **Pas de cache KV, pas de beam search.** Décodage glouton, décodeur rejoué à chaque token. Invisible à 16 tokens, à revoir si un jour tu génères des paragraphes.
- **Le sens n'est pas détecté.** C'est `--sens` (donc le champ `TÂCHE`) qui décide. `fr-minion` par défaut.
- **Pas de seed.** Deux entraînements donnent des chiffres légèrement différents ; l'exactitude, elle, reste stable.

---

## 12. Journal des mesures (pour retrouver d'où viennent les chiffres)

| mesure | valeur |
|---|---|
| paires dans `data/corpus_francais_minionese.txt` | 536 |
| tokens distincts utilisés (table locale) | 776 |
| tokens inconnus, phrases FR du domaine | 0,4 % |
| longueur des cibles | 2 à 14 tokens (médiane 6) |
| longueur du prompt | 17 à 29 tokens |
| pas d'entraînement, vocab 200 019 | 4,8 s |
| pas d'entraînement, vocab 778 | 0,04 s |
| 1 epoch, vocab 200 019 | 735 s |
| 1 epoch, vocab 778 (sortie : `Da.` à toutes les phrases) | 1,2 s |
| 60 epochs, vocab 778 | 77 s |
| train loss après 60 epochs | 0,003 |
| test accuracy après 60 epochs | 0,77 |
| paires d'entraînement exactes | 50/50 |
| paires de validation exactes | 2/25 |
| paramètres | 1,72 M |
| checkpoint | 6,6 Mo |
| chargement du checkpoint | 0,32 s |
| génération de 16 tokens (modèle chargé) | 0,28 s |
| appel CLI complet | ~1,5 s |

---

## 13. Pour la suite (hors périmètre de cette guideline)

`Web_interface/script.js` contient déjà, dans un bloc commenté « À BRANCHER PLUS TARD », le contrat complet :

- requête : `POST /api/traduire`, corps `{ "texte": "...", "sens": "fr-minion" | "minion-fr" }`
- réponse : du texte **en flux**, chaque morceau étant ajouté à la bulle avec un appel à `minion.onToken()` (c'est ce qui fait bouger la bouche)

La fonction à brancher derrière est `generer(checkpoint_path, sens, texte, device)` : elle prend une phrase, rend une phrase, et ne fait aucune hypothèse sur l'appelant. Il ne manque qu'un petit serveur HTTP autour, qui appelle `generer()` puis écrit la réponse morceau par morceau.
