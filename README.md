# Minions Translator

Un petit traducteur **français ↔ minions** basé sur un Transformer encodeur-décodeur écrit en PyTorch.

Le projet comprend l'entraînement du tokenizer BPE, l'entraînement du modèle, une CLI interactive et une interface web avec un serveur local.

## Fonctionnalités

- Traduction dans les deux sens : `Français → Minionese` et `Minionese → Français`
- Tokenizer BPE entraîné sur le corpus du projet
- Entraînement et inférence en ligne de commande
- Interface web avec un minion animé (`Web_interface/`)
- Modèles pré-entraînés fournis dans `models/`

## Prérequis

- [uv](https://docs.astral.sh/uv/) (recommandé)
- Environ 1 Go d'espace disque

## Installation

Depuis la racine du projet :

```bash
uv sync
```

Cette commande crée l'environnement virtuel `.venv` et installe les dépendances (`torch`, `tokenizers`, `numpy`, `click`).

## Entraînement

Les modèles sont déjà présents dans `models/`. Cette étape n'est utile que si vous voulez les régénérer.

### 1. Entraîner le tokenizer

```bash
uv run minions --mode train_tokenizer
```

Le fichier `models/tokenizer.json` est créé.

### 2. Entraîner le Transformer

```bash
uv run minions --mode train_transformer --epochs 10
```

Le modèle est sauvegardé dans `models/minionese_transformer.pt`.

## Utilisation

### En ligne de commande

```bash
uv run minions --mode inferance
```

Le programme demande le sens de traduction puis le texte à traduire. Il reste actif jusqu'à `Ctrl+C`.

```text
Traduire vers le minions ? (Sinon traduire vers le français) [Y/n]:
Entrez le texte à traduire: Mon ami a besoin d'aide.
Traduction : Mi amigo need-o help-y.
```

### Interface web

```bash
uv run python serve_minion.py
```

Puis ouvrez [http://localhost:8000](http://localhost:8000).

Sur macOS, vous pouvez aussi double-cliquer sur `Web_interface/demarrer.command`.

L'interface permet de saisir un texte, d'inverser le sens de traduction et de consulter le code du projet.

## Options de la CLI

| Option | Défaut | Description |
|---|---|---|
| `--mode` | `train_transformer` | `train_transformer`, `train_tokenizer` ou `inferance` |
| `--text` | `data/corpus_francais_minionese.txt` | Corpus utilisé pour l'entraînement |
| `--epochs` | `10` | Nombre d'époques |
| `--batch_size` | `32` | Taille des lots |
| `--context_length` | `64` | Longueur maximale de la source |
| `--target_length` | `32` | Longueur maximale de la cible |

Exemple :

```bash
uv run minions --mode train_transformer --epochs 20 --batch_size 16
```

## Structure du projet

```text
transformer/
├── data/                       # Corpus français ↔ minionese
├── models/                     # Tokenizer et modèle entraînés
├── src/minions_translator/     # Code du projet
│   ├── cli.py                  # Point d'entrée de la CLI
│   ├── transformer.py          # Architecture du Transformer
│   ├── utils.py                # Tokenizer, embeddings, attention…
│   ├── train.py                # Entraînement
│   └── inference.py            # Chargement du modèle et traduction
├── Web_interface/              # Interface web
├── serve_minion.py             # Serveur local (fichiers + API de traduction)
└── pyproject.toml
```

## Limites

Le modèle est entraîné sur un corpus réduit. Il reproduit bien les phrases apprises, mais généralise peu sur des phrases inédites. Pour de meilleurs résultats, ajoutez des paires dans `data/corpus_francais_minionese.txt` et relancez l'entraînement.
