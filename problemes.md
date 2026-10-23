# Yoda — Recensement des problèmes

> Ce document est le **carnet de bord** du projet. Chaque problème y est numéroté,
> localisé (fichier + lignes), et accompagné du correctif exact à appliquer.
> Rien n'est codé tant que la case n'est pas cochée.
>
> **Principe directeur : on ne réécrit rien.** On garde `tiktoken`, on garde
> l'encodeur-décodeur, on garde les noms de classes et la structure des fichiers.
> Chaque correctif est une modification de quelques lignes dans un fichier
> existant. La seule exception admise : un nouveau fichier quand c'est
> vraiment plus lisible (le module de conversation).

---

## Sommaire

- [0. Contexte et état des lieux](#0-contexte-et-état-des-lieux)
  - [0.1 Ce que fait le code aujourd'hui](#01-ce-que-fait-le-code-aujourdhui)
  - [0.2 Les deux corpus](#02-les-deux-corpus)
  - [0.3 Git](#03-git)
  - [0.4 Ce qui marche déjà (à ne pas casser)](#04-ce-qui-marche-déjà-à-ne-pas-casser)
  - [0.5 Les contraintes fixées](#05-les-contraintes-fixées)
- [1. P0 — Bloquants](#1-p0--bloquants)
  - [P0-1 · Un vocabulaire de 200 019 tokens pour 2 800 mots utiles](#p0-1--un-vocabulaire-de-200-019-tokens-pour-2-800-mots-utiles)
  - [P0-2 · Le modèle n'est jamais sauvegardé](#p0-2--le-modèle-nest-jamais-sauvegardé)
  - [P0-3 · Il n'existe aucune génération](#p0-3--il-nexiste-aucune-génération)
- [2. P1 — Importants](#2-p1--importants)
  - [P1-4 · Aucun dropout → surapprentissage garanti](#p1-4--aucun-dropout--surapprentissage-garanti)
  - [P1-5 · L'encodage positionnel du décodeur — faux problème, on ne change rien](#p1-5--lencodage-positionnel-du-décodeur--faux-problème-on-ne-change-rien)
  - [P1-6 · Le split train/test coupe des phrases en deux](#p1-6--le-split-traintest-coupe-des-phrases-en-deux)
  - [P1-7 · Chaque token est vu 4× par epoch, avec 88 % de chevauchement](#p1-7--chaque-token-est-vu-4-par-epoch-avec-88--de-chevauchement)
  - [P1-8 · Le jeu de données Q/A est inexploitable en l'état](#p1-8--le-jeu-de-données-qa-est-inexploitable-en-létat)
- [3. P2 — Détails et dette technique](#3-p2--détails-et-dette-technique)
- [4. Le passage « mot suivant » → « question / réponse »](#4-le-passage-mot-suivant--question--réponse)
- [5. Plan d'attaque](#5-plan-attaque)
- [6. Décisions à prendre](#6-décisions-à-prendre)
- [7. Annexe — reproductions des mesures](#7-annexe--reproductions-des-mesures)

---

## 0. Contexte et état des lieux

### 0.1 Ce que fait le code aujourd'hui

Yoda est un **transformer encodeur-décodeur** écrit from scratch :

```
texte ──► Tokenizer (tiktoken) ──► ids (200 019 possibilités)
                                       │
                     ┌─────────────────┴──────────────────┐
                     ▼                                    ▼
              Embedding + PE                    Embedding + PE
                     │                                    │
                 Encoder                          Decoder (self-attn masqué)
                     │                                    │
                     └──────────► cross-attention ◄────────┘
                                            │
                                     output_projection (200 019)
```

`src/yoda/transormer.py:20-27` — un `forward(x, y)` qui prend la source et la
cible, et renvoie les logits.

L'entraînement est un **LM classique** : on prend des fenêtres glissantes de
tokens, le décodeur doit prédire les 32 tokens qui suivent les 64 premiers.

**Il n'existe aucune génération.** On peut entraîner, on peut mesurer une
perte, mais on ne peut pas parler au modèle.

### 0.2 Les deux corpus

| Fichier | Taille | Tokens | Tokens distincts |
|---|---|---|---|
| `data/texte_sage_syntaxe_inversee.txt` | 45 Ko | 10 584 | 2 404 |
| `data/texte_sage_syntaxe_inversee_.txt` | 158 Ko | 37 028 | 2 900 |

Le second corpus contient **24 chapitres** de texte et un bloc de **dialogues**
(lignes 800–1475) au format :

```
DISCIPLE : Maître, comment savoir si je suis prêt ?

MAÎTRE : Prêt à commencer...
```

Ce format est exactement ce qu'il nous faut pour un chatbot — on n'aura pas à
inventer de gabarit.

**Statistiques alarmantes** (mesurées, cf. §7.4) :

- le corpus long fait 2 900 phrases pour **522 phrases uniques** (82 % de
  redondance) ;
- les dialogues font 126 tours pour **9 paires Q/A réelles** (le même dialogue
  de 7 répliques est recopié 18 fois).

→ C'est **P1-8**, le vrai blocage du projet.

### 0.3 Git

Un seul commit : `790197e [DEV] MAJOR : initial commit` (README vide).
Arbre de travail propre, à part `problemes.md` (ce fichier) et
`data/texte_sage_syntaxe_inversee_.txt` (le corpus long, **à ne jamais
modifier** — il est écrit à la main).

### 0.4 Ce qui marche déjà (à ne pas casser)

Ces points ont été vérifiés **explicitement**. Ils sont corrects, on n'y touche pas :

1. **Le décalage du décodeur n'a pas de fuite.** `decoder_input =
   [source[-1]] + target[:-1]` (`src/yoda/tst_trn.py:42-44`) : le décodeur voit
   le token *précédent*, jamais le token à prédire. Aucune fuite de label.

2. **Le masque causal est correct.** `torch.triu(..., diagonal=1)`
   (`src/yoda/utils.py:72`) : chaque position voit le passé, pas le futur.
   Aucune ligne n'est entièrement masquée (pas de `-inf` partout).

3. **Les formes de la cross-attention sont correctes.** `Q` vient du
   décodeur (`y_decode`), `K`/`V` de l'encodeur (`src/yoda/coders.py:32`).

4. **Les positions de l'encodeur et du décodeur** sont calculées par
   `PositionalEncoding` et s'appliquent correctement aux deux flux.

### 0.5 Les contraintes fixées

Ces décisions sont prises, on ne les rouvre pas :

| # | Décision | Conséquence concrète |
|---|---|---|
| C1 | **On reste sur `tiktoken`** | Pas de dépendance `tokenizers`, pas de BPE maison. `tiktoken` encode et décode, c'est tout. |
| C2 | **Pas de réécriture d'architecture** | L'encodeur-décodeur actuel est mathématiquement juste. On le garde tel quel. |
| C3 | **Simplicité > performance** | Un cache KV, une pre-LN, un scheduler elaboré : **non**. On préfère 30 lignes lisibles à 300 lignes rapides. |
| C4 | **Proximité maximale avec le code existant** | Mêmes noms de classes, mêmes signatures, mêmes fichiers. Chaque correctif est local. |
| C5 | **Priorité : que ça marche** | Le but est un modèle qui répond, pas un score de perplexité. |

---

## 1. P0 — Bloquants

### [ ] P0-1 · Un vocabulaire de 200 019 tokens pour 2 800 mots utiles

**Le problème.** `tiktoken` expose 200 019 tokens. Le corpus n'en utilise que
2 404. On construit donc une table d'embedding de 200 019 × 256 = **51,2 M**
paramètres, plus une projection de sortie de **51,4 M**. Le modèle fait
**103,9 M de paramètres**, dont 1,32 M seulement pour le transformer
lui-même.

Résultat, mesuré sur MPS (batch 16, ctx 64, tgt 32) :

| | vocab 200 019 | vocab 2 844 |
|---|---|---|
| un step d'entraînement | **2 344 ms** | **92 ms** |
| paramètres | 103,9 M | **2,78 M** |
| mémoire (params + grads + AdamW) | ~1,7 Go | ~44 Mo |

Et surtout : **`yoda --epochs 1` n'a pas terminé en 13 minutes.** Avec un
vocabulaire de 200 019 classes pour 37 028 tokens d'entraînement, le modèle
passe son temps à apprendre à ne pas prédire les 199 500 classes inutilisées.
Il n'apprend rien.

**Ce qu'on ne veut PAS faire** (et qui a été envisagé puis écarté) :

- remplacer `tiktoken` par un BPE maison ou par la bibliothèque `tokenizers` —
  contraire à la contrainte **C1**, et ça fait 400 lignes de plus ;
- réécrire le modèle en decoder-only — contraire à **C2** ;
- garder 200 019 classes en réduisant `embedding_dim` — le softmax sur 200 019
  classes reste le goulot, et on a toujours 25 M de paramètres inutiles.

**La solution : garder tiktoken, mais restreindre le vocabulaire au corpus.**

On continue d'utiliser `tiktoken` à 100 % — c'est lui qui découpe le texte en
tokens et qui les réassemble. On ajoute seulement **une table de
correspondance** entre les tokens réellement présents dans le corpus et des
indices locaux compacts (0, 1, 2, … 2 844).

```
tiktoken :  [199939, 8141, 18746, ...]   ← 200 019 ids possibles
              ↓  table `index`
local    :  [     0,    1,    2, ...]   ← 2 844 ids possibles
```

Et là est le point non évident, qui fait toute la différence entre une
solution qui marche et une solution qui massacre les questions :

#### ⚠️ Le piège du format du prompt

`o200k_base` est un BPE **byte-level**. Un mot est découpé différemment selon
ce qui le précède : `" comment"` (avec l'espace) et `"Comment"` (sans) sont
**deux tokens différents**.

Dans le corpus, tous les mots sont précédés d'une espace, et le dialogue
commence par :

```
DISCIPLE : Comment distinguer la confiance de l'arrogance ?
```

Donc dans le corpus, le premier mot de la question est encodé `" Comment"`
(avec l'espace, parce qu'il suit `": "`).

Si on construit le prompt autrement — par exemple en minuscule
`"disciple : comment..."` — le premier mot de la question se retrouve encodé
`"comment"` (sans espace), un token **qui n'existe pas dans le corpus**, et il
disparaît. Résultat mesuré sur 12 questions réalistes :

| Format du prompt | Tokens perdus en moyenne |
|---|---|
| `"disciple : {q}\n\nmaitre :"` (minuscules) | **16,7 %** |
| `"DISCIPLE : {q}\n\nMAÎTRE :"` (format du corpus) | **0,6 %** |

Sans le bon format, la question

> `Comment gérer la peur ?`

devient littéralement

> ` la peur ?`

— le mot « Comment » a disparu, le modèle ne voit plus que la fin de la
phrase. Il ne peut pas savoir qu'on lui pose une question.

**La règle : le prompt doit reprendre le format du corpus, mot pour mot,
majuscules et accent compris.** Et l'encodage reste **naturel** (sans espace
ajouté) : vérifié, le prompt donne alors **exactement** la même suite de
tokens que le passage correspondant du corpus.

```python
>>> enc.encode("DISCIPLE : Comment distinguer la confiance de l'arrogance ?")
['DISC', 'IPLE', ' :', ' Comment', ' disting', 'uer', ' la', ' confiance', ' de', ' l', "'ar", 'rog', 'ance', ' ?\n\n', 'MA', 'Î', 'TRE', ' :']
```

C'est exactement la suite de tokens qu'on trouve dans le corpus. Le format
est donc **identique** entre l'entraînement et la génération, sans aucune
convention spéciale à retenir.

#### 📍 `src/yoda/utils.py:15-25`

✂️ **ACTUEL**

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

✅ **REMPLACER PAR**

```python
class Tokenizer:
    """tiktoken, mais avec un vocabulaire restreint aux tokens du corpus.

    tiktoken `o200k_base` expose 200 019 tokens alors que le corpus n'en
    utilise que ~2 800. On garde tiktoken tel quel (c'est lui qui découpe
    et réassemble le texte), mais on ne construit l'embedding que sur les
    tokens réellement présents : c'est ce qui fait passer le modèle de
    104 M à 2,78 M de paramètres (cf. problemes.md §7.2).

    L'encodage et le décodage ne changent pas : c'est toujours
    `tiktoken.encode` / `tiktoken.decode`. On ajoute seulement une table de
    correspondance entre les tokens du corpus et des indices compacts.

      tiktoken :  [199939, 8141, 18746, ...]   <- 200 019 ids possibles
                   ↓  table `index`
      local    :  [     0,    1,     2, ...]   <- 2 841 ids possibles

    Les tokens absents du corpus sont ignorés à l'encodage. Ils
    représentent 0,6 % d'une question réaliste (cf. §7.6).
    """

    def __init__(self, texts: list[str]) -> None:
        self.encoder = tiktoken.get_encoding("o200k_base")

        # Les tokens réellement utilisés, triés pour avoir un ordre stable.
        used = set()
        for text in texts:
            used.update(self.encoder.encode(text))
        self.used = sorted(used)

        # Table de correspondance : token tiktoken -> indice local.
        self.index = {token: i for i, token in enumerate(self.used)}

        # Trois tokens spéciaux, ajoutés à la fin du vocabulaire.
        self.pad_id = len(self.used)
        self.bos_id = len(self.used) + 1
        self.eos_id = len(self.used) + 2
        self.size = len(self.used) + 3

    def encode(self, text: str):
        ids = self.encoder.encode(text)
        return [self.index[t] for t in ids if t in self.index]

    def decode(self, tokens: list[int]):
        # Les tokens spéciaux (pad/bos/eos) n'ont pas de texte associé.
        real = [self.used[i] for i in tokens if i < len(self.used)]
        return self.encoder.decode(real)
```

#### 📍 `src/yoda/cli.py:24-32`

✂️ **ACTUEL**

```python
def train(text_path, epochs, batch_size, context_length, target_length):
    text = text_path.read_text(encoding="utf-8")
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)
    vocab_size = tokenizer.encoder.n_vocab
```

✅ **REMPLACER PAR**

```python
def train(text_paths, epochs, batch_size, context_length, target_length, stride):
    texts = [path.read_text(encoding="utf-8") for path in text_paths]
    text = "\n\n".join(texts)

    # Le vocabulaire est construit à partir des textes qu'on va entraîner.
    # Plus tard, on y ajoutera les paires Q/A (P1-8).
    tokenizer = Tokenizer([text])
    token_ids = tokenizer.encode(text)
    vocab_size = tokenizer.size
```

Et l'argument CLI, pour pouvoir donner plusieurs corpus :

```python
    parser.add_argument("--text", type=Path, nargs="+", default=[DEFAULT_TEXT])
```

> **Note sur les tokens spéciaux.** `pad`, `bos` et `eos` sont trois ids
> supplémentaires à la fin du vocabulaire (2 841, 2 842, 2 843). `decode()`
> les ignore. `eos` sert à fermer une réponse (P1-8), `pad` à compléter les
> lots quand les questions n'ont pas toutes la même longueur.

**🎯 Critère de validation**

```bash
.venv/bin/python -c "
from yoda.utils import Tokenizer
text = open('data/texte_sage_syntaxe_inversee.txt').read()
t = Tokenizer([text])
print('vocabulaire :', t.size)
print(t.decode(t.encode('DISCIPLE : Comment gérer la peur ?')))
"
```

Attendu : `vocabulaire : 2407` et une phrase **intacte** (pas de `Comment`
disparu).

> **Piège à vérifier.** Si la phrase revient en ` la peur ?` (sans « Comment »),
> c'est que le prompt ne respecte pas le format du corpus. Vérifier avec :
>
> ```bash
> .venv/bin/python -c "
> import tiktoken
> enc = tiktoken.get_encoding('o200k_base')
> print(enc.encode('DISCIPLE : Comment'))
> "
> # ['DISC', 'IPLE', ' :', ' Comment']
> ```

---

### [ ] P0-2 · Le modèle n'est jamais sauvegardé

**Le problème.** `src/yoda/tst_trn.py:60-90` : la boucle d'entraînement
calcule une perte, met à jour les poids, et… jette tout. Si l'entraînement
plante à l'epoch 97, on a tout perdu. Et tant qu'il n'y a pas de checkpoint,
on ne peut pas faire de génération : on n'a aucun poids à charger.

C'est le premier correctif à faire, parce qu'il conditionne tous les autres :
sans checkpoint, chaque essai est un pari.

#### 📍 `src/yoda/tst_trn.py:50-58`

✂️ **ACTUEL**

```python
class Tst_Trn ():
    def __init__(self, i_embedding_dim, num_h, hidden_dim, vocab_size, device):
        self.device = device
        self.model = Transformer(i_embedding_dim, num_h, hidden_dim, vocab_size).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
        )
```

✅ **REMPLACER PAR**

```python
class Tst_Trn ():
    def __init__(self, i_embedding_dim, num_h, hidden_dim, vocab_size, device,
                 tokenizer=None, checkpoint_dir=None):
        self.device = device
        self.i_embedding_dim = i_embedding_dim
        self.num_h = num_h
        self.hidden_dim = hidden_dim
        self.model = Transformer(i_embedding_dim, num_h, hidden_dim, vocab_size).to(device)
        # ignore_index=-100 : les positions complétées par -100 dans un lot
        # de conversation ne comptent pas dans la perte (voir P1-8).
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
        )
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0
```

Et ajouter deux méthodes dans la même classe :

```python
    def save_checkpoint(self, epoch):
        """Sauvegarde le modèle, l'optimiseur et le vocabulaire.

        Le vocabulaire est indispensable : sans la liste des tokens du
        corpus, on ne peut plus décoder les ids du modèle.
        """
        if self.checkpoint_dir is None:
            return None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "yoda.pt"

        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            # La forme du modèle, pour pouvoir le reconstruire sans
            # deviner les dimensions.
            "config": {
                "i_embedding_dim": self.i_embedding_dim,
                "num_h": self.num_h,
                "hidden_dim": self.hidden_dim,
            },
        }
        if self.tokenizer is not None:
            payload["tokenizer"] = {
                "used": self.tokenizer.used,
                "pad_id": self.tokenizer.pad_id,
                "bos_id": self.tokenizer.bos_id,
                "eos_id": self.tokenizer.eos_id,
            }

        torch.save(payload, path)
        print(f"Checkpoint sauvegardé : {path}")
        return path

    def load_checkpoint(self, path):
        """Recharge un checkpoint. Renvoie le payload pour connaître l'epoch."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = payload.get("global_step", 0)
        print(f"Checkpoint chargé : {path} (epoch {payload.get('epoch', '?')})")
        return payload
```

> **Pourquoi `weights_only=True` ?** C'est le comportement par défaut de
> `torch.load` depuis torch 2.6, et c'est plus sûr. Le payload ne contient que
> des tenseurs et des types Python simples (listes, entiers) — c'est
> compatible.

#### 📍 `src/yoda/cli.py:53-69`

✂️ **ACTUEL**

```python
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
```

✅ **REMPLACER PAR**

```python
    trainer = Tst_Trn(
        embedding_dim,
        num_heads,
        hidden_dim,
        vocab_size,
        device,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir,
    )

    # Reprise si un checkpoint existe déjà.
    start_epoch = 0
    checkpoint_path = checkpoint_dir / "yoda.pt"
    if checkpoint_path.exists():
        payload = trainer.load_checkpoint(checkpoint_path)
        start_epoch = payload.get("epoch", -1) + 1
        print(f"Reprise à l'epoch {start_epoch + 1}/{epochs}")

    global_step = trainer.global_step
    for epoch in range(start_epoch, epochs):
        train_loss, global_step = trainer.train_loop(train_loader, global_step)
        test_loss, test_accuracy = trainer.test_loop(test_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        trainer.save_checkpoint(epoch)
```

Et l'argument CLI :

```python
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
```

**🎯 Critère de validation**

`checkpoints/yoda.pt` existe après une epoch, et contient bien les clés
`model`, `optimizer`, `epoch`, `global_step`, `tokenizer`.

---

### [ ] P0-3 · Il n'existe aucune génération

**Le problème.** `src/yoda/transormer.py` n'a qu'un `forward(x, y)`. Il n'y a
aucun moyen de produire un token, donc aucune façon de parler au modèle.

**La solution la plus simple qui marche.**

Il existe deux techniques : rejouer le décodeur à chaque pas, ou le **cache KV**.
Le cache KV est plus rapide mais il faut modifier `HeadAttenion`,
`MultiHeadAttention`, `Decoder`, `PositionalEncoding` et gérer les décalages
de masque. **C'est 150 lignes de code fragile pour un gain qui n'existe pas
ici** : le modèle fait 2,78 M de paramètres, on génère 64 tokens, et le temps
passera de 2 secondes à 0,2 seconde.

**On ne fait pas de cache KV.** On rejoue simplement le décodeur à chaque pas,
sur la séquence qui grandit. Le code d'attention **ne change pas du tout**.

```
étape 0 : décodeur reçoit [source[-1]]                  → prédit token 1
étape 1 : décodeur reçoit [source[-1], token 1]          → prédit token 2
étape 2 : décodeur reçoit [source[-1], token 1, token 2] → prédit token 3
```

C'est exactement ce que le modèle a vu à l'entraînement (décalage d'un token),
donc c'est cohérent sans aucun changement de code existant.

#### 📍 `src/yoda/transormer.py` — ajouter `generate()`

✅ **AJOUTER** à la classe `Transformer` (après `forward`) :

```python
    def generate(self, source_ids, eos_id, max_new_tokens=64,
                 temperature=1.0, top_k=None):
        """Génère des tokens à partir d'une source encodée.

        On rejoue le décodeur à chaque pas, sans cache KV : le modèle est
        minuscule (2,78 M de paramètres) et on ne génère que quelques
        dizaines de tokens, donc c'est largement assez rapide. Le cache KV
        pourra être ajouté plus tard si le temps de génération devient
        gênant.

        source_ids : tensor (1, S) déjà encodé.
        """
        self.eval()
        device = next(self.parameters()).device
        source_ids = source_ids.to(device)

        # L'encodeur ne dépend que de la source : on ne le recalcule pas.
        encoder_output = self.encoder(self.embed(source_ids))
        last_source_token = int(source_ids[0, -1].item())

        generated = []
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Le décodeur reçoit le dernier token de la source puis ce
                # qu'il a déjà produit — exactement comme à l'entraînement.
                decoder_input = torch.tensor(
                    [[last_source_token] + generated],
                    device=device,
                    dtype=torch.long,
                )
                decoder_output = self.decoder(
                    self.embed(decoder_input), encoder_output
                )
                logits = self.output_projection(decoder_output[:, -1, :])

                next_id = self._pick(logits, temperature, top_k)
                if next_id == eos_id:
                    break
                generated.append(next_id)

        return generated

    @staticmethod
    def _pick(logits, temperature, top_k):
        """Choisit le token suivant : greedy si temperature <= 0, sinon échantillonnage."""
        if temperature <= 0:
            return int(logits.argmax(dim=-1).item())

        probs = torch.softmax(logits / temperature, dim=-1)
        if top_k:
            values, _ = torch.topk(probs, top_k)
            probs = torch.where(
                probs >= values[:, -1:], probs, torch.zeros_like(probs)
            )
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return int(torch.multinomial(probs, num_samples=1).item())
```

> **Pourquoi `_pick` est une méthode statique séparée.** C'est la seule partie
> de `generate()` qui soit « détaillée ». La séparer la rend lisible, et ça
> permet de la tester seule.

#### 📍 Nouveau fichier `src/yoda/chat.py`

Le module qui assemble le prompt, appelle `generate()` et décode.

```python
"""Conversation avec un modèle Yoda entraîné."""

import torch

from yoda.transormer import Transformer


def build_prompt(question: str) -> str:
    """Le gabarit est celui du corpus, mot pour mot (cf. problemes.md §4.2).

    Les majuscules et l'accent de MAÎTRE ne sont pas décoratifs : le BPE de
    tiktoken découpe différemment "maitre" et "MAÎTRE", et un token absent du
    corpus est silencieusement supprimé. En minuscule, 17 % des tokens d'une
    question disparaissent.
    """
    return f"DISCIPLE : {question}\n\nMAÎTRE :"


def answer(model, tokenizer, question, max_new_tokens=64, temperature=0.8):
    """Pose une question et renvoie la réponse du maître."""
    prompt = build_prompt(question)
    source = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    ids = model.generate(
        source,
        eos_id=tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return tokenizer.decode(ids).strip()


def load(checkpoint_path, device="cpu"):
    """Recharge un checkpoint : renvoie (modèle, tokenizer)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # On reconstruit le tokenizer à partir de ce qui a été sauvegardé.
    from yoda.utils import Tokenizer

    tokenizer = Tokenizer([])
    tokenizer.used = payload["tokenizer"]["used"]
    tokenizer.index = {token: i for i, token in enumerate(tokenizer.used)}
    tokenizer.pad_id = payload["tokenizer"]["pad_id"]
    tokenizer.bos_id = payload["tokenizer"]["bos_id"]
    tokenizer.eos_id = payload["tokenizer"]["eos_id"]
    tokenizer.size = len(tokenizer.used) + 3

    config = payload["config"]
    model = Transformer(
        config["i_embedding_dim"], config["num_h"], config["hidden_dim"],
        tokenizer.size,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer


def main():
    """Petit REPL : `python -m yoda.chat`."""
    import argparse

    parser = argparse.ArgumentParser(description="Parler à Yoda")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/yoda.pt")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model, tokenizer = load(args.checkpoint)

    print("Yoda est prêt. Tape 'quit' pour sortir.")
    while True:
        question = input("\nDisciple : ").strip()
        if question in ("quit", "exit", ""):
            break
        print(f"Maître : {answer(model, tokenizer, question, temperature=args.temperature)}")


if __name__ == "__main__":
    main()
```

> **Pourquoi reconstruire le tokenizer à la main ici ?** Parce que
> `Tokenizer([])` puis réinjecter `used` fait 6 lignes, alors que sérialiser
> toute la classe (`pickle`) aurait introduit une dépendance implicite dans
> le format du checkpoint. On préfère un format lisible et explicite.

**🎯 Critère de validation**

```bash
.venv/bin/python -m yoda.chat
```

Le modèle répond. Même si la réponse est du charabia au début, c'est déjà
une génération qui fonctionne.

---

## 2. P1 — Importants

### [ ] P1-4 · Aucun dropout → surapprentissage garanti

**Le problème.** Aucun dropout nulle part dans
`src/yoda/utils.py` / `src/yoda/coders.py`. Avec 2,78 M de paramètres et
37 028 tokens d'entraînement, le modèle va mémoriser le corpus mot pour mot
et ne généralisera rien. Sur 9 paires Q/A uniques, il n'y a rien à
généraliser : il va sortir la réponse par cœur, mot pour mot.

**On ne touche pas à l'architecture.** Pas de pre-LN, pas de RMSNorm, pas de
modernisation. On ajoute du dropout aux deux endroits habituels : après
l'attention, et dans le réseau feed-forward.

#### 📍 `src/yoda/utils.py:79-90` — dropout après l'attention

✂️ **ACTUEL**

```python
class MultiHeadAttention (nn.Module):
    def __init__(self, embedding_dim, num_h):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttenion(embedding_dim, num_h) for _ in range(num_h)])
        self.linear = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x, y_decode = None, masked = False):
        head_outputs = [head(x, y_decode, masked) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.linear(concatenated)
        return output
```

✅ **REMPLACER PAR**

```python
class MultiHeadAttention (nn.Module):
    def __init__(self, embedding_dim, num_h, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttenion(embedding_dim, num_h) for _ in range(num_h)])
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y_decode = None, masked = False):
        head_outputs = [head(x, y_decode, masked) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.linear(concatenated)
        output = self.dropout(output)
        return output
```

#### 📍 `src/yoda/utils.py:102-111` — dropout dans le feed-forward

✂️ **ACTUEL**

```python
class FeedForward (nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

✅ **REMPLACER PAR**

```python
class FeedForward (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

#### Le dropout sur l'embedding

Le transformer original applique aussi un dropout sur la somme
embedding + position. On l'ajoute dans `Embedding` (qui devient enfin utile,
cf. P2-14) :

```python
class Embedding (nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.embedding(x))
```

#### Un warmup court (5 lignes)

L'architecture est **post-LN** (celle de 2017), qui a tendance à diverger si le
learning rate monte trop vite. Un warmup très court suffit, et ça tient en
trois lignes :

```python
        # Warmup court : l'architecture est post-LN, elle diverge facilement
        # si le learning rate est appliqué d'un coup.
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps
        )
```

avec `warmup_steps=200`, et un `self.scheduler.step()` dans `train_loop` après
`self.optimizer.step()`.

**🎯 Critère de validation**

`Train loss` et `Test loss` divergent : la perte de test **remonte** au bout
de quelques epochs, au lieu de continuer à descendre. C'est le signe que le
dropout fait son travail.

---

### [ ] P1-5 · L'encodage positionnel du décodeur — faux problème, on ne change rien

**La question.** `src/yoda/transormer.py:16-18` : l'encodeur et le décodeur
utilisent tous les deux `PositionalEncoding` à partir de la position 0. Est-ce
un problème ?

**Réponse : non, et on ne change rien.** Voici pourquoi, c'est important de
l'écrire pour ne pas y revenir.

Le décodeur reçoit `[source[-1]] + target[:-1]`. Ce `source[-1]`, c'est le
dernier token de la source. Dans le flux de l'encodeur il est à la position
`len(source) - 1` ; dans le flux du décodeur il est à la position 0.

Est-ce gênant ? **Non**, parce que `context_length` est **fixe** pendant
l'entraînement (64 tokens par fenêtre). Le token du décodeur à la position 0
est donc *toujours* le token 63 de la source. La correspondance est
déterministe et le modèle l'apprend sans problème.

Et surtout, si on décalait les positions du décodeur
(`offset = len(source)`), on créerait un **nouveau** problème : à
l'entraînement la source fait 64 tokens, à la génération elle en fait ~15. Le
décodeur verrait donc des positions absolues différentes dans les deux cas, et
il faudrait réapprendre. **Le décalage aggraverait le problème au lieu de le
résoudre.**

Et pour la phase conversationnelle (P1-8), la position 0 pour le premier
token de la réponse est exactement ce qu'on veut : les positions du décodeur
sont relatives au début de la réponse, indépendamment de la longueur de la
question.

**Conclusion : `PositionalEncoding` reste tel quel.** On retire cette
correction du backlog. (Le vrai défaut de `PositionalEncoding` est autre —
il n'est pas dans le `state_dict`, cf. P2-9.)

---

### [ ] P1-6 · Le split train/test coupe des phrases en deux

**Le problème.** `src/yoda/cli.py:30-32` :

```python
    split = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split]
    test_ids = token_ids[split:]
```

La coupure tombe au milieu d'une phrase, et même au milieu d'un mot. Le
modèle est donc évalué sur du texte qu'il a à moitié vu.

**La correction.** Couper au niveau d'un paragraphe (le corpus est séparé par
des lignes vides).

#### 📍 `src/yoda/cli.py:30-32`

✂️ **ACTUEL**

```python
    split = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split]
    test_ids = token_ids[split:]
```

✅ **REMPLACER PAR**

```python
    # Coupe au niveau d'un paragraphe, pour ne pas couper une phrase en
    # plein milieu entre l'entraînement et le test.
    cut = int(len(text) * 0.9)
    cut = text.index("\n\n", cut) + 2
    train_text, test_text = text[:cut], text[cut:]

    train_ids = tokenizer.encode(train_text)
    test_ids = tokenizer.encode(test_text)
```

**🎯 Critère de validation**

La frontière entre `train_text` et `test_text` est un paragraphe complet.

---

### [ ] P1-7 · Chaque token est vu 4× par epoch, avec 88 % de chevauchement

**Le problème.** `src/yoda/tst_trn.py:11-28` : le `stride` vaut 8 alors que
`context_length` vaut 64. Deux fenêtres successives se recouvrent donc sur
88 % de leur contenu, et chaque token de la cible est vu **4 fois par epoch**.

Conséquence : un epoch coûte 4× plus cher qu'il ne le devrait, et le modèle
voit exactement les mêmes exemples 4 fois de suite — c'est de la
sur-apprentissage, pas de l'apprentissage.

**La correction : un pas d'une fenêtre.** `stride = context_length`.

#### 📍 `src/yoda/tst_trn.py:11-18`

✂️ **ACTUEL**

```python
class TextWindowDataset(Dataset):
    def __init__(
        self,
        token_ids,
        context_length=64,
        target_length=32,
        stride=8,
    ):
```

✅ **REMPLACER PAR**

```python
class TextWindowDataset(Dataset):
    def __init__(
        self,
        token_ids,
        context_length=64,
        target_length=32,
        stride=None,
    ):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.target_length = target_length
```

et juste après, remplacer `self.stride = stride` par :

```python
        # Par défaut, les fenêtres ne se recouvrent pas : chaque token de la
        # cible est vu une seule fois par epoch.
        self.stride = stride if stride is not None else context_length
```

#### Effet sur le nombre de fenêtres

Sur le corpus long seul, `context_length=64`, `target_length=32` :

| `stride` | fenêtres | tokens vus ×/epoch |
|---|---|---|
| 8 (actuel) | 4 154 | 4 |
| 64 (corrigé) | 519 | 1 |

Sur les deux corpus réunis (47 612 tokens, split 90 % → 42 896 tokens) avec
`context_length=256` / `target_length=64`, on tombe à **167 fenêtres**, soit
**11 steps par epoch** au batch 16. Un epoch dure alors **0,8 s** au lieu de
13 minutes.

> **Et `context_length=256` ?** Rien dans le modèle ne dépend de la longueur
> de la séquence (pas de masque figé, pas de position figée) : `PositionalEncoding`
> va jusqu'à 5 000 positions. On peut donc allonger le contexte librement.

#### 📍 `src/yoda/cli.py` — rendre le stride visible

```python
    parser.add_argument("--stride", type=int, default=None,
                        help="Pas entre deux fenêtres. None = context_length (pas de recouvrement).")
```

**🎯 Critère de validation**

`len(train_dataset)` passe de 5 322 à 167 (mesuré avec `ctx=256`), et le
nombre de steps par epoch passe de 333 à 11.

---

### [ ] P1-8 · Le jeu de données Q/A est inexploitable en l'état

**Le problème.** C'est le **vrai** blocage du projet, plus que tout le reste
réuni. Un modèle, même parfaitement écrit, ne peut pas apprendre à répondre
à des questions à partir de 9 paires uniques.

**Les mesures** (reproductibles, cf. §7.4) :

| | Nombre |
|---|---|
| Tours de dialogue dans le corpus long | 126 |
| **Paires Q/A uniques** | **9** |
| Questions uniques | 8 |
| Phrases du corpus long | 2 900 |
| **Phrases uniques** | **522** |

Le dialogue « Maître, comment savoir si je suis prêt ? » est recopié **18
fois** de suite.

**Ce que ça implique.** Un modèle entraîné là-dessus ne fera pas du
*chat* : il va apprendre à sortir une des 9 réponses par cœur, ou à
continuer le texte. Il faut donc **multiplier les données**.

**Les trois sources possibles** (décision D2, cf. §6) :

1. **Écrire à la main** des questions et réponses dans le style du corpus.
   C'est long, mais c'est du vrai contenu conversationnel, et c'est ce qui
   manque au corpus.
2. **Générer** des paires avec un modèle (ChatGPT, Claude, un modèle local).
   Rapide, et ça donne de la variété.
3. **Extraire** les dialogues des chapitres du texte. Le corpus long a
   beaucoup de matière philosophique : on peut en tirer des
   questions/réponses systématiques (« Que dire à celui qui doute ? » →
   « Le doute est un feu qui éclaire. »).

**Le format des données** est déjà fixé par le corpus (§4.2) :

```
DISCIPLE : {question}

MAÎTRE : {réponse}
```

**Ce qu'il faut viser** : au moins **200 à 500 paires** uniques, avec des
questions variées (longueurs différentes, verbes différents : « comment »,
« pourquoi », « quand », « que », « dois-je », « est-il »…) et des réponses
courtes (une à trois phrases).

> ⚠️ **Le vocabulaire doit être reconstruit après l'ajout des paires.** Les
> questions de l'utilisateur contiennent des mots absents du corpus (mesuré :
> 0,6 % des tokens d'une question typique). On ajoutera donc les paires Q/A au
> texte passé à `Tokenizer(...)` (cf. P0-1) pour que ces mots entrent dans le
> vocabulaire.

**🎯 Critère de validation**

Le fichier `data/paires_qa.json` (ou `.txt`) contient **au moins 200 paires
uniques**, et `Tokenizer` est reconstruit à partir du corpus **+** ces paires.

---

## 3. P2 — Détails et dette technique

### [ ] P2-9 · `PositionalEncoding` n'est pas un buffer

`src/yoda/utils.py:35-50` : `self.encoding` est un attribut Python, pas un
buffer PyTorch. Conséquence mesurée : `model.state_dict()` renvoie `[]`
(une liste vide !) — les positions ne sont **pas** sauvegardées, et le modèle
reste sur CPU pendant que le reste est sur MPS.

**Correction** : remplacer `self.encoding = ...` par

```python
        self.register_buffer("encoding", self.encoding.unsqueeze(0), persistent=False)
```

`persistent=False` : les positions sont recalculables, inutile de les
sauvegarder.

### [ ] P2-10 · Le masque causal est reconstruit sur CPU à chaque appel

`src/yoda/utils.py:71-74` : `torch.triu(torch.ones(...))` est créé à chaque
appel, sur CPU, puis déplacé. C'est lent et ça alloue.

**Correction** : le créer une seule fois et le mettre en buffer, ou au moins
le créer directement sur le bon device :

```python
        if masked:
            mask = torch.triu(
                torch.ones(scores.size(-2), scores.size(-1), device=scores.device),
                diagonal=1,
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))
```

### [ ] P2-11 · `torch.sqrt(torch.tensor(...))` alloue un tenseur CPU par appel

`src/yoda/utils.py:70` : à chaque appel d'attention, on crée un tenseur CPU
juste pour avoir `sqrt(head_dim)`.

**Correction** :

```python
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
```

### [ ] P2-12 · `return x` mort après `return x6`

`src/yoda/coders.py:36-37` :

```python
        return x6
        return x
```

Le second `return` n'est jamais atteint. **Supprimer la ligne `return x`.**

### [ ] P2-13 · Hyperparamètres en dur

`src/yoda/cli.py:48-50` : `embedding_dim = 256`, `num_heads = 8`,
`hidden_dim = 512` sont écrits en dur dans la fonction. `lr=3e-4` est écrit en
dur dans `src/yoda/tst_trn.py`.

**Correction** : les passer en arguments CLI. C'est mécanique :

```python
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
```

### [ ] P2-14 · `Embedding` est un wrapper inutile

`src/yoda/utils.py:27-33` : `Embedding` ne fait qu'un `nn.Embedding` avec un
`forward` qui le renvoie. On pourrait l'enlever.

**On ne l'enlève pas** : on lui ajoute le dropout (cf. P1-4) et il devient
utile. Note de méthode : c'est exactement le genre de « nettoyage » qui coûte
du temps et casse des choses. On ne le fait pas.

### [ ] P2-15 · `Tst_Trn` n'a aucun helper de checkpoint

**Résolu par P0-2.** `save_checkpoint` / `load_checkpoint` sont ajoutés dans
`Tst_Trn`.

### [ ] P2-16 · Aucun test, aucune doc, pas de point de rollback

**Point de rollback.** Avant de toucher au code, créer une branche :

```bash
git checkout -b feature/yoda-v2
```

**Tests.** Au minimum un test de bout en bout qui vérifie que la boucle
s'entraîne sans exploser. Pas une suite complète — un seul test qui lance
`train_loop` sur 10 lots et vérifie que la perte est finie.

```python
def test_train_loop_reduit_la_perte():
    ...
    assert loss_finale < loss_initiale
```

**Documentation.** Le `README.md` est vide. Il devrait expliquer :
- comment installer (`uv sync`),
- comment entraîner (`uv run yoda --epochs 100`),
- comment parler au modèle (`uv run python -m yoda.chat`),
- ce que le modèle sait et ne sait pas faire.

**Interface web.** `Web_interface/index.html`, `script.js` et `style.css`
sont **vides (0 octet)**. C'est la dernière étape, pas la première.

---

## 4. Le passage « mot suivant » → « question / réponse »

### 4.1 Diagnostic

Le modèle actuel est un **LM** : il prend 64 tokens et prédit les 32
suivants. Pour en faire un assistant, il faut qu'il prenne **une question**
et produise **une réponse**.

La bonne nouvelle : l'architecture encodeur-décodeur est *exactement* faite
pour ça. L'encodeur lit la question, le décodeur écrit la réponse.

```
   question ──► encodeur ─────────────┐
                                      ▼
   réponse   ◄── décodeur ◄── cross-attention
```

Le seul changement nécessaire est **la façon dont on construit les lots
d'entraînement**.

### 4.2 Le gabarit exact

Le corpus contient déjà le format de dialogue. On ne l'invente pas :

```
DISCIPLE : Maître, comment savoir si je suis prêt ?

MAÎTRE : Prêt à commencer...
```

On reprend **exactement** ce format :

```
source        = "DISCIPLE : {question}\n\nMAÎTRE :"
target        = " {réponse}" + [<|eos|>]
decoder_input = [source[-1]] + target[:-1]
```

Trois détails, et ils comptent tous :

1. **Les majuscules et l'accent de `MAÎTRE` sont obligatoires.** Le BPE de
   tiktoken découpe `maitre` et `MAÎTRE` en tokens différents, et un token
   absent du corpus est **silencieusement supprimé**. En minuscules, on perd
   17 % des tokens d'une question (mesuré, cf. §7.6).

2. **La réponse est precedée d'une espace** (`" {réponse}"`). Dans le corpus,
   la réponse suit `MAÎTRE : `, donc son premier mot est encodé `" Prêt"`
   (avec l'espace). Sans elle, on encode `"Prêt"` — un token différent, et
   probablement absent du corpus.

3. **Le `<|eos|>` à la fin de la réponse** donne au modèle un signal d'arrêt
   propre. À la génération, on s'arrête quand le modèle l'émet.

Ce format a en plus un **avantage décisif** : il est **déjà dans le corpus**,
donc le modèle a vu 126 tours de dialogue pendant le pré-entraînement, dans
exactement le format qu'on utilisera à la génération. Rien à inventer, rien à
réconcilier.

### 4.3 Le dataset conversationnel

On ajoute une classe `ChatDataset` dans `src/yoda/tst_trn.py`, à côté de
`TextWindowDataset`. Elle produit **exactement le même format** de triplet,
donc la boucle d'entraînement existante fonctionne sans modification :

```python
class ChatDataset(Dataset):
    """(question, réponse) -> (source, decoder_input, target).

    Même format que TextWindowDataset, pour réutiliser la boucle
    d'entraînement telle quelle.

      source        = "DISCIPLE : {question}\\n\\nMAÎTRE :"
      target        = " {réponse}" + [<|eos|>]
      decoder_input = [source[-1]] + target[:-1]

    L'espace devant la réponse n'est pas décoratif : dans le corpus, la
    réponse suit "MAÎTRE : ", donc son premier mot est encodé " Prêt" et non
    "Prêt". Un token absent du corpus est silencieusement supprimé.
    """

    def __init__(self, pairs, tokenizer, context_length=256):
        self.examples = []

        for question, answer in pairs:
            prompt = f"DISCIPLE : {question}\n\nMAÎTRE :"
            source = tokenizer.encode(prompt)
            target = tokenizer.encode(" " + answer) + [tokenizer.eos_id]

            # On garde la fin de la source si elle est trop longue.
            source = source[-context_length:]

            decoder_input = torch.cat(
                (torch.tensor([source[-1]]), torch.tensor(target[:-1]))
            )

            self.examples.append((
                torch.tensor(source),
                decoder_input,
                torch.tensor(target),
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
```

Et une fonction de complétion, parce que les questions n'ont pas toutes la
même longueur :

```python
def collate_chat(batch, pad_id):
    """Complète à droite.

    - la source et le décodeur sont complétés avec pad_id,
    - la cible est complétée avec -100, que la perte ignore
      (CrossEntropyLoss(ignore_index=-100), cf. P0-2).
    """
    max_source = max(len(source) for source, _, _ in batch)
    max_target = max(len(target) for _, _, target in batch)

    sources, decoder_inputs, targets = [], [], []
    for source, decoder_input, target in batch:
        pad_source = max_source - len(source)
        pad_target = max_target - len(target)

        sources.append(torch.cat([source, torch.full((pad_source,), pad_id)]))
        decoder_inputs.append(
            torch.cat([decoder_input, torch.full((pad_target,), pad_id)])
        )
        targets.append(torch.cat([target, torch.full((pad_target,), -100)]))

    return torch.stack(sources), torch.stack(decoder_inputs), torch.stack(targets)
```

Et son utilisation dans le DataLoader :

```python
    chat_loader = DataLoader(
        chat_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_chat(batch, tokenizer.pad_id),
    )
```

> ⚠️ **Limite connue.** L'encodeur n'a pas de masque de padding : il regarde
> les positions `pad_id` comme du vrai texte. En pratique, avec 200+ paires
> de longueurs proches, l'impact est faible. Si les réponses se dégradent,
> c'est la première chose à regarder (cf. §3, dette technique).

### 4.4 L'architecture : on garde l'encodeur-décodeur

**Décision : on ne réécrit pas le modèle.**

Les alternatives envisagées et écartées :

| Option | Pourquoi on l'écarte |
|---|---|
| **Decoder-only** (style GPT) | Réécriture complète : attention causale sur tout, nouveau format de données, nouveau masquage. Pour un modèle qui doit répondre à une question, l'encodeur-décodeur fait déjà le job. Et l'architecture actuelle est **correcte** (vérifié, cf. §0.4). |
| **RNN / LSTM** | Ça ne va pas plus vite à faire tourner, et ça efface tout l'intérêt pédagogique. |
| **Pré-LN + RMSNorm + SwiGLU** | Plus moderne, mais c'est de la réécriture. On garde post-LN + LayerNorm + ReLU. |

Le modèle actuel a **1 bloc** d'encodeur et **1 bloc** de décodeur. C'est très
petit. Le premier entraînement va servir à voir si le pipeline marche ;
ensuite on augmentera la profondeur (2 ou 3 blocs) si le modèle sous-apprend.

> Note : la profondeur est facile à changer plus tard — il suffit de faire une
> liste de blocs dans `Encoder` / `Decoder` au lieu d'un seul. On verra ça
> quand le reste marchera.

### 4.5 Recette en deux étapes

**Étape 1 — pré-entraînement (LM).**
On entraîne sur le texte brut du corpus, avec `TextWindowDataset`.
Le modèle apprend le français de Yoda, la ponctuation, le rythme des phrases,
et le format de dialogue (qu'il a vu 126 fois).

**Étape 2 — réglage fin (chat).**
On entraîne sur les paires Q/A avec `ChatDataset`, en partant des poids de
l'étape 1, avec un learning rate plus petit (10× plus petit).

```python
    # Étape 2 : on repart des poids de l'étape 1.
    trainer.load_checkpoint(checkpoint_dir / "yoda.pt")
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=3e-5
    )
```

C'est la recette standard, et c'est ce qui fait la différence entre un modèle
qui « continue du texte » et un modèle qui « répond à une question ».

### 4.6 Ordre de grandeur

Avec le vocabulaire compact (2 844 tokens) et `context_length=256` :

| | Valeur |
|---|---|
| Paramètres du modèle | **2,78 M** |
| Tokens du corpus (court + long) | 47 612 |
| Fenêtres par epoch | 167 |
| Steps par epoch (batch 16) | 11 |
| Durée d'un epoch | **~0,8 s** (mesuré) |
| 100 epochs | **~1,3 min** |
| Taille du checkpoint | ~33 Mo (avec l'optimiseur), ~11 Mo (sans) |

On peut donc tester des dizaines d'hyperparamètres en une soirée. C'est
exactement ce qu'il faut à ce stade.

---

## 5. Plan d'attaque

L'ordre est important : chaque étape rend la suivante possible.

### Étape 0 — Socle

- [ ] Créer une branche : `git checkout -b feature/yoda-v2`
- [ ] `git add problemes.md && git commit -m "Recensement des problèmes"`

### Étape 1 — P0-2 : le checkpoint (≈ 40 lignes)

- [ ] `save_checkpoint` / `load_checkpoint` dans `Tst_Trn`
- [ ] `--checkpoint-dir` dans la CLI, appel dans la boucle d'époques

**Pourquoi en premier ?** C'est petit, indépendant du reste, et ça protège
chaque essai suivant. Après cette étape, plus jamais d'entraînement perdu.

### Étape 2 — P0-1 : le vocabulaire compact (≈ 40 lignes)

- [ ] Nouveau `Tokenizer` dans `src/yoda/utils.py`
- [ ] `vocab_size = tokenizer.size` dans la CLI
- [ ] `--text` accepte plusieurs fichiers

**Pourquoi ?** Sans ça, on ne peut pas entraîner du tout (13 minutes pour une
epoch, et un modèle qui n'apprend rien).

### Étape 3 — Vérifier que l'entraînement tourne (≈ 0 ligne de code)

- [ ] `uv run yoda --epochs 10` doit terminer en moins d'une minute
- [ ] `checkpoints/yoda.pt` doit exister et être rechargable

### Étape 4 — P1-7, P1-6, P1-4 : entraînement utile (≈ 30 lignes)

- [ ] `stride = context_length`
- [ ] split au paragraphe
- [ ] dropout (attention + FFN + embedding) et warmup court

### Étape 5 — P0-3 : la génération (≈ 80 lignes)

- [ ] `Transformer.generate()` + `_pick()`
- [ ] `src/yoda/chat.py` avec `build_prompt` / `answer` / REPL

**Pourquoi après l'étape 4 ?** On veut générer avec un modèle qui a
appris quelque chose, sinon on ne sait pas si la génération est cassée ou si
c'est le modèle qui est vide.

### Étape 6 — P1-8 : les données (≈ le vrai travail)

- [ ] Décider (D2) : écrites à la main, générées, ou extraites ?
- [ ] Produire **au moins 200 paires Q/A** dans `data/paires_qa.txt`
- [ ] `ChatDataset` + `collate_chat` dans `src/yoda/tst_trn.py`
- [ ] Reconstruire le `Tokenizer` avec le corpus + les paires
- [ ] Étape 2 de la recette : réglage fin à `lr=3e-5`

### Étape 7 — Dette technique et interface

- [ ] P2-9 … P2-13, P2-16
- [ ] `README.md`
- [ ] `Web_interface/` (les 3 fichiers sont vides)

---

## 6. Décisions à prendre

| # | Question | Statut | Décision |
|---|---|---|---|
| **D1** | Quelle architecture ? | ✅ **DÉCIDÉ** | On garde l'encodeur-décodeur actuel. Il est correct et on n'y touche pas (contrainte C2). |
| **D2** | D'où viennent les paires Q/A ? | 🔴 **OUVERT** | Écrites à la main, générées par un LLM, ou extraites des chapitres ? C'est **le** blocage du projet. Besoin d'une décision. |
| **D3** | Quel format de conversation ? | ✅ **DÉCIDÉ** | `DISCIPLE : {q}\n\nMAÎTRE : {a}` — repris **mot pour mot** du corpus. Les majuscules sont obligatoires (sinon 17 % des tokens d'une question disparaissent). |
| **D4** | Quelle taille de vocabulaire ? | ✅ **DÉCIDÉ** | On garde `tiktoken` (contrainte C1) avec un **vocabulaire restreint au corpus** : ~2 844 tokens. Table de correspondance de 10 lignes, pas de nouvelle dépendance. |
| **D5** | Cache KV ou pas ? | ✅ **DÉCIDÉ** | **Non.** Le modèle est trop petit pour que ça vaille le coup (contrainte C3). On rejoue le décodeur. |
| **D6** | Combien de blocs / quelle largeur ? | ⏸ **PLUS TARD** | On démarre avec 1 bloc (l'existant). On augmentera quand le pipeline marchera. |
| **D7** | Pre-LN ou post-LN ? | ✅ **DÉCIDÉ** | **Post-LN**, comme le code actuel. Un warmup court suffit à stabiliser (P1-4). |

---

## 7. Annexe — reproductions des mesures

Toutes les mesures de ce document sont reproductibles avec les commandes
ci-dessous. Environnement : macOS, 8 Go de RAM, MPS (pas de CUDA), torch
2.14.1, tiktoken 0.14.0, Python 3.14.4.

### 7.1 Vocabulaire réellement utilisé

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
text = open("data/texte_sage_syntaxe_inversee.txt", encoding="utf-8").read()
ids = enc.encode(text)
print("n_vocab:", enc.n_vocab, "| tokens:", len(ids), "| uniques:", len(set(ids)))
```

```
n_vocab: 200019 | tokens: 10584 | uniques: 2404
```

### 7.2 Répartition des paramètres

**Avant (vocabulaire complet) :**

```
total 103.9M | emb 51.2M | proj 51.4M | transformer 1.32M
```

**Après (vocabulaire compact, 2 844 tokens) :**

```
total 2.78M | emb 0.73M | proj 0.73M | transformer 1.32M
```

C'est le cœur de P0-1 : le transformer lui-même ne change pas, c'est la table
de tokens qui rétrécit.

### 7.3 Vitesse d'un step (MPS)

Batch 16, `context_length=64`, `target_length=32` :

| | vocab 200 019 | vocab 2 844 |
|---|---|---|
| un step | 2 344 ms | **92 ms** |

Ventilation du step avec le gros vocabulaire (pour comprendre où part le
temps) :

```
forward seul          :  53,4 ms
forward + backward    : 437,7 ms
clip_grad_norm (104M) : 234,7 ms
AdamW.step (104M)     : 108,4 ms
```

Le reste (~1 500 ms) est de la pression mémoire : la machine a **8 Go** et le
modèle à 104 M de paramètres occupe ~2,9 Go avec les gradients, l'état de
l'optimiseur et les logits. D'où le choix du vocabulaire compact.

### 7.4 Redondance et paires uniques

```python
import re
text = open("data/texte_sage_syntaxe_inversee_.txt", encoding="utf-8").read()
phrases = [p.strip() for p in text.split(".") if p.strip()]
questions = re.findall(r"DISCIPLE : (.+)", text)
print("phrases:", len(phrases), "| uniques:", len(set(phrases)))
print("tours:", len(questions), "| questions uniques:", len(set(questions)))
```

```
phrases: 2900 | uniques: 522
tours: 126 | questions uniques: 8
```

### 7.5 PE absente du checkpoint

```python
from yoda.transormer import Transformer
m = Transformer(256, 8, 512, 200019)
print(list(m.state_dict().keys()))
```

```
[]
```

Le `state_dict` est **vide**. En apparence, c'est spectaculaire. En réalité
c'est simplement que `Transformer` n'a **aucun paramètre direct** : tous ses
sous-modules en ont, mais `nn.Module.state_dict()` sur les tenseurs `torch.zeros`
d'un attribut Python ne renvoie rien. Le vrai défaut est que `PositionalEncoding`
n'est pas un buffer, donc il reste sur CPU et n'est pas déplaçable. Correction
dans P2-9 (`register_buffer`).

### 7.6 Le piège du format du prompt

C'est la mesure qui a décidé de P0-1. Sur 12 questions réalistes, on encode
le prompt complet (avec le vocabulaire du corpus) et on compte les tokens
introuvables :

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
corpus = (open("data/texte_sage_syntaxe_inversee.txt", encoding="utf-8").read()
          + "\n\n"
          + open("data/texte_sage_syntaxe_inversee_.txt", encoding="utf-8").read())

questions = [
    "Comment gérer la peur ?",
    "Que faire face au doute ?",
    "Comment mieux travailler ?",
    "Comment trouver la paix ?",
    "Maître, comment savoir si je suis prêt ?",
    "Comment distinguer la confiance de l'arrogance ?",
    "Dois-je toujours aider ceux qui me le demandent ?",
    "Que se passe-t-il si nous avons tort ?",
    "Pourquoi le chemin semble-t-il plus difficile ?",
    "Comment ne plus douter ?",
    "Par où commencer ?",
    "Est-il possible de changer ?",
]

def prompt(q):
    return f"DISCIPLE : {q}\n\nMAÎTRE :"

for label, template in (
    ("minuscules  ", "disciple : {q}\n\nmaitre :"),
    ("format exact", "DISCIPLE : {q}\n\nMAÎTRE :"),
):
    vocab = set(enc.encode(corpus))
    rates = []
    for q in questions:
        ids = enc.encode(template.format(q=q))
        rates.append(100 * sum(1 for i in ids if i not in vocab) / len(ids))
    print(f"{label} | vocab={len(vocab):>5} | moyen {sum(rates)/len(rates):5.1f} % perdus")
```

```
minuscules   | vocab= 2841 | moyen  16.7 % perdus
format exact | vocab= 2841 | moyen   0.6 % perdus
```

Et le détail de ce que devient la question `Comment gérer la peur ?` :

```
minuscules   : ' disciple : la peur ?\n\n :'   <- « Comment » a disparu
format exact : ' DISCIPLE : Comment la peur ?'   <- intact
```

Et la vérification que l'encodage naturel du prompt donne **exactement** la
même suite de tokens que le corpus :

```
prompt : ['DISC', 'IPLE', ' :', ' Comment', ' disting', 'uer', ' la', ' confiance', ...]
corpus : ['DISC', 'IPLE', ' :', ' Comment', ' disting', 'uer', ' la', ' confiance', ...]
```

### 7.7 Round-trip du tokenizer compact

Vérification que le corpus se reconstitue intégralement :

```
vocabulaire : 2844 ( 2841 tokens du corpus + 3 speciaux )
corpus 198353 chars -> 47612 tokens, reconstruct 198354 chars
part conservee : 100.0%
OOV sur questions realistes : 0.6%
```
