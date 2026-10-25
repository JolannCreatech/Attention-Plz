#!/usr/bin/env python3
import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from minions_translator.inference import load_model, translate

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "Web_interface"

# Dossiers qu'on ne montre pas dans le gestionnaire de fichiers : trop lourds
# ou inutiles pour lire le code.
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# Au-delà, on n'envoie que le début du fichier au navigateur.
READ_LIMIT = 400_000

print("Chargement du modèle (ça peut prendre quelques secondes)...")
MODEL, TOKENIZER, DEVICE, CONFIG = load_model()
print(f"Modèle prêt sur {DEVICE} !")
print(f"Serveur prêt sur http://localhost:8000")


def safe_path(raw: str) -> Path | None:
    """Résout un chemin relatif sans jamais sortir de la racine du dépôt."""
    rel = unquote(raw or ".").lstrip("/")
    target = (ROOT / rel).resolve()
    if target != ROOT and ROOT not in target.parents:
        return None
    return target


def lister(directory: Path) -> list[dict]:
    """Contenu d'un dossier, dossiers d'abord puis fichiers, triés."""
    entries = []
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except OSError:
        return entries

    for child in children:
        if child.name in SKIP_DIRS or child.name.startswith("."):
            continue
        try:
            est_dossier = child.is_dir()
            taille = 0 if est_dossier else child.stat().st_size
        except OSError:
            continue
        entries.append(
            {
                "name": child.name,
                "path": child.relative_to(ROOT).as_posix(),
                "type": "dir" if est_dossier else "file",
                "size": taille,
            }
        )
    return entries


def lire_fichier(chemin: Path) -> dict:
    """Lit un fichier texte. Les binaires sont signalés, pas envoyés."""
    taille = chemin.stat().st_size
    with chemin.open("rb") as f:
        brut = f.read(READ_LIMIT)

    tronque = taille > len(brut)
    try:
        contenu = brut.decode("utf-8")
        binaire = False
    except UnicodeDecodeError:
        contenu = ""
        binaire = True

    return {
        "path": chemin.relative_to(ROOT).as_posix(),
        "size": taille,
        "content": contenu,
        "binary": binaire,
        "truncated": tronque,
    }


class MinionHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Servir les fichiers depuis Web_interface/
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data, status=200):
        corps = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", "no-store")
        self._set_cors()
        self.end_headers()
        self.wfile.write(corps)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/files":
            demandes = parse_qs(parsed.query)
            rel = demandes.get("path", ["."])[0]
            cible = safe_path(rel)

            if cible is None:
                self._send_json({"error": "chemin interdit"}, 403)
                return
            if not cible.is_dir():
                self._send_json({"error": "dossier introuvable"}, 404)
                return

            parent = cible.relative_to(ROOT).as_posix()
            if parent == ".":
                parent = ""
            self._send_json(
                {
                    "path": "" if cible == ROOT else parent,
                    "name": cible.name if cible != ROOT else ROOT.name,
                    "parent": (
                        None
                        if cible == ROOT
                        else (cible.parent.relative_to(ROOT).as_posix() or "")
                    ),
                    "entries": lister(cible),
                }
            )
            return

        if parsed.path == "/api/file":
            demandes = parse_qs(parsed.query)
            rel = demandes.get("path", [""])[0]
            cible = safe_path(rel)

            if cible is None:
                self._send_json({"error": "chemin interdit"}, 403)
                return
            if not cible.is_file():
                self._send_json({"error": "fichier introuvable"}, 404)
                return

            try:
                self._send_json(lire_fichier(cible))
            except OSError as e:
                self._send_json({"error": f"lecture impossible : {e}"}, 500)
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/traduire":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                data = json.loads(raw_body.decode("utf-8"))

                texte = data.get("texte", "").strip()
                sens = data.get("sens", "fr-minion")  # "fr-minion" ou "minion-fr"

                if not texte:
                    self.send_response(400)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self._set_cors()
                    self.end_headers()
                    self.wfile.write("Texte vide".encode("utf-8"))
                    return

                # Mapping avec ton package
                # fr-minion -> vers le minion
                # minion-fr -> vers le français
                tominion = sens == "fr-minion"
                traduction = translate(MODEL, TOKENIZER, DEVICE, tominion, texte)

                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._set_cors()
                self.end_headers()
                self.wfile.write(traduction.encode("utf-8"))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self._set_cors()
                self.end_headers()
                self.wfile.write(f"Erreur: {e}".encode("utf-8"))
                return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self._set_cors()
        self.end_headers()
        self.wfile.write(b"Not Found")

    def log_message(self, *args):
        # Un peu moins verbeux
        return


if __name__ == "__main__":
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, MinionHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur")
        httpd.server_close()
