#!/usr/bin/env python3
import json
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

from minions_translator.inference import load_model, translate

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "Web_interface"

print("Chargement du modèle (ça peut prendre quelques secondes)...")
MODEL, TOKENIZER, DEVICE, CONFIG = load_model()
print(f"Modèle prêt sur {DEVICE} !")
print(f"Serveur prêt sur http://localhost:8000")


class MinionHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Servir les fichiers depuis Web_interface/
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

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
