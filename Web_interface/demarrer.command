#!/bin/zsh
# Double-clique sur ce fichier pour lancer le serveur du modèle (macOS).
# Il démarre serve_minion.py sur http://localhost:8000 puis ouvre la page.

cd "$(dirname "$0")/.." || exit 1

echo "Démarrage du serveur du modèle sur http://localhost:8000 ..."
echo "(Ctrl+C pour arrêter)"

# Ouvre la page une fois le serveur prêt.
(sleep 6 && open http://localhost:8000) &

if [ -x .venv/bin/python ]; then
  exec .venv/bin/python serve_minion.py
elif command -v uv >/dev/null 2>&1; then
  exec uv run python serve_minion.py
else
  exec python3 serve_minion.py
fi
