(() => {
  'use strict';

  const filEl = document.getElementById('fil');
  const listeEl = document.getElementById('liste');
  const nomEl = document.getElementById('nom-fichier');
  const metaEl = document.getElementById('meta-fichier');
  const contenuEl = document.getElementById('contenu');
  const erreurEl = document.getElementById('erreur');

  if (!filEl || !listeEl || !nomEl || !metaEl || !contenuEl || !erreurEl) {
    console.error('Gestionnaire de fichiers : éléments HTML introuvables.');
    return;
  }

  /* Le serveur du modèle sert aussi cette page. Ouverte depuis un autre
     serveur (ou en file://), on retombe sur localhost:8000. */
  const API = (location.protocol === 'http:' || location.protocol === 'https:') &&
    location.port === '8000'
    ? location.origin
    : 'http://localhost:8000';

  let dossier = '';
  let fichierActif = '';

  /* ------------------------------------------------------------------ Outils */

  function tailleLisible(octets) {
    if (octets < 1024) return octets + ' o';
    if (octets < 1024 * 1024) return (octets / 1024).toFixed(1).replace('.', ',') + ' Ko';
    return (octets / (1024 * 1024)).toFixed(1).replace('.', ',') + ' Mo';
  }

  function afficherErreur(message) {
    erreurEl.textContent = message;
    erreurEl.hidden = false;
  }

  function effacerErreur() {
    erreurEl.hidden = true;
    erreurEl.textContent = '';
  }

  async function getJson(url) {
    const reponse = await fetch(url, { cache: 'no-store' });
    if (!reponse.ok) {
      let detail = '';
      try {
        const corps = await reponse.json();
        detail = corps.error ? ' — ' + corps.error : '';
      } catch (_) {
        /* pas de JSON, tant pis */
      }
      throw new Error('HTTP ' + reponse.status + detail);
    }
    return reponse.json();
  }

  /* -------------------------------------------------------------- Fil d'Ariane */

  function dessinerFil(chemin, nomRacine) {
    filEl.textContent = '';

    const racine = document.createElement('button');
    racine.type = 'button';
    racine.textContent = nomRacine || 'racine';
    racine.title = 'Revenir à la racine du dépôt';
    racine.addEventListener('click', () => chargerDossier(''));
    filEl.appendChild(racine);

    if (!chemin) {
      racine.classList.add('fil__courant');
      return;
    }

    const morceaux = chemin.split('/');
    let cumul = '';

    morceaux.forEach((morceau, index) => {
      cumul = cumul ? cumul + '/' + morceau : morceau;
      const separateur = document.createElement('span');
      separateur.textContent = '/';
      filEl.appendChild(separateur);

      const cible = cumul;
      const bouton = document.createElement('button');
      bouton.type = 'button';
      bouton.textContent = morceau;
      bouton.addEventListener('click', () => chargerDossier(cible));

      if (index === morceaux.length - 1) bouton.classList.add('fil__courant');
      filEl.appendChild(bouton);
    });
  }

  /* ---------------------------------------------------------------- Explorateur */

  function dessinerListe(entrees) {
    listeEl.textContent = '';

    if (!entrees.length) {
      const vide = document.createElement('li');
      vide.textContent = 'Dossier vide.';
      vide.style.padding = '10px';
      vide.style.fontSize = '0.8rem';
      vide.style.color = 'rgba(232, 238, 245, 0.4)';
      listeEl.appendChild(vide);
      return;
    }

    for (const entree of entrees) {
      const item = document.createElement('li');
      const bouton = document.createElement('button');
      bouton.type = 'button';

      const icone = document.createElement('span');
      icone.className = 'icone';
      icone.textContent = entree.type === 'dir' ? '▸' : '·';
      bouton.appendChild(icone);

      const nom = document.createElement('span');
      nom.textContent = entree.name + (entree.type === 'dir' ? '/' : '');
      bouton.appendChild(nom);

      if (entree.type === 'file') {
        const taille = document.createElement('span');
        taille.className = 'taille';
        taille.textContent = tailleLisible(entree.size);
        bouton.appendChild(taille);
        if (entree.path === fichierActif) bouton.classList.add('actif');
      }

      bouton.title = entree.path;
      bouton.addEventListener('click', () => {
        if (entree.type === 'dir') chargerDossier(entree.path);
        else chargerFichier(entree.path);
      });

      item.appendChild(bouton);
      listeEl.appendChild(item);
    }
  }

  /* ------------------------------------------------------------------ Lecture */

  function dessinerFichier(donnees) {
    nomEl.textContent = donnees.path;
    metaEl.textContent = tailleLisible(donnees.size);

    contenuEl.textContent = '';

    if (donnees.binary) {
      const avis = document.createElement('p');
      avis.className = 'avis';
      avis.textContent = 'Fichier binaire : il ne peut pas être affiché comme du texte.';
      contenuEl.appendChild(avis);
      return;
    }

    if (!donnees.content) {
      const avis = document.createElement('p');
      avis.className = 'avis';
      avis.textContent = 'Fichier vide.';
      contenuEl.appendChild(avis);
      return;
    }

    const pre = document.createElement('pre');
    const code = document.createElement('code');

    for (const ligne of donnees.content.split('\n')) {
      const span = document.createElement('span');
      span.className = 'ligne';
      span.textContent = ligne || ' ';
      code.appendChild(span);
    }

    pre.appendChild(code);
    contenuEl.appendChild(pre);

    if (donnees.truncated) {
      const avis = document.createElement('p');
      avis.className = 'avis';
      avis.textContent = 'Fichier trop volumineux : seul le début est affiché (' +
        tailleLisible(donnees.content.length) + ' sur ' + tailleLisible(donnees.size) + ').';
      contenuEl.appendChild(avis);
    }

    contenuEl.scrollTop = 0;
  }

  async function chargerFichier(chemin) {
    effacerErreur();
    fichierActif = chemin;
    nomEl.textContent = chemin;
    metaEl.textContent = 'chargement…';

    for (const bouton of listeEl.querySelectorAll('button')) {
      bouton.classList.toggle('actif', bouton.title === chemin);
    }

    try {
      const donnees = await getJson(API + '/api/file?path=' + encodeURIComponent(chemin));
      dessinerFichier(donnees);
    } catch (e) {
      nomEl.textContent = chemin;
      metaEl.textContent = '';
      contenuEl.textContent = '';
      afficherErreur('Impossible de lire ce fichier : ' + e.message);
    }
  }

  async function chargerDossier(chemin) {
    effacerErreur();

    try {
      const donnees = await getJson(API + '/api/files?path=' + encodeURIComponent(chemin));
      dossier = donnees.path;
      dessinerFil(donnees.path, donnees.name);
      dessinerListe(donnees.entries);

      fichierActif = '';
      nomEl.textContent = 'Aucun fichier sélectionné';
      metaEl.textContent = '';

      const vide = document.createElement('p');
      vide.className = 'lecteur__vide';
      vide.textContent = 'Choisis un fichier à gauche pour l’afficher.';
      contenuEl.textContent = '';
      contenuEl.appendChild(vide);
    } catch (e) {
      listeEl.textContent = '';
      afficherErreur(
        'Impossible de lister les fichiers : ' + e.message +
        ' — vérifie que le serveur tourne (python serve_minion.py).'
      );
    }
  }

  /* ------------------------------------------------------------------ Départ */

  chargerDossier('');
})();
