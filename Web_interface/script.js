/* ============================================================================
   Traducteur minion — logique de l'interface
   ========================================================================== */

const minionEl = document.getElementById('minion');
const bulleEl  = document.getElementById('bulle');
const texteEl  = document.getElementById('bulle-texte');
const echoEl   = document.getElementById('echo');
const formEl   = document.getElementById('ask');
const inputEl  = document.getElementById('question');
const sendEl   = document.getElementById('send');
const sensEl   = document.getElementById('sens');
const sensTexteEl = document.getElementById('sens-texte');

const attendre = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ------------------------------------------------------------------ La bouche */
const MOUTH_CLOSED = 0;
const MOUTH_MAX = 1;

function setMouth(value) {
  const v = Math.min(MOUTH_MAX, Math.max(MOUTH_CLOSED, value));
  minionEl.style.setProperty('--mouth-open', v.toFixed(3));
}

/* Respiration au repos */
let talking = false;
let idleStart = performance.now();

function breathe(now) {
  if (!talking) {
    const t = (now - idleStart) / 1000;
    setMouth(0.04 + 0.05 * Math.sin(t * 1.6));
  }
  requestAnimationFrame(breathe);
}

requestAnimationFrame(breathe);

let flapTimer = null;

function flap() {
  setMouth(0.3 + Math.random() * 0.7);
  clearTimeout(flapTimer);
  flapTimer = setTimeout(() => {
    if (talking) setMouth(0.06);
  }, 90);
}

const minion = {
  setMouth,

  startTalking() {
    talking = true;
    clearTimeout(flapTimer);
  },

  stopTalking() {
    talking = false;
    clearTimeout(flapTimer);
    minionEl.classList.remove('is-talking');
    minionEl.classList.remove('is-answering');
    setMouth(MOUTH_CLOSED);
    idleStart = performance.now();
  },

  onToken() {
    if (!talking) minion.startTalking();
    flap();
  },
};

/* ------------------------------------------------------- Réaction au clic */
let reacting = false;

async function reagir() {
  if (reacting || talking || occupe) return;

  reacting = true;
  talking = true;
  clearTimeout(flapTimer);
  minionEl.classList.add('is-reaction');

  const trame = [0.95, 0.3, 0.88, 0.22, 0.9, 0.05];

  for (const ouverture of trame) {
    setMouth(ouverture);
    await attendre(105);
  }

  minionEl.classList.remove('is-reaction');
  minion.stopTalking();
  reacting = false;
}

minionEl.addEventListener('click', reagir);

minionEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    reagir();
  }
});

/* -------------------------------------------------------------------- La bulle */
const bulle = {
  start() {
    texteEl.textContent = '';
    bulleEl.dataset.state = 'parle';
  },

  append(morceau) {
    texteEl.textContent += morceau;
  },

  finish() {
    bulleEl.dataset.state = 'idle';
  },

  set(texte) {
    texteEl.textContent = texte;
    bulleEl.dataset.state = 'idle';
  },
};

/* -------------------------------------------------------------- Le sens de traduction */
const TEXTES = {
  'fr-minion': {
    bouton: 'Français → Minion',
    aria: 'Sens de traduction : français vers minion. Cliquer pour inverser.',
    exemple: 'Bonjour mon ami !',
    accueil: 'Bello ! Écris en français, et en minion je te le dis !',
  },
  'minion-fr': {
    bouton: 'Minion → Français',
    aria: 'Sens de traduction : minion vers français. Cliquer pour inverser.',
    exemple: 'Bello amigo !',
    accueil: 'Bello ! Écris en minion, et en français je te le dis !',
  },
};

let sens = 'fr-minion';

function appliquerSens() {
  const reglages = TEXTES[sens];

  sensTexteEl.textContent = reglages.bouton;
  sensEl.setAttribute('aria-label', reglages.aria);
  inputEl.placeholder = reglages.exemple;

  bulle.set(reglages.accueil);
  echoEl.hidden = true;
}

sensEl.addEventListener('click', () => {
  if (occupe) return;

  sens = sens === 'fr-minion' ? 'minion-fr' : 'fr-minion';
  inputEl.value = '';
  appliquerSens();
  inputEl.focus();
});

/* -------------------------------------------------------------- La conversation */
let occupe = false;

function verrouiller(actif) {
  inputEl.disabled = actif;
  sendEl.disabled = actif;
  sensEl.disabled = actif;

  if (!actif) inputEl.focus();
}

async function demander(texte) {
  if (occupe) return;
  occupe = true;

  echoEl.textContent = `Toi — ${texte}`;
  echoEl.hidden = false;

  document.body.classList.add('is-thinking');
  verrouiller(true);
  bulle.set('…');

  try {
    console.log('Envoi requête:', texte, sens);
    const reponse = await fetch('/api/traduire', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texte, sens }),
    });
    console.log('Réponse:', reponse.status);

    if (!reponse.ok) throw new Error(`HTTP ${reponse.status}`);

    const traduction = await reponse.text();
    console.log('Traduction:', traduction);

    document.body.classList.remove('is-thinking');
    bulle.start();
    minion.startTalking();
    // Uniquement pendant la réponse - bras levés
    minionEl.classList.add('is-answering');

    const morceaux = traduction.match(/\S+\s*|\s+/g) ?? [traduction];

    for (const morceau of morceaux) {
      await attendre(24 + Math.random() * 36);
      bulle.append(morceau);
      minion.onToken();
    }

    bulle.finish();
  } catch (erreur) {
    bulle.set('Bello ?! Quelque chose a cassé. Réessaie, tu peux.');
    console.error(erreur);
  } finally {
    document.body.classList.remove('is-thinking');
    minionEl.classList.remove('is-answering');
    minion.stopTalking();
    occupe = false;
    verrouiller(false);
  }
}

formEl.addEventListener('submit', (event) => {
  event.preventDefault();

  const texte = inputEl.value.trim();
  if (!texte) {
    inputEl.focus();
    return;
  }

  inputEl.value = '';
  demander(texte);
});

appliquerSens();
