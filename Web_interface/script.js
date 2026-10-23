/* ============================================================================
   Yoda — logique de l'interface
   ----------------------------------------------------------------------------
   Trois rôles, séparés volontairement :

     1. `yoda`  → la bouche. Une seule chose à retenir : `yoda.onToken()` à
                  chaque token reçu, et la bouche bouge.
     2. `bulle` → le texte affiché.
     3. `demander()` → ce qui se passe quand on pose une question.

   Plus, à part : `reagir()`, déclenchée quand on clique sur Yoda (il ferme
   les yeux et grogne).

   Aujourd'hui `demander()` invente une réponse et la fait défiler token par
   token. Le jour où le modèle tourne derrière un serveur, il suffit de
   remplacer cette fonction (voir le bloc commenté tout en bas).
   ========================================================================== */

const yodaEl   = document.getElementById('yoda');
const bulleEl  = document.getElementById('bulle');
const texteEl  = document.getElementById('bulle-texte');
const echoEl   = document.getElementById('echo');
const formEl   = document.getElementById('ask');
const inputEl  = document.getElementById('question');
const sendEl   = document.getElementById('send');

const attendre = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ------------------------------------------------------------------ La bouche
   `--mouth-open` va de 0 (fermée) à 1 (grande ouverte). C'est la seule chose
   que le CSS attend pour animer la mâchoire.                               */

const MOUTH_CLOSED = 0;
const MOUTH_MAX = 1;

function setMouth(value) {
  const v = Math.min(MOUTH_MAX, Math.max(MOUTH_CLOSED, value));
  yodaEl.style.setProperty('--mouth-open', v.toFixed(3));
}

/* Respiration au repos : la bouche s'entrouvre à peine, lentement. */
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

/* Un battement de bouche. Appelé une fois par token. */
let flapTimer = null;

function flap() {
  setMouth(0.3 + Math.random() * 0.7);
  clearTimeout(flapTimer);
  flapTimer = setTimeout(() => {
    if (talking) setMouth(0.06);
  }, 90);
}

/* L'API que le reste du code utilise. */
const yoda = {
  setMouth,

  startTalking() {
    talking = true;
    clearTimeout(flapTimer);
    setMouth(0.12);
  },

  stopTalking() {
    talking = false;
    clearTimeout(flapTimer);
    setMouth(MOUTH_CLOSED);
    idleStart = performance.now();
  },

  /** À appeler à chaque token reçu : la bouche s'ouvre d'un coup. */
  onToken() {
    if (!talking) yoda.startTalking();
    flap();
  },
};

/* ------------------------------------------------------- Réaction au clic
   Yoda ferme les yeux (classe `is-reaction`, gérée en CSS) et ouvre la bouche
   en trois temps, comme s'il grognait. On ne le fait pas pendant qu'il parle
   ni pendant qu'il réfléchit, pour ne pas se battre avec l'animation.      */

let reacting = false;

async function reagir() {
  if (reacting || talking || occupe) return;

  reacting = true;
  talking = true; // fige la respiration le temps de la réaction
  clearTimeout(flapTimer);
  yodaEl.classList.add('is-reaction');

  const trame = [0.95, 0.3, 0.88, 0.22, 0.9, 0.05];

  for (const ouverture of trame) {
    setMouth(ouverture);
    await attendre(105);
  }

  yodaEl.classList.remove('is-reaction');
  yoda.stopTalking();
  reacting = false;
}

yodaEl.addEventListener('click', reagir);

yodaEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    reagir();
  }
});

/* -------------------------------------------------------------------- La bulle */

const bulle = {
  /** Vide la bulle et affiche le curseur clignotant. */
  start() {
    texteEl.textContent = '';
    bulleEl.dataset.state = 'parle';
  },

  /** Ajoute un morceau de texte (un token, ou quelques caractères). */
  append(morceau) {
    texteEl.textContent += morceau;
  },

  /** Fin de la réponse. */
  finish() {
    bulleEl.dataset.state = 'idle';
  },

  /** Message système ("…" pendant la réflexion, message d'erreur…). */
  set(texte) {
    texteEl.textContent = texte;
    bulleEl.dataset.state = 'idle';
  },
};

/* ---------------------------------------------------------- Réponses d'attente
   En attendant le vrai modèle. Une réponse est choisie à partir de la question,
   pour qu'une même question donne toujours la même réponse.                 */

const REPONSES = [
  "La peur, un feu elle est. Elle éclaire autant qu'elle brûle. Regarde-la, ne la fuis pas.",
  "Le doute, un compagnon fidèle il est. Trop vite marcher, il ralentit. Mais jamais il ne ment.",
  "Ton chemin, tu le portes. Nul autre ne peut marcher à ta place, ni pour toi se tromper.",
  "Patience, tu dois apprendre. Le temps, un maître sévère il est, mais jamais injuste.",
  "Ce que tu cherches, souvent déjà là il est. Tes yeux, il suffit de les ouvrir.",
  "La colère, un nuage elle est. Passe, elle passe. Mais ce qu'elle cache, longtemps il reste.",
  "Petit tu es, et pourtant tout l'univers en toi respire. L'oublier, voilà le seul vrai danger.",
  "Une question poser, c'est déjà la moitié du chemin. L'autre moitié, marcher tu dois.",
  "Le silence, une réponse aussi il est. Écouter, tu apprendras plus vite qu'en parlant.",
  "Rien ne sert de courir vers la lumière. La lumière, à ta vitesse elle vient.",
];

function choisirReponse(question) {
  let empreinte = 0;
  for (const caractere of question) {
    empreinte = (empreinte * 31 + caractere.codePointAt(0)) % 997;
  }
  return REPONSES[empreinte % REPONSES.length];
}

/* ------------------------------------------------------------ Faux flux de tokens
   Découpe le texte en petits morceaux et les livre un par un, avec un délai
   irrégulier : c'est exactement le rythme d'un vrai flux.                   */

async function fluxSimule(texte, surToken) {
  // On garde les espaces avec le mot qui précède, pour ne pas les perdre.
  const morceaux = texte.match(/\S+\s*|\s+/g) ?? [texte];

  for (const morceau of morceaux) {
    await attendre(26 + Math.random() * 48);
    surToken(morceau);
  }
}

/* -------------------------------------------------------------- La conversation */

let occupe = false;

async function demander(question) {
  if (occupe) return;
  occupe = true;

  echoEl.textContent = `Toi — ${question}`;
  echoEl.hidden = false;

  document.body.classList.add('is-thinking');
  inputEl.disabled = true;
  sendEl.disabled = true;
  bulle.set('…');

  // Le temps qu'un modèle réfléchisse, grossièrement.
  await attendre(420 + Math.random() * 320);

  const reponse = choisirReponse(question);

  document.body.classList.remove('is-thinking');

  bulle.start();
  yoda.startTalking();

  await fluxSimule(reponse, (morceau) => {
    bulle.append(morceau);
    yoda.onToken();
  });

  yoda.stopTalking();
  bulle.finish();

  occupe = false;
  inputEl.disabled = false;
  sendEl.disabled = false;
  inputEl.focus();
}

formEl.addEventListener('submit', (event) => {
  event.preventDefault();

  const question = inputEl.value.trim();
  if (!question) {
    inputEl.focus();
    return;
  }

  inputEl.value = '';
  demander(question);
});

/* ============================================================================
   À BRANCHER PLUS TARD — le vrai modèle
   ----------------------------------------------------------------------------
   Le contrat ne change pas : à chaque token reçu, on ajoute le texte à la
   bulle et on appelle `yoda.onToken()`.

   async function demander(question) {
     if (occupe) return;
     occupe = true;

     echoEl.textContent = `Toi — ${question}`;
     echoEl.hidden = false;

     document.body.classList.add('is-thinking');
     inputEl.disabled = true;
     sendEl.disabled = true;
     bulle.set('…');

     try {
       const reponse = await fetch('/api/chat', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ question }),
       });

       if (!reponse.ok) throw new Error(`HTTP ${reponse.status}`);

       document.body.classList.remove('is-thinking');
       bulle.start();
       yoda.startTalking();

       const lecteur = reponse.body.getReader();
       const decodeur = new TextDecoder();

       while (true) {
         const { value, done } = await lecteur.read();
         if (done) break;

         const morceau = decodeur.decode(value, { stream: true });
         bulle.append(morceau);
         yoda.onToken();          // ← la bouche bouge à chaque token
       }

       bulle.finish();
     } catch (erreur) {
       bulle.set("La Force, troublée elle est. Réessayer, tu dois.");
       console.error(erreur);
     } finally {
       document.body.classList.remove('is-thinking');
       yoda.stopTalking();
       occupe = false;
       inputEl.disabled = false;
       sendEl.disabled = false;
       inputEl.focus();
     }
   }
   ========================================================================== */
