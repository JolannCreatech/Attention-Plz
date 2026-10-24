(() => {
  'use strict';

  const minionEl = document.getElementById('minion');
  const bulleEl = document.getElementById('bulle');
  const texteEl = document.getElementById('bulle-texte');
  const echoEl = document.getElementById('echo');
  const formEl = document.getElementById('ask');
  const inputEl = document.getElementById('question');
  const sendEl = document.getElementById('send');
  const sensEl = document.getElementById('sens');
  const sensTexteEl = document.getElementById('sens-texte');
  const etatEl = document.getElementById('etat');

  if (!minionEl || !bulleEl || !texteEl || !formEl || !inputEl || !sendEl || !sensEl || !sensTexteEl) {
    console.error('Interface minion : éléments HTML introuvables.');
    return;
  }

  const attendre = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /* ------------------------------------------------------------------ La bouche */
  const MOUTH_CLOSED = 0;
  const MOUTH_MAX = 1;

  function setMouth(value) {
    const v = Math.min(MOUTH_MAX, Math.max(MOUTH_CLOSED, value));
    minionEl.style.setProperty('--mouth-open', v.toFixed(3));
  }

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

  /* ------------------------------------------------------- Réaction */
  let reacting = false;
  let occupe = false;

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

  /* -------------------------------------------------------------------- Bulle */
  const bulle = {
    start() { texteEl.textContent = ''; bulleEl.dataset.state = 'parle'; },
    append(morceau) { texteEl.textContent += morceau; },
    finish() { bulleEl.dataset.state = 'idle'; },
    set(texte) { texteEl.textContent = texte; bulleEl.dataset.state = 'idle'; },
  };

  /* -------------------------------------------------------------- Sens */
  const TEXTES = {
    'fr-minion': { bouton: 'Français → Minion', aria: 'Sens de traduction : français vers minion. Cliquer pour inverser.', exemple: 'Bonjour mon ami !', accueil: 'Bello ! Écris en français, et en minion je te le dis !' },
    'minion-fr': { bouton: 'Minion → Français', aria: 'Sens de traduction : minion vers français. Cliquer pour inverser.', exemple: 'Bello amigo !', accueil: 'Bello ! Écris en minion, et en français je te le dis !' },
  };

  let sens = 'fr-minion';

  function appliquerSens() {
    const r = TEXTES[sens];
    sensTexteEl.textContent = r.bouton;
    sensEl.setAttribute('aria-label', r.aria);
    inputEl.placeholder = r.exemple;
    bulle.set(r.accueil);
    if (echoEl) echoEl.hidden = true;
  }

  sensEl.addEventListener('click', () => {
    if (occupe) return;
    sens = sens === 'fr-minion' ? 'minion-fr' : 'fr-minion';
    inputEl.value = '';
    appliquerSens();
    inputEl.focus();
  });

  /* ------------------------------------------------------------ API */
  function setEtat(t, txt) {
    if (!etatEl) return;
    etatEl.dataset.etat = t;
    etatEl.textContent = txt;
  }

  function getApiUrl() {
    if (location.port === '8000') return location.origin + '/api/traduire';
    return 'http://localhost:8000/api/traduire';
  }

  async function traduire(texte, direction) {
    const url = getApiUrl();
    setEtat('ok', 'modèle connecté');
    const c = new AbortController();
    const timeout = setTimeout(() => c.abort(), 120000);
    try {
      const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texte, sens: direction }),
        signal: c.signal,
        cache: 'no-store',
      });
      if (!r.ok) {
        const d = await r.text().catch(() => '');
        throw new Error('HTTP ' + r.status + (d ? ' — ' + d.trim() : ''));
      }
      const t = (await r.text()).trim();
      if (!t) throw new Error('réponse vide du modèle');
      return t;
    } catch (e) {
      if (e && e.name === 'AbortError') throw new Error('le modèle met trop de temps à répondre');
      if (e && (e.message.indexOf('Failed to fetch') !== -1 || e.message.indexOf('NetworkError') !== -1)) {
        throw new Error('impossible de contacter http://localhost:8000 — lance .venv/bin/python serve_minion.py');
      }
      throw e;
    } finally {
      clearTimeout(timeout);
    }
  }

  /* -------------------------------------------------------------- Conversation */
  function verrouiller(actif) {
    inputEl.disabled = actif;
    sendEl.disabled = actif;
    sensEl.disabled = actif;
    if (!actif) inputEl.focus();
  }

  async function envoyer() {
    const texte = inputEl.value.trim();
    if (!texte || occupe) return;
    occupe = true;
    if (echoEl) { echoEl.textContent = `Toi — ${texte}`; echoEl.hidden = false; }
    document.body.classList.add('is-thinking');
    verrouiller(true);
    bulle.set('…');
    try {
      const traduction = await traduire(texte, sens);
      document.body.classList.remove('is-thinking');
      bulle.start();
      minion.startTalking();
      minionEl.classList.add('is-answering');
      const morceaux = traduction.match(/\S+\s*|\s+/g) ?? [traduction];
      for (const morceau of morceaux) {
        await attendre(24 + Math.random() * 36);
        bulle.append(morceau);
        minion.onToken();
      }
      bulle.finish();
    } catch (e) {
      bulle.set('Bello ?! ' + e.message);
      console.error(e);
    } finally {
      document.body.classList.remove('is-thinking');
      minionEl.classList.remove('is-answering');
      minion.stopTalking();
      occupe = false;
      verrouiller(false);
      inputEl.value = '';
    }
  }

  formEl.addEventListener('submit', (e) => { e.preventDefault(); e.stopPropagation(); envoyer(); }, true);
  formEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); envoyer(); } }, true);
  inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); envoyer(); } }, true);
  sendEl.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); envoyer(); }, true);

  appliquerSens();
  setEtat('ok', 'prêt — envoie vers localhost:8000');

  window.__minionPret = true;
})();
