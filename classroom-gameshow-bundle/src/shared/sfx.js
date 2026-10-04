'use strict';
/* =========================================================
   SOUND: every effect is synthesised with Web Audio.
   No audio files, no music. Respects the global Sound setting.
   ========================================================= */
CGB.sfx = (() => {
  let ctx = null;
  function ac() {
    if (!ctx) { try { ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { ctx = null; } }
    return ctx;
  }
  function tone(freq, dur, type, vol, when, slide) {
    if (!CGB.settings.get('sound')) return;
    const c = ac(); if (!c || c.state !== 'running') return;
    const t = c.currentTime + (when || 0);
    const o = c.createOscillator(), g = c.createGain();
    o.type = type || 'sine';
    o.frequency.setValueAtTime(freq, t);
    if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(30, freq * slide), t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(vol || 0.1, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); g.connect(c.destination);
    o.start(t); o.stop(t + dur + 0.05);
  }
  /* Browsers only allow audio after a user gesture */
  function unlock() {
    if (!CGB.settings.get('sound')) return;
    const c = ac(); if (c && c.state === 'suspended') c.resume();
  }
  ['pointerdown', 'keydown'].forEach(ev => window.addEventListener(ev, unlock, { capture: true, passive: true }));
  return {
    tone, unlock,
    click() { tone(660, 0.05, 'triangle', 0.05); },
    correct() { [523, 659, 784].forEach((f, i) => tone(f, 0.16, 'triangle', 0.09, i * 0.07)); },
    wrong() { tone(220, 0.28, 'square', 0.05, 0, 0.7); },
    // "3, 2, 1, show me!" countdown beeps; the last one is higher and longer
    tick(go) { if (go) [880, 1175].forEach((f, i) => tone(f, 0.2, 'triangle', 0.08, i * 0.06)); else tone(660, 0.08, 'triangle', 0.06); }
  };
})();
