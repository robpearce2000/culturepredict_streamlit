'use strict';
/* =========================================================
   LAUNCHER MUSIC: one original loop, played with Web Audio so it repeats with no gap.
   It plays on the launcher only. Choosing a game plays a short fanfare and fades the music out; coming back
   fades it in again. It starts on the teacher's first click or key press (browsers block sound before that),
   follows the Sound and Music settings, and pauses while the tab is hidden. Nothing is shown if it can't play.
   ========================================================= */
CGB.music = (() => {
  const LEVEL = 0.2, FANFARE = 0.42, FADE_IN = 1.6, FADE_OUT = 1;   // gentle: the loop is quieter than the effects
  let ctx = null, master = null, loopBuf = null, fanBuf = null, src = null, loading = null;
  let gestured = false, hidden = document.hidden, wantOn = false, state = 'idle', gen = 0, fanfares = 0;

  const allowed = () => !!CGB.settings.get('sound') && CGB.settings.get('music') !== false;
  const onLauncher = () => !CGB.app || CGB.app.current === 'launcher';
  const bytes = b64 => { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a.buffer; };
  const decode = data => new Promise((res, rej) => { const p = ctx.decodeAudioData(bytes(data), res, rej); if (p && p.catch) p.catch(rej); });

  function ensure() {
    if (ctx) return ctx;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);
    } catch (e) { ctx = null; }
    return ctx;
  }
  function load() {
    if (loading) return loading;
    if (!ensure() || !CGB.MUSIC_DATA) return Promise.resolve(false);
    loading = Promise.all([decode(CGB.MUSIC_DATA.loop), decode(CGB.MUSIC_DATA.fanfare)])
      .then(([l, f]) => { loopBuf = l; fanBuf = f; return true; })
      .catch(() => false);
    return loading;
  }
  function ramp(to, secs) {
    const g = master.gain, t = ctx.currentTime;
    g.cancelScheduledValues(t); g.setValueAtTime(g.value, t); g.linearRampToValueAtTime(to, t + secs);
  }
  function stopSrc() { if (src) { try { src.stop(); } catch (e) { /* already stopped */ } src.disconnect(); src = null; } }

  /* the one place that decides whether music should be sounding right now */
  function sync() {
    const should = gestured && wantOn && allowed() && onLauncher() && !hidden;
    if (!ctx) { if (should) load().then(ok => ok && sync()); return; }
    const my = ++gen;
    if (should) {
      if (!loopBuf) { load().then(ok => ok && my === gen && sync()); return; }
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});
      if (!src) {
        src = ctx.createBufferSource(); src.buffer = loopBuf; src.loop = true; src.loopStart = 0; src.loopEnd = loopBuf.duration;
        src.connect(master); src.start();
      }
      ramp(LEVEL, FADE_IN); state = 'playing';
    } else if (src) {
      const quick = !allowed() || hidden;   // muting takes effect at once; leaving the launcher fades over a second
      if (quick) { master.gain.cancelScheduledValues(ctx.currentTime); master.gain.value = 0; stopSrc(); if (hidden && ctx.state === 'running') ctx.suspend().catch(() => {}); }
      else { ramp(0, FADE_OUT); const s = src; setTimeout(() => { if (src === s && my === gen) stopSrc(); }, FADE_OUT * 1000 + 80); }
      state = 'stopped';
    }
  }
  function firstGesture() {
    if (gestured) return;
    gestured = true;
    ['pointerdown', 'keydown'].forEach(ev => window.removeEventListener(ev, firstGesture, true));
    if (!ensure()) return;
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});
    sync();
  }
  ['pointerdown', 'keydown'].forEach(ev => window.addEventListener(ev, firstGesture, { capture: true, passive: true }));
  document.addEventListener('visibilitychange', () => { hidden = document.hidden; if (!hidden && ctx && ctx.state === 'suspended' && gestured) ctx.resume().catch(() => {}); sync(); });
  CGB.settings.onChange(k => { if (k === 'sound' || k === 'music') sync(); });

  return {
    /* the app calls this whenever the screen changes; chosen = a game was just picked from the launcher */
    screen(isLauncher, chosen) {
      wantOn = isLauncher;
      if (chosen && gestured && allowed() && ctx && fanBuf) {
        fanfares++;
        const s = ctx.createBufferSource(), g = ctx.createGain(); g.gain.value = FANFARE;
        s.buffer = fanBuf; s.connect(g); g.connect(ctx.destination); s.start();
      }
      sync();
    },
    /* Start from the launcher's own first paint: the teacher's first click or key press also starts it */
    init() { wantOn = onLauncher(); sync(); },
    /* @test-only: what the music is doing, for tests */
    test: {
      info: () => ({ state, gestured, hasCtx: !!ctx, playing: !!src, gain: master ? master.gain.value : 0, ctxState: ctx ? ctx.state : 'none', fanfares, loopSeconds: loopBuf ? loopBuf.duration : 0, loaded: !!loopBuf && !!fanBuf }),
      settle: () => { if (ctx && master) { master.gain.cancelScheduledValues(ctx.currentTime); } }
    }
    /* @end-test-only */
  };
})();
