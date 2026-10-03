'use strict';
/* =========================================================
   GLOBAL SETTINGS (shared by the launcher and every game)
   sound:         true | false
   textSize:      'normal' | 'large'
   reducedMotion: true | false   (defaults to the operating system setting)
   quality:       'high' | 'low' (low = no shadows, no bloom, pixel ratio 1)
   ========================================================= */
CGB.settings = (() => {
  const osReduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const lowEnd = (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2) || (navigator.deviceMemory && navigator.deviceMemory <= 2);
  const defaults = { sound: true, textSize: 'normal', reducedMotion: osReduce, quality: lowEnd ? 'low' : 'high' };
  const saved = CGB.store.getJSON('settings', {});
  const s = Object.assign({}, defaults, saved && typeof saved === 'object' ? saved : {});
  const listeners = [];
  function apply() {
    const root = document.documentElement;
    root.dataset.text = s.textSize === 'large' ? 'large' : 'normal';
    root.dataset.motion = s.reducedMotion ? 'reduced' : 'full';
    root.dataset.quality = s.quality;
  }
  function set(key, value) {
    if (!(key in defaults)) return;
    s[key] = value;
    CGB.store.setJSON('settings', s);
    apply();
    listeners.forEach(fn => { try { fn(key, value, s); } catch (e) { /* a listener must not break the others */ } });
  }
  apply();
  return {
    get: k => s[k],
    all: () => Object.assign({}, s),
    set,
    onChange(fn) { listeners.push(fn); },
    /* Shared quality numbers for the 3D scenes */
    pixelRatio: () => s.quality === 'low' ? 1 : Math.min(window.devicePixelRatio || 1, 2),
    reduced: () => !!s.reducedMotion
  };
})();
