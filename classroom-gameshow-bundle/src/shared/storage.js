'use strict';
/* =========================================================
   STORAGE
   Everything lives in localStorage on this device. Every call is
   wrapped so Showtime still runs (without saving) if storage is
   blocked, full or unavailable.
   ========================================================= */
window.CGB = window.CGB || {};
CGB.VERSION = '1.5.0';
/* Every random choice that affects play (question order, boards, counters, physics
   nudges) goes through CGB.random, so tests can make runs repeatable. Purely visual
   randomness (sparkles, camera shake) uses Math.random. */
CGB.random = Math.random;
/* @test-only: removed from the shipped file by build.js */
CGB.test = {
  // mulberry32: a small, fast seeded generator
  seed(n) {
    let a = n >>> 0;
    CGB.random = () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
};
if (window.__SHOWTIME_SEED__ != null) CGB.test.seed(window.__SHOWTIME_SEED__);
/* @end-test-only */
/* A quiet thank-you line at the bottom of every end screen (text only, no link) */
CGB.REVIEW_NOTE = '<p class="cgb-review">Enjoying Showtime? A short review on Tes helps other teachers find it. Thank you!</p>';
CGB.store = (() => {
  const PREFIX = 'cgb.';
  let ok = true;
  try { const k = PREFIX + '__test'; localStorage.setItem(k, '1'); localStorage.removeItem(k); } catch (e) { ok = false; }
  const memory = {};   // fallback so a session still works when storage is blocked
  function get(k) {
    try { const v = localStorage.getItem(PREFIX + k); return v; } catch (e) { return Object.prototype.hasOwnProperty.call(memory, k) ? memory[k] : null; }
  }
  function set(k, v) {
    memory[k] = v;
    try { localStorage.setItem(PREFIX + k, v); return true; } catch (e) { return false; }
  }
  function del(k) {
    delete memory[k];
    try { localStorage.removeItem(PREFIX + k); } catch (e) { /* ignore */ }
  }
  function getJSON(k, fallback) {
    try { const v = JSON.parse(get(k)); return v == null ? fallback : v; } catch (e) { return fallback; }
  }
  function setJSON(k, v) { return set(k, JSON.stringify(v)); }
  /* Raw access to keys written by the earlier stand-alone versions (for a one-off import) */
  function rawKeys() { try { return Object.keys(localStorage); } catch (e) { return []; } }
  function rawGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  return { get, set, del, getJSON, setJSON, rawKeys, rawGet, get available() { return ok; } };
})();

CGB.escapeHtml = t => String(t).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
