'use strict';
/* =========================================================
   STORAGE
   Everything lives in localStorage on this device. Every call is
   wrapped so the bundle still runs (without saving) if storage is
   blocked, full or unavailable.
   ========================================================= */
window.CGB = window.CGB || {};
CGB.VERSION = '1.0.0';
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
