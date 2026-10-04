'use strict';
/* =========================================================
   SHARED QUESTION BANK
   - Built-in packs plus the teacher's own sets
   - One active set, used by every game
   - Wrong-answer history per player name, shared by every game
   - JSON backup (export / import) of sets and history
   ========================================================= */
CGB.bank = (() => {
  const store = CGB.store;

  /* The plain-text format: Subject / Topic lines apply until changed; Q: then A:.
     An optional "Difficulty: 1-5" line (or "Level:") applies to the questions after it,
     until the next Difficulty or Topic line. Older games simply ignore it. */
  function parse(text) {
    const out = [];
    let subject = '', topic = '', level = 0, pendingQ = null;
    String(text || '').split('\n').forEach(line => {
      const t = line.trim();
      if (/^subject:/i.test(t)) { subject = t.replace(/^subject:/i, '').trim(); level = 0; }
      else if (/^topic:/i.test(t)) { topic = t.replace(/^topic:/i, '').trim(); level = 0; }
      else if (/^(difficulty|level):/i.test(t)) { const n = parseInt(t.replace(/^(difficulty|level):/i, ''), 10); level = n >= 1 && n <= 5 ? n : 0; }
      else if (/^q:/i.test(t)) pendingQ = t.replace(/^q:/i, '').trim();
      else if (/^a:/i.test(t) && pendingQ) {
        const q = { subject: subject || 'Custom', topic: topic || 'General', q: pendingQ, a: t.replace(/^a:/i, '').trim() };
        if (level) q.level = level;
        out.push(q);
        pendingQ = null;
      }
    });
    return out;
  }
  /* Turn questions back into the plain-text format (used for editing and copying) */
  function toText(questions) {
    const lines = [];
    let s = null, t = null, l = 0;
    questions.forEach(q => {
      if (q.subject !== s || q.topic !== t) {
        if (lines.length) lines.push('');
        lines.push('Subject: ' + q.subject, 'Topic: ' + q.topic);
        s = q.subject; t = q.topic; l = 0;
      }
      if ((q.level || 0) !== l && q.level) { lines.push('Difficulty: ' + q.level); }
      l = q.level || l;
      lines.push('Q: ' + q.q, 'A: ' + q.a);
    });
    return lines.join('\n');
  }
  /* A rough difficulty score for questions without a Difficulty line:
     longer, explanatory or calculation answers count as harder. */
  function estimateLevel(q) {
    if (q.level) return q.level * 10;
    const words = q.a.split(/\s+/).length;
    let score = Math.min(words, 24) / 3;
    if (/\d/.test(q.q) && /\d/.test(q.a)) score += 3;                         // a calculation
    if (/^(why|explain|how does|how is|how do|describe|what is the difference)/i.test(q.q)) score += 2.5;
    if (/equation|formula/i.test(q.q)) score += 1.5;
    if (/true or false|: nervous or hormonal|which is faster|which has longer/i.test(q.q)) score -= 2;
    if (words <= 2) score -= 1;
    return score;
  }

  const P = CGB.PACKS;
  const bio = parse(P.biology), chem = parse(P.chemistry), phys = parse(P.physics);
  const BUILTIN = [
    { id: 'gcse-combined-mix', name: 'GCSE Combined Science starter pack: all three sciences (AQA-style)', short: 'Combined Science: mixed', questions: bio.concat(chem, phys) },
    { id: 'gcse-combined-biology', name: 'GCSE Combined Science starter pack: Biology (AQA-style)', short: 'Combined Science: Biology', questions: bio },
    { id: 'gcse-combined-chemistry', name: 'GCSE Combined Science starter pack: Chemistry (AQA-style)', short: 'Combined Science: Chemistry', questions: chem },
    { id: 'gcse-combined-physics', name: 'GCSE Combined Science starter pack: Physics (AQA-style)', short: 'Combined Science: Physics', questions: phys },
    { id: 'homeostasis-l1', name: 'Homeostasis L1 (Biology)', short: 'Homeostasis L1', questions: parse(P.homeostasis) }
  ].map(s => Object.assign(s, { builtin: true }));

  let custom = [];
  let activeId = BUILTIN[0].id;
  let history = {};      // key (lower-case name) -> { name, entries: [ {subject, topic, q, a, date, game} ] }
  const listeners = [];
  const emit = () => listeners.forEach(fn => { try { fn(); } catch (e) { /* ignore */ } });

  function validQuestion(q) { return q && typeof q.q === 'string' && typeof q.a === 'string' && q.q.trim() && q.a.trim(); }
  function cleanQuestion(q) {
    const c = { subject: String(q.subject || 'Custom').slice(0, 80), topic: String(q.topic || 'General').slice(0, 120), q: String(q.q).slice(0, 600), a: String(q.a).slice(0, 600) };
    const l = parseInt(q.level, 10); if (l >= 1 && l <= 5) c.level = l;
    return c;
  }
  function cleanSet(s) {
    if (!s || !Array.isArray(s.questions)) return null;
    const qs = s.questions.filter(validQuestion).map(cleanQuestion);
    if (!qs.length) return null;
    return { id: String(s.id || newId()), name: String(s.name || 'My questions').slice(0, 60), questions: qs, created: s.created || new Date().toISOString().slice(0, 10) };
  }
  function newId() { return 'set-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7); }

  function load() {
    const c = store.getJSON('sets', []);
    custom = Array.isArray(c) ? c.map(cleanSet).filter(Boolean) : [];
    const h = store.getJSON('history', {});
    history = h && typeof h === 'object' && !Array.isArray(h) ? h : {};
    const a = store.get('activeSet');
    if (a && all().some(s => s.id === a)) activeId = a;
  }
  function saveSets() { return store.setJSON('sets', custom); }
  function saveHistory() { return store.setJSON('history', history); }

  /* One-off import of data saved by the two earlier stand-alone games on this device */
  function migrateLegacy() {
    if (store.get('migrated') === '1') return;
    const keys = store.rawKeys();
    const legacySets = [];
    [['tp_sets'], ['questionSets']].forEach(([k]) => {
      try { const v = JSON.parse(store.rawGet(k)); if (Array.isArray(v)) v.forEach(s => legacySets.push(s)); } catch (e) { /* skip */ }
    });
    legacySets.forEach(s => { const c = cleanSet(s); if (c && !custom.some(x => x.id === c.id)) custom.push(c); });
    keys.forEach(k => {
      const m = /^(?:tp_)?wronglog:(.+)$/.exec(k);
      if (!m) return;
      try {
        const arr = JSON.parse(store.rawGet(k));
        if (Array.isArray(arr)) arr.forEach(e => { if (validQuestion(e)) addHistoryEntry(m[1], Object.assign({}, cleanQuestion(e), { date: e.date || '', game: '' })); });
      } catch (e) { /* skip */ }
    });
    try { const host = JSON.parse(store.rawGet('tp_host')); if (host && !store.get('host')) store.setJSON('host', host); } catch (e) { /* skip */ }
    try { const n = JSON.parse(store.rawGet('tp_names')); if (Array.isArray(n) && !store.get('names')) store.setJSON('names', n); } catch (e) { /* skip */ }
    saveSets(); saveHistory();
    store.set('migrated', '1');
  }

  function all() { return BUILTIN.concat(custom); }
  function get(id) { return all().find(s => s.id === id) || null; }
  function active() { return get(activeId) || BUILTIN[0]; }
  function setActive(id) { if (!get(id)) return false; activeId = id; store.set('activeSet', id); emit(); return true; }
  function summary(set) {
    const subjects = {}, topics = {};
    set.questions.forEach(q => { subjects[q.subject] = 1; topics[q.topic] = 1; });
    return { subjects: Object.keys(subjects), topics: Object.keys(topics), count: set.questions.length };
  }
  function addSet(name, text) {
    const qs = parse(text);
    if (!qs.length) return { ok: false, error: 'No Q: and A: pairs found. Check the format.' };
    const set = cleanSet({ id: newId(), name: (name || '').trim() || 'My questions', questions: qs });
    custom.push(set);
    if (!saveSets()) { /* still usable for this session */ }
    activeId = set.id; store.set('activeSet', set.id);
    emit();
    return { ok: true, set, saved: store.available };
  }
  function updateSet(id, name, text) {
    const s = custom.find(x => x.id === id);
    if (!s) return { ok: false, error: 'Built-in packs cannot be changed. Save a copy instead.' };
    if (text != null) {
      const qs = parse(text);
      if (!qs.length) return { ok: false, error: 'No Q: and A: pairs found. Check the format.' };
      s.questions = qs.map(cleanQuestion);
    }
    if (name != null && name.trim()) s.name = name.trim().slice(0, 60);
    saveSets(); emit();
    return { ok: true, set: s };
  }
  function deleteSet(id) {
    const before = custom.length;
    custom = custom.filter(s => s.id !== id);
    if (custom.length === before) return false;
    if (activeId === id) { activeId = BUILTIN[0].id; store.set('activeSet', activeId); }
    saveSets(); emit();
    return true;
  }

  /* ---------- History ---------- */
  const key = n => String(n || '').trim().toLowerCase();
  function addHistoryEntry(name, e) {
    const k = key(name); if (!k) return;
    if (!history[k]) history[k] = { name: String(name).trim(), entries: [] };
    history[k].entries.push(e);
    while (history[k].entries.length > 300) history[k].entries.shift();
  }
  function logWrong(name, q, game) {
    if (!q) return;
    addHistoryEntry(name, { subject: q.subject, topic: q.topic, q: q.q, a: q.a, date: new Date().toISOString().slice(0, 10), game: game || '' });
    saveHistory();
  }
  function wrongLog(name) { const h = history[key(name)]; return h ? h.entries.slice() : []; }
  function weakTopics(name, n) {
    const m = {};
    wrongLog(name).forEach(e => { m[e.topic] = (m[e.topic] || 0) + 1; });
    return Object.entries(m).sort((a, b) => b[1] - a[1]).slice(0, n || 5);
  }
  function clearHistory(name) { delete history[key(name)]; saveHistory(); emit(); }
  /* Fill a game's question-set dropdown. Options use the short name so they are
     never cut off; the question count goes in the field's label instead. */
  function fillSelect(sel) {
    const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    sel.innerHTML = all().map(s => `<option value="${esc(s.id)}" title="${esc(s.name)}">${esc(s.short || s.name)}</option>`).join('');
    const a = active();
    sel.value = a.id;
    sel.title = a.name;
    const lab = sel.id && document.querySelector(`label[for="${sel.id}"]`);
    if (lab) lab.textContent = `Question set: ${a.questions.length} questions, shared with every game`;
  }
  function players() { return Object.values(history).map(h => ({ name: h.name, count: h.entries.length })).sort((a, b) => a.name.localeCompare(b.name)); }

  /* ---------- Question picking (each game session gets its own picker) ---------- */
  function createPicker() {
    const used = new Set();
    let setId = null;
    return {
      reset() { used.clear(); },
      pick(playerName, focusWeak) {
        const set = active();
        if (set.id !== setId) { used.clear(); setId = set.id; }
        const bank = set.questions;
        let pool = bank.map((q, i) => i).filter(i => !used.has(i));
        if (!pool.length) { used.clear(); pool = bank.map((q, i) => i); }
        if (focusWeak && playerName) {
          const weak = new Set(wrongLog(playerName).map(e => e.topic));
          const weakPool = pool.filter(i => weak.has(bank[i].topic));
          if (weakPool.length && CGB.random() < 0.6) pool = weakPool;
        }
        const i = pool[Math.floor(CGB.random() * pool.length)];
        used.add(i);
        return bank[i];
      }
    };
  }

  /* ---------- Backup ---------- */
  function exportData() {
    return {
      format: 'classroom-gameshow-bundle-backup', formatVersion: 1, appVersion: CGB.VERSION,
      exported: new Date().toISOString(),
      activeSet: activeId,
      sets: JSON.parse(JSON.stringify(custom)),
      history: JSON.parse(JSON.stringify(history))
    };
  }
  /* Merge a backup into this device. Nothing already here is deleted. */
  function importData(data) {
    if (!data || data.format !== 'classroom-gameshow-bundle-backup') return { ok: false, error: 'This file is not a Showtime: Classroom Gameshows backup.' };
    let added = 0, updated = 0, entries = 0;
    (Array.isArray(data.sets) ? data.sets : []).forEach(s => {
      const c = cleanSet(s); if (!c) return;
      const existing = custom.find(x => x.id === c.id);
      if (!existing) { custom.push(c); added++; }
      else if (JSON.stringify(existing) !== JSON.stringify(c)) { Object.assign(existing, c); updated++; }
    });
    const h = data.history && typeof data.history === 'object' ? data.history : {};
    Object.keys(h).forEach(k => {
      const rec = h[k]; if (!rec || !Array.isArray(rec.entries)) return;
      const name = rec.name || k;
      const mine = wrongLog(name);
      // count what is already here so re-importing the same file adds nothing
      const have = {};
      mine.forEach(e => { const sig = JSON.stringify([e.subject, e.topic, e.q, e.a, e.date, e.game || '']); have[sig] = (have[sig] || 0) + 1; });
      rec.entries.forEach(e => {
        if (!validQuestion(e)) return;
        const c = Object.assign(cleanQuestion(e), { date: String(e.date || ''), game: String(e.game || '') });
        const sig = JSON.stringify([c.subject, c.topic, c.q, c.a, c.date, c.game]);
        if (have[sig]) { have[sig]--; return; }
        addHistoryEntry(name, c); entries++;
      });
    });
    if (data.activeSet && get(data.activeSet)) { activeId = data.activeSet; store.set('activeSet', activeId); }
    saveSets(); saveHistory(); emit();
    return { ok: true, added, updated, entries };
  }

  load();
  migrateLegacy();
  return {
    parse, toText, estimateLevel, all, get, active, setActive, summary, addSet, updateSet, deleteSet,
    logWrong, wrongLog, weakTopics, clearHistory, players, createPicker, fillSelect,
    exportData, importData, onChange(fn) { listeners.push(fn); },
    builtinIds: BUILTIN.map(s => s.id)
  };
})();
