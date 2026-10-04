'use strict';
/* =========================================================
   SHARED QUESTION BANK
   - Built-in packs plus the teacher's own sets
   - One subject and exam board, chosen on the main screen and read by every game
   - One active set for each subject, used by every game
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

  /* Subjects and exam boards. Every subject can hold the teacher's own sets; built-in packs
     exist for the sciences only so far. A pack with a board is shown for that board when the
     subject has any pack for it; until then every pack for the subject is shown. */
  const SUBJECTS = [
    { id: 'biology', label: 'Biology' }, { id: 'chemistry', label: 'Chemistry' }, { id: 'physics', label: 'Physics' },
    { id: 'combined', label: 'Combined Science' }, { id: 'maths', label: 'Maths' }, { id: 'english', label: 'English' },
    { id: 'history', label: 'History' }, { id: 'geography', label: 'Geography' }
  ];
  const BOARDS = [{ id: 'aqa', label: 'AQA' }, { id: 'edexcel', label: 'Edexcel' }];
  const isSubject = k => SUBJECTS.some(x => x.id === k);
  const isBoard = k => BOARDS.some(x => x.id === k);

  const P = CGB.PACKS;
  const bio = parse(P.biology), chem = parse(P.chemistry), phys = parse(P.physics);
  const BUILTIN = [
    { id: 'gcse-combined-mix', name: 'GCSE Combined Science starter pack: all three sciences (AQA-style)', short: 'Combined Science: mixed', subjects: ['combined'], questions: bio.concat(chem, phys) },
    { id: 'gcse-combined-biology', name: 'GCSE Combined Science starter pack: Biology (AQA-style)', short: 'Combined Science: Biology', subjects: ['combined', 'biology'], questions: bio },
    { id: 'gcse-combined-chemistry', name: 'GCSE Combined Science starter pack: Chemistry (AQA-style)', short: 'Combined Science: Chemistry', subjects: ['combined', 'chemistry'], questions: chem },
    { id: 'gcse-combined-physics', name: 'GCSE Combined Science starter pack: Physics (AQA-style)', short: 'Combined Science: Physics', subjects: ['combined', 'physics'], questions: phys },
    { id: 'homeostasis-l1', name: 'Homeostasis L1 (Biology)', short: 'Homeostasis L1', subjects: ['biology', 'combined'], questions: parse(P.homeostasis) }
  ].map(s => Object.assign(s, { builtin: true, board: 'aqa' }));

  let custom = [];
  let subject = 'combined', board = 'aqa';
  let activeIds = {};    // subject -> id of the set in use for it
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
    const c = { id: String(s.id || newId()), name: String(s.name || 'My questions').slice(0, 60), questions: qs, created: s.created || new Date().toISOString().slice(0, 10) };
    if (isSubject(s.subject)) c.subject = s.subject;   // sets from before subjects existed have none and show under every subject
    return c;
  }
  function newId() { return 'set-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7); }

  function load() {
    const c = store.getJSON('sets', []);
    custom = Array.isArray(c) ? c.map(cleanSet).filter(Boolean) : [];
    const h = store.getJSON('history', {});
    history = h && typeof h === 'object' && !Array.isArray(h) ? h : {};
    const sj = store.get('subject'); if (isSubject(sj)) subject = sj;
    const bd = store.get('board'); if (isBoard(bd)) board = bd;
    const ids = store.getJSON('activeSets', {});
    activeIds = ids && typeof ids === 'object' && !Array.isArray(ids) ? ids : {};
    const a = store.get('activeSet');    // the single set in use before subjects existed
    if (a && !activeIds[subject] && available().some(s => s.id === a)) activeIds[subject] = a;
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
  const setSubjects = s => s.builtin ? s.subjects : s.subject ? [s.subject] : null;   // null: any subject
  /* The sets for the chosen subject (and board, once that subject has a pack for it) */
  function available(sj) {
    sj = sj || subject;
    const inSubject = all().filter(s => { const k = setSubjects(s); return !k || k.includes(sj); });
    const forBoard = inSubject.filter(s => s.board === board);
    return forBoard.length ? inSubject.filter(s => !s.board || s.board === board) : inSubject;
  }
  /* The set in use for the chosen subject, or null when the subject has no questions yet */
  function active() {
    const list = available();
    return list.find(s => s.id === activeIds[subject]) || list[0] || null;
  }
  function setActive(id) {
    if (!available().some(s => s.id === id)) return false;
    activeIds[subject] = id; store.setJSON('activeSets', activeIds); emit(); return true;
  }
  function setSubject(k) { if (!isSubject(k) || k === subject) return false; subject = k; store.set('subject', k); emit(); return true; }
  function setBoard(k) { if (!isBoard(k) || k === board) return false; board = k; store.set('board', k); emit(); return true; }
  const label = (list, k) => (list.find(x => x.id === k) || {}).label || '';
  /* What the main screen and the setup cards say about the chosen subject */
  function subjectNote() {
    const sj = label(SUBJECTS, subject), bd = label(BOARDS, board);
    const builtin = available().filter(s => s.builtin);
    if (!builtin.length) return `Built-in ${sj} packs are coming soon. For now, add your own ${sj} questions in the Question bank.`;
    if (!builtin.some(s => s.board === board)) return `The built-in ${sj} packs are AQA-style. ${bd} packs are coming soon.`;
    return '';
  }
  /* Custom sets can be moved to another subject from the Question bank */
  function setSubjectOf(id, k) {
    const s = custom.find(x => x.id === id); if (!s) return false;
    if (isSubject(k)) s.subject = k; else delete s.subject;
    saveSets(); emit(); return true;
  }
  function summary(set) {
    const subjects = {}, topics = {};
    set.questions.forEach(q => { subjects[q.subject] = 1; topics[q.topic] = 1; });
    return { subjects: Object.keys(subjects), topics: Object.keys(topics), count: set.questions.length };
  }
  function addSet(name, text, sj) {
    const qs = parse(text);
    if (!qs.length) return { ok: false, error: 'No Q: and A: pairs found. Check the format.' };
    const set = cleanSet({ id: newId(), name: (name || '').trim() || 'My questions', questions: qs, subject: sj || subject });
    custom.push(set);
    if (!saveSets()) { /* still usable for this session */ }
    if (set.subject && set.subject !== subject) { subject = set.subject; store.set('subject', subject); }
    activeIds[subject] = set.id; store.setJSON('activeSets', activeIds);
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
    Object.keys(activeIds).forEach(k => { if (activeIds[k] === id) delete activeIds[k]; });
    store.setJSON('activeSets', activeIds);
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
  /* Take back the most recent wrong answer logged for this question (class-mode undo) */
  function unlogWrong(name, q) {
    const h = q && history[key(name)]; if (!h) return;
    for (let i = h.entries.length - 1; i >= 0; i--) if (h.entries[i].q === q.q) { h.entries.splice(i, 1); break; }
    if (!h.entries.length) delete history[key(name)];
    saveHistory();
  }
  function wrongLog(name) { const h = history[key(name)]; return h ? h.entries.slice() : []; }
  function weakTopics(name, n) {
    const m = {};
    wrongLog(name).forEach(e => { m[e.topic] = (m[e.topic] || 0) + 1; });
    return Object.entries(m).sort((a, b) => b[1] - a[1]).slice(0, n || 5);
  }
  function clearHistory(name) { delete history[key(name)]; saveHistory(); emit(); }
  /* Fill a question-set dropdown with the chosen subject's sets. Options use the short name so
     they are never cut off; the full name is in the tooltip. */
  function fillSelect(sel) {
    const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const a = active();
    sel.innerHTML = a ? available().map(s => `<option value="${esc(s.id)}" title="${esc(s.name)}">${esc(s.short || s.name)} (${s.questions.length})</option>`).join('')
      : '<option value="">No questions yet</option>';
    sel.disabled = !a;
    sel.value = a ? a.id : '';
    sel.title = a ? a.name : '';
  }
  function players() { return Object.values(history).map(h => ({ name: h.name, count: h.entries.length })).sort((a, b) => a.name.localeCompare(b.name)); }

  /* ---------- Question picking (each game session gets its own picker) ---------- */
  const WEAK_SHARE = 0.35;
  function createPicker() {
    const used = new Set();
    let setId = null;
    return {
      reset() { used.clear(); },
      /* names: the playing teams. Now and then a question comes from a topic one of them got
         wrong before, so weak topics come round again without the teacher having to ask. */
      pick(names) {
        const set = active();
        if (!set) return null;
        if (set.id !== setId) { used.clear(); setId = set.id; }
        const bank = set.questions;
        let pool = bank.map((q, i) => i).filter(i => !used.has(i));
        if (!pool.length) { used.clear(); pool = bank.map((q, i) => i); }
        const weak = new Set();
        [].concat(names || []).forEach(n => wrongLog(n).forEach(e => weak.add(e.topic)));
        if (weak.size) {
          const weakPool = pool.filter(i => weak.has(bank[i].topic));
          if (weakPool.length && weakPool.length < pool.length && CGB.random() < WEAK_SHARE) pool = weakPool;
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
      subject, board, activeSets: Object.assign({}, activeIds),
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
    if (data.activeSets && typeof data.activeSets === 'object') Object.keys(data.activeSets).forEach(k => { if (isSubject(k) && get(data.activeSets[k])) activeIds[k] = data.activeSets[k]; });
    else if (data.activeSet && get(data.activeSet) && available().some(s => s.id === data.activeSet)) activeIds[subject] = data.activeSet;
    store.setJSON('activeSets', activeIds);
    saveSets(); saveHistory(); emit();
    return { ok: true, added, updated, entries };
  }

  load();
  migrateLegacy();
  return {
    parse, toText, estimateLevel, all, get, active, setActive, available, summary, addSet, updateSet, deleteSet, setSubjectOf,
    SUBJECTS, BOARDS, subject: () => subject, board: () => board, setSubject, setBoard, subjectNote,
    subjectLabel: k => label(SUBJECTS, k || subject), boardLabel: k => label(BOARDS, k || board),
    logWrong, unlogWrong, wrongLog, weakTopics, clearHistory, players, createPicker, fillSelect,
    exportData, importData, onChange(fn) { listeners.push(fn); },
    builtinIds: BUILTIN.map(s => s.id)
  };
})();
