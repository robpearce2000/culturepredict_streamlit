'use strict';
/* =========================================================
   SHARED QUESTION BANK
   - Built-in packs (one per specification topic) plus the teacher's own sets
   - One subject and exam board, chosen on the main screen and read by every game
   - The packs and sets ticked for that subject and board, used by every game as one set
   - Wrong-answer history per player name, shared by every game
   - JSON backup (export / import) of sets and history
   ========================================================= */
CGB.bank = (() => {
  const store = CGB.store;

  /* The plain-text format teachers paste in: Subject / Topic lines apply until changed; Q: then A:.
     Optional lines after a question's A: line add to that question:
       Accept: other accepted answers, separated by ";"
       Note: a note for the teacher (a misconception, a marking hint)
     Optional lines that apply to the questions after them, until the next Topic line:
       Difficulty: 1, 2 or 3   (or "Tier: 1-3", the name used before version 1.1; see TIERS below)
       Tier: Higher            (Higher tier only content; "Tier: Foundation" ends it)
     Sets written for the first standalone games used "Level: 1-5": 1-2 read as 1, 3 as 2, 4-5 as 3. */
  const fromLevel = n => n <= 2 ? 1 : n === 3 ? 2 : 3;
  function parse(text) {
    const out = [];
    let subject = '', topic = '', diff = 0, higher = false, pendingQ = null, last = null;
    String(text || '').split('\n').forEach(line => {
      const t = line.trim();
      const val = re => t.replace(re, '').trim();
      if (/^subject:/i.test(t)) { subject = val(/^subject:/i); diff = 0; higher = false; last = null; }
      else if (/^topic:/i.test(t)) { topic = val(/^topic:/i); diff = 0; higher = false; last = null; }
      else if (/^tier:/i.test(t)) {
        const v = val(/^tier:/i), n = parseInt(v, 10);
        if (n >= 1 && n <= 3) diff = n;
        else if (/^higher/i.test(v)) higher = true;
        else if (/^(foundation|both|all)/i.test(v)) higher = false;
      }
      else if (/^difficulty:/i.test(t)) { const n = parseInt(val(/^difficulty:/i), 10); diff = n >= 1 && n <= 3 ? n : n >= 4 && n <= 5 ? 3 : 0; }
      else if (/^level:/i.test(t)) { const n = parseInt(val(/^level:/i), 10); diff = n >= 1 && n <= 5 ? fromLevel(n) : 0; }
      else if (/^q:/i.test(t)) { pendingQ = val(/^q:/i); last = null; }
      else if (/^a:/i.test(t) && pendingQ) {
        const q = { subject: subject || 'Custom', topic: topic || 'General', q: pendingQ, a: val(/^a:/i) };
        if (diff) q.difficulty = diff;
        if (higher) q.tier = 'Higher';
        out.push(q); last = q;
        pendingQ = null;
      }
      else if (/^accept:/i.test(t) && last) last.accept = val(/^accept:/i).split(';').map(x => x.trim()).filter(Boolean);
      else if (/^note:/i.test(t) && last) last.notes = val(/^note:/i);
      else if (/^calc:/i.test(t) && last) { if (/^(yes|y|true|1)$/i.test(val(/^calc:/i))) last.calc = true; }   // a calculation: 45 s to answer outside Maths
    });
    return out;
  }
  /* Turn questions back into the plain-text format (used for editing and copying) */
  function toText(questions) {
    const lines = [];
    let s = null, t = null, d = 0, h = false;
    questions.forEach(q => {
      if (q.subject !== s || q.topic !== t) {
        if (lines.length) lines.push('');
        lines.push('Subject: ' + q.subject, 'Topic: ' + q.topic);
        s = q.subject; t = q.topic; d = 0; h = false;
      }
      const qd = difficultyOf(q);
      if (qd !== d) { lines.push('Difficulty: ' + qd); d = qd; }
      const qh = q.tier === 'Higher';
      if (qh !== h) { lines.push(qh ? 'Tier: Higher' : 'Tier: Foundation'); h = qh; }
      lines.push('Q: ' + q.q, 'A: ' + q.a);
      if (q.accept && q.accept.length) lines.push('Accept: ' + q.accept.join('; '));
      if (q.notes) lines.push('Note: ' + q.notes);
      if (q.calc) lines.push('Calc: yes');
    });
    return lines.join('\n');
  }
  /* DIFFICULTY (explained in DECISIONS.md and the Question bank help). A question's difficulty is
     set by the most demanding thing it asks, judged the way exam boards build mark schemes:
     the command word, the assessment objective, and whether it is Higher-tier-only content.
       1  Recall: one fact, name, term or number. Name, State, Which, What is (a one-word answer).
       2  Describe and apply: a definition, a process or test and its result, an equation to
          recall, several facts, or a one-step use of a rule. Describe, Give, What does ... do.
       3  Explain and extend: a reason (why or how), a comparison, a calculation where the
          equation isn't given, a new context, or Higher-tier-only content. Explain, Why,
          Compare, Suggest, Calculate.
     Every built-in question has its difficulty set by hand. Teachers' questions without a
     Difficulty line get one from the same rules, read from the wording. (The overnight brief and
     versions 1.0.x called these "tiers"; "tier" now means Foundation or Higher, as exam boards use it.) */
  const TIERS = [{ tier: 1, label: 'Recall' }, { tier: 2, label: 'Describe and apply' }, { tier: 3, label: 'Explain and extend' }];
  function estimateTier(q) {
    const Q = q.q, words = q.a.split(/\s+/).length;
    if (/^(why|explain|suggest|compare|evaluate|justify|predict)\b|how (does|do|is|are|can|did) .* (affect|change|work|help|cause|keep|make|stop|speed)|what is the difference|differences? between|\bcalculate\b/i.test(Q)) return 3;
    if (/\d/.test(Q) && /\d/.test(q.a) && !/equation|formula/i.test(Q)) return (Q.match(/\d+(\.\d+)?\s*[a-zA-Zµ°%Ω]/g) || []).length >= 2 ? 3 : 2;   // a calculation: from two quantities, or a simpler one
    if (/^(describe|outline|give|name|state|list) (two|three|four|the (equation|formula|word equation)|an? (equation|formula))\b|equation|formula|^(describe|outline|what does .* do)/i.test(Q) || words > 5) return 2;
    return 1;
  }
  const difficultyOf = q => q.difficulty || estimateTier(q);

  /* Subjects and exam boards. Built-in packs are one per specification topic (CGB.PACKDATA, made
     by tools/packs.js from packs/); teachers' own sets can be filed under any subject, or Other. */
  const SUBJECTS = [
    { id: 'biology', label: 'Biology' }, { id: 'chemistry', label: 'Chemistry' }, { id: 'physics', label: 'Physics' },
    { id: 'maths', label: 'Maths' }, { id: 'history', label: 'History' }, { id: 'geography', label: 'Geography' },
    { id: 'other', label: 'Other' }
  ];
  const BOARDS = [{ id: 'aqa', label: 'AQA' }, { id: 'edexcel', label: 'Edexcel' }];
  const isSubject = k => SUBJECTS.some(x => x.id === k);
  const isBoard = k => BOARDS.some(x => x.id === k);
  // sets filed under subjects that no longer exist (Combined Science, English) show under Other
  const subjectOfSet = s => isSubject(s.subject) ? s.subject : s.subject ? 'other' : null;

  /* ---------- Built-in topic packs ---------- */
  const DATA = CGB.PACKDATA || { specs: {}, packs: [] };
  const SPEC_NAMES = { '1GA0': 'Geography A', '1GB0': 'Geography B' };
  const BUILTIN = DATA.packs.map(p => {
    const board = p.board.toLowerCase(), sj = (SUBJECTS.find(x => x.id === p.subject) || {}).label || p.subject;
    const style = `${p.board}-style`, course = SPEC_NAMES[p.specCode] ? SPEC_NAMES[p.specCode] + ': ' : '';
    const pack = {
      id: p.id, builtin: true, board, subject: p.subject, specCode: p.specCode, specVersion: DATA.specs[p.board + ' ' + p.specCode] || '',
      topic: p.topic, ref: p.ref, order: p.order, subtopics: p.subtopics, course: SPEC_NAMES[p.specCode] || '',
      name: `${p.topic} (${style} GCSE ${course ? course.slice(0, -2) : sj} ${p.specCode}, ${p.ref})`, short: course + p.topic, count: p.n
    };
    let qs = null;   // expanded the first time the pack is used
    Object.defineProperty(pack, 'questions', {
      enumerable: true,
      get() {
        if (!qs) qs = (p.q || CGB.unpackJSON(p.z)).map(r => {
          const q = { id: r[0], board: p.board, subject: sj, specCode: p.specCode, specRef: r[1], topic: p.topic, subtopic: p.subtopics[r[2]] || p.topic, difficulty: r[3], tier: r[4] ? 'Higher' : 'Foundation and Higher', q: r[5], a: r[6], accept: r[7] ? r[7].split(';') : [], hexOk: !!r[8], notes: r[9] || '' };
          if (r[10]) Object.assign(q, r[10]);
          return q;
        });
        return qs;
      }
    });
    return pack;
  });

  /* Combined Science: the same topic packs seen through the Combined Science specification (AQA
     Trilogy 8464, Edexcel 1SC0). Questions on separate-science-only content are left out, and the
     rest carry their Combined Science reference and tier (tools/spec-map, applied by tools/packs.js). */
  const SCIENCES = ['biology', 'chemistry', 'physics'];
  const COURSES = [{ id: 'separate', label: 'Separate science' }, { id: 'combined', label: 'Combined Science' }];
  const COMBINED = BUILTIN.filter(p => DATA.packs.find(d => d.id === p.id).comb).map(base => {
    const c = DATA.packs.find(d => d.id === base.id).comb, style = `${base.board === 'aqa' ? 'AQA' : 'Edexcel'}-style`;
    const pack = {
      id: base.id + '~comb', builtin: true, combined: true, board: base.board, subject: base.subject, specCode: c.specCode, specVersion: c.version,
      topic: c.topic, ref: c.ref, order: c.order, course: 'Combined Science', separateId: base.id,
      name: `${c.topic} (${style} GCSE ${c.name} ${c.specCode}, ${c.ref})`, short: c.topic, count: DATA.packs.find(d => d.id === base.id).nc
    };
    let qs = null;
    Object.defineProperty(pack, 'questions', {
      enumerable: true,
      get() {
        if (!qs) qs = base.questions.filter(q => !q.sepOnly).map(q => Object.assign({}, q, { specCode: c.specCode, specRef: q.combRef || q.specRef, separateRef: q.specRef, tier: q.combTier || q.tier }));
        return qs;
      }
    });
    // Category Clash columns: only the subtopics with questions left in Combined Science
    Object.defineProperty(pack, 'subtopics', { enumerable: true, get() { return base.subtopics.filter(t => pack.questions.some(q => q.subtopic === t)); } });
    return pack;
  });

  let custom = [];
  let subject = 'biology', board = 'aqa';
  let selections = {};   // "subject|board" -> { mixed: specCode } or { ids: [pack and set ids] }
  let higher = true;     // include Higher tier only questions
  let course = 'separate';   // Biology, Chemistry and Physics: separate science or Combined Science
  let history = {};      // key (lower-case name) -> { name, entries: [ {subject, topic, q, a, date, game} ] }
  const listeners = [];
  let cached = null;
  const emit = () => { cached = null; listeners.forEach(fn => { try { fn(); } catch (e) { /* ignore */ } }); };

  function validQuestion(q) { return q && typeof q.q === 'string' && typeof q.a === 'string' && q.q.trim() && q.a.trim(); }
  function cleanQuestion(q) {
    const c = { subject: String(q.subject || 'Custom').slice(0, 80), topic: String(q.topic || 'General').slice(0, 120), q: String(q.q).slice(0, 600), a: String(q.a).slice(0, 600) };
    // difficulty: saved as "tier" (1-3) before version 1.1, and "level" (1-5) by the first games
    const d = parseInt(q.difficulty, 10), t = parseInt(q.tier, 10), l = parseInt(q.level, 10);
    if (d >= 1 && d <= 3) c.difficulty = d; else if (t >= 1 && t <= 3) c.difficulty = t; else if (l >= 1 && l <= 5) c.difficulty = fromLevel(l);
    if (q.tier === 'Higher') c.tier = 'Higher';
    if (Array.isArray(q.accept)) { const a = q.accept.map(x => String(x).trim().slice(0, 120)).filter(Boolean).slice(0, 12); if (a.length) c.accept = a; }
    if (q.notes) c.notes = String(q.notes).slice(0, 300);
    if (q.calc) c.calc = true;
    return c;
  }
  function cleanSet(s) {
    if (!s || !Array.isArray(s.questions)) return null;
    const qs = s.questions.filter(validQuestion).map(cleanQuestion);
    if (!qs.length) return null;
    const c = { id: String(s.id || newId()), name: String(s.name || 'My questions').slice(0, 60), questions: qs, created: s.created || new Date().toISOString().slice(0, 10) };
    if (s.subject) c.subject = String(s.subject).slice(0, 30);   // kept as saved; one no longer listed shows under Other
    return c;
  }
  function newId() { return 'set-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7); }

  function load() {
    const c = store.getJSON('sets', []);
    custom = Array.isArray(c) ? c.map(cleanSet).filter(Boolean) : [];
    const h = store.getJSON('history', {});
    history = h && typeof h === 'object' && !Array.isArray(h) ? h : {};
    const sj = store.get('subject'); if (isSubject(sj)) subject = sj; else if (sj === 'combined') subject = 'biology'; else if (sj) subject = 'other';
    const bd = store.get('board'); if (isBoard(bd)) board = bd;
    const sel = store.getJSON('packSelections', {});
    selections = sel && typeof sel === 'object' && !Array.isArray(sel) ? sel : {};
    higher = store.get('higherTier') !== '0';
    if (COURSES.some(c => c.id === store.get('course'))) course = store.get('course');
    // a teacher's own set chosen before version 1.1 stays chosen
    const ids = store.getJSON('activeSets', {});
    if (ids && typeof ids === 'object') Object.keys(ids).forEach(k => {
      const sjk = isSubject(k) ? k : 'other', set = custom.find(x => x.id === ids[k]);
      BOARDS.forEach(b => { const key = sjk + '|' + b.id; if (set && !selections[key]) selections[key] = { ids: [set.id] }; });
    });
    const one = store.get('activeSet');   // the single set in use before subjects existed (version 1.0 beta)
    if (one && custom.some(x => x.id === one) && !selections[subject + '|' + board]) selections[subject + '|' + board] = { ids: [one] };
  }
  function saveSets() { return store.setJSON('sets', custom); }
  function saveHistory() { return store.setJSON('history', history); }
  function saveSelections() { store.setJSON('packSelections', selections); }

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

  function all() { return BUILTIN.concat(COMBINED, custom); }
  const hasCourses = sj => SCIENCES.includes(sj || subject);
  const combinedNow = sj => hasCourses(sj) && course === 'combined';
  /* History and Geography GCSEs are not tiered: no Higher tier switch, nothing left out */
  const tiered = sj => !['history', 'geography'].includes(sj || subject);
  function get(id) { return all().find(s => s.id === id) || null; }
  /* The built-in topic packs for a subject and board, in specification order */
  function packs(sj, bd) {
    sj = sj || subject; bd = bd || board;
    return (combinedNow(sj) ? COMBINED : BUILTIN).filter(p => p.subject === sj && p.board === bd).sort((a, b) => a.specCode.localeCompare(b.specCode) || a.order - b.order);
  }
  /* The specifications those packs come from (two for Edexcel Geography) */
  function courses(sj, bd) {
    const m = new Map();
    packs(sj, bd).forEach(p => { if (!m.has(p.specCode)) m.set(p.specCode, { specCode: p.specCode, name: p.course, version: p.specVersion, count: 0 }); m.get(p.specCode).count += p.count; });
    return Array.from(m.values());
  }
  /* The teacher's own sets for a subject (sets saved before subjects existed show under every subject) */
  function ownSets(sj) { sj = sj || subject; return custom.filter(s => { const k = subjectOfSet(s); return !k || k === sj; }); }
  /* Everything that can be chosen for the subject and board: topic packs, then the teacher's sets */
  function available(sj) { return packs(sj).concat(ownSets(sj)); }

  /* ---------- What is chosen ---------- */
  const selKey = () => subject + '|' + board + (combinedNow() ? '|combined' : '');
  function selection() {
    const s = selections[selKey()], cs = courses();
    if (s && s.mixed && cs.some(c => c.specCode === s.mixed)) return { mixed: s.mixed };
    if (s && Array.isArray(s.ids)) { const ids = s.ids.filter(id => available().some(x => x.id === id)); if (ids.length) return { ids }; }
    // nothing chosen yet (or what was chosen has gone): every topic, or the teacher's first set
    if (cs.length) return { mixed: cs[0].specCode };
    const own = ownSets();
    return own.length ? { ids: [own[0].id] } : { ids: [] };
  }
  function setSelection(sel) {
    if (sel && sel.mixed) selections[selKey()] = { mixed: String(sel.mixed) };
    else selections[selKey()] = { ids: (sel && sel.ids || []).filter(id => available().some(x => x.id === id)) };
    saveSelections(); emit(); return true;
  }
  /* Tick or untick one pack or set (the dropdown on the main screen) */
  function toggle(id) {
    const cur = selection();
    let ids = cur.mixed ? [] : cur.ids.slice();
    ids = ids.includes(id) ? ids.filter(x => x !== id) : ids.concat(id);
    return setSelection(ids.length ? { ids } : { mixed: (courses()[0] || {}).specCode });
  }
  function setCourse(k) { if (!COURSES.some(c => c.id === k) || k === course) return false; course = k; store.set('course', k); emit(); return true; }
  function setHigher(on) { higher = !!on; store.set('higherTier', higher ? '1' : '0'); emit(); }

  /* The question set every game uses: what is chosen, as one set. With one topic chosen,
     Category Clash takes its columns from the specification's subtopics (groupBy). */
  function active() {
    if (cached) return cached;
    const sel = selection();
    const chosen = sel.mixed ? packs().filter(p => p.specCode === sel.mixed) : sel.ids.map(get).filter(Boolean);
    if (!chosen.length) return null;
    let questions = [].concat(...chosen.map(s => s.questions));
    const leaveOut = !higher && tiered();
    const dropped = leaveOut ? questions.filter(q => q.tier === 'Higher').length : 0;
    if (leaveOut) questions = questions.filter(q => q.tier !== 'Higher');
    if (!questions.length) return null;
    const one = chosen.length === 1 ? chosen[0] : null;
    const course = sel.mixed ? (courses().find(c => c.specCode === sel.mixed) || {}).name : '';
    const short = sel.mixed ? `Mixed: all ${course ? course + ' ' : ''}topics` : one ? one.short || one.name : `${chosen.length} ${chosen.every(s => s.builtin) ? 'topics' : 'sets'}: ${chosen.map(s => s.builtin ? s.topic : s.name).join(', ')}`;
    cached = {
      id: 'sel:' + selKey() + ':' + (sel.mixed ? 'mixed-' + sel.mixed : sel.ids.join('+')) + (leaveOut ? ':f' : ''),
      name: sel.mixed ? `${short} (${label(BOARDS, board)}-style GCSE ${label(SUBJECTS, subject)})` : one ? one.name : short,
      short, questions, parts: chosen, builtin: chosen.every(s => s.builtin), withoutHigher: dropped,
      groupBy: one && one.builtin && one.subtopics.length > 1 ? 'subtopic' : 'topic'
    };
    return cached;
  }
  function setSubject(k) { if (!isSubject(k) || k === subject) return false; subject = k; store.set('subject', k); emit(); return true; }
  function setBoard(k) { if (!isBoard(k) || k === board) return false; board = k; store.set('board', k); emit(); return true; }
  const label = (list, k) => (list.find(x => x.id === k) || {}).label || '';
  /* What the main screen and the setup cards say about the chosen subject */
  function subjectNote() {
    const sj = label(SUBJECTS, subject), bd = label(BOARDS, board);
    if (subject === 'other') return ownSets().length ? '' : 'Add your own questions for any subject in the Question bank.';
    if (!packs().length) return `Built-in ${bd}-style ${sj} packs are coming soon. For now, add your own ${sj} questions in the Question bank.`;
    if (!higher && tiered()) return `Foundation tier: questions on Higher tier only content${combinedNow() ? ' in Combined Science' : ''} are left out.`;
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
    if (set.subject && isSubject(set.subject) && set.subject !== subject) { subject = set.subject; store.set('subject', subject); }
    selections[selKey()] = { ids: [set.id] }; saveSelections();
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
    Object.keys(selections).forEach(k => { const s = selections[k]; if (s && Array.isArray(s.ids)) s.ids = s.ids.filter(x => x !== id); });
    saveSelections(); saveSets(); emit();
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
      subject, board, packSelections: JSON.parse(JSON.stringify(selections)), higherTier: higher, course,
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
    if (data.packSelections && typeof data.packSelections === 'object') Object.keys(data.packSelections).forEach(k => { if (!selections[k]) selections[k] = data.packSelections[k]; });
    saveSelections();
    saveSets(); saveHistory(); emit();
    return { ok: true, added, updated, entries };
  }

  load();
  migrateLegacy();
  return {
    parse, toText, TIERS, difficultyOf, tierOf: difficultyOf, estimateTier, all, get, active, available, packs, courses, ownSets, summary, addSet, updateSet, deleteSet, setSubjectOf,
    selection, setSelection, toggle, higher: () => higher, setHigher, tiered,
    COURSES, course: () => course, setCourse, hasCourses, combined: combinedNow, specs: () => Object.assign({}, DATA.specs),
    SUBJECTS, BOARDS, subject: () => subject, board: () => board, setSubject, setBoard, subjectNote,
    subjectLabel: k => label(SUBJECTS, k || subject), boardLabel: k => label(BOARDS, k || board),
    logWrong, unlogWrong, wrongLog, weakTopics, clearHistory, players, createPicker,
    exportData, importData, onChange(fn) { listeners.push(fn); },
    builtinIds: BUILTIN.map(s => s.id), subjectOfSet
  };
})();
