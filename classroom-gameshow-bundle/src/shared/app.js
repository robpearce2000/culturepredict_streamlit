'use strict';
/* =========================================================
   APP SHELL: launcher, routing between games, modals
   (settings, question bank, about, leave game)
   ========================================================= */
CGB.games = {};
CGB.registerGame = function (id, def) { CGB.games[id] = Object.assign({ id, ready: false }, def); };

/* Create a WebGL renderer, or return null if this computer can't do 3D */
CGB.createRenderer = function (opts) {
  try {
    const r = new THREE.WebGLRenderer(Object.assign({ antialias: true, powerPreference: 'high-performance' }, opts || {}));
    if (!r.getContext()) return null;
    return r;
  } catch (e) { return null; }
};
CGB.noWebGLMessage = '<div class="cgb-panel" style="max-width:560px;margin:15vh auto 0;padding:24px 26px;"><h2 style="font-family:var(--display);font-weight:400;margin:0 0 8px;">3D graphics are switched off</h2><p class="hint" style="font-size:1rem">This browser could not start 3D graphics (WebGL). Try another browser (Chrome, Edge, Firefox or Safari), or ask IT to turn on hardware acceleration.</p></div>';

/* Starting a game: take the keyboard off the setup card (a name box or the Start button), so
   the game's keys work at once (Space for "show me", Enter to pick) instead of going to
   controls that are about to disappear */
CGB.leaveField = () => { const a = document.activeElement; if (a && a !== document.body && a.closest && a.closest('.setup') && a.blur) a.blur(); };

/* Two-tap confirm for destructive buttons */
CGB.armButton = function (btn, armedText, onConfirm) {
  if (btn.dataset.armed === '1') { btn.dataset.armed = ''; onConfirm(); return; }
  const original = btn.innerHTML;
  btn.dataset.armed = '1'; btn.textContent = armedText;
  setTimeout(() => { if (btn.isConnected && btn.dataset.armed === '1') { btn.dataset.armed = ''; btn.innerHTML = original; } }, 3500);
};

/* ---------- Modals ---------- */
CGB.modal = (() => {
  const stack = [];
  function open(id) {
    const m = document.getElementById(id);
    if (!m || !m.hidden) return;
    stack.push({ m, back: document.activeElement });
    m.hidden = false;
    const f = m.querySelector('[data-autofocus]') || m.querySelector('button, input, select, textarea');
    if (f) setTimeout(() => f.focus(), 0);
  }
  function close(id) {
    const i = stack.findIndex(s => s.m.id === id);
    if (i < 0) return;
    const s = stack.splice(i, 1)[0];
    s.m.hidden = true;
    if (s.back && s.back.focus && s.back.isConnected) s.back.focus();
  }
  function top() { return stack.length ? stack[stack.length - 1].m : null; }
  document.addEventListener('keydown', e => {
    const t = top(); if (!t) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); close(t.id); return; }
    if (e.key === 'Tab') {      // keep focus inside the open modal
      const f = Array.from(t.querySelectorAll('button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])')).filter(el => !el.disabled && el.offsetParent !== null);
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  }, true);
  document.addEventListener('click', e => {
    const c = e.target.closest('[data-close]'); if (c) { close(c.dataset.close); return; }
    if (e.target.classList && e.target.classList.contains('modal')) close(e.target.id);
  });
  return { open, close, isOpen: () => stack.length > 0, top };
})();

/* ---------- Settings controls (rendered on the launcher and in the settings modal) ---------- */
CGB.renderSettings = function (el, compact) {
  const S = CGB.settings;
  const rows = [
    { key: 'sound', label: 'Sound', opts: [[true, 'On'], [false, 'Off']], hint: 'All sounds are made by the browser. There is no music.' },
    { key: 'textSize', label: 'Text size', opts: [['normal', 'Normal'], ['large', 'Large']], hint: 'Large makes every panel and question easier to read from the back.' },
    { key: 'reducedMotion', label: 'Reduced motion', opts: [[false, 'Off'], [true, 'On']], hint: 'Stops camera sway, screen shake and flashes.' },
    { key: 'quality', label: 'Graphics', opts: [['high', 'High'], ['low', 'Low']], hint: 'Low turns off shadows and glow for older classroom PCs.' }
  ];
  el.innerHTML = rows.map(r => `<div class="${compact ? 'setting-compact' : 'field'}">
      <span class="lab" id="lab-${compact ? 'c' : 'm'}-${r.key}">${r.label}</span>
      <div class="seg" role="group" aria-labelledby="lab-${compact ? 'c' : 'm'}-${r.key}" data-key="${r.key}">${r.opts.map(([v, t]) => `<button type="button" data-v='${JSON.stringify(v)}' aria-pressed="${S.get(r.key) === v}">${t}</button>`).join('')}</div>
      ${compact ? '' : `<div class="hint">${r.hint}</div>`}
    </div>`).join('');
  el.onclick = e => {
    const b = e.target.closest('.seg button'); if (!b) return;
    const key = b.parentElement.dataset.key;
    S.set(key, JSON.parse(b.dataset.v));
  };
};
/* Setup cards must fit on screen without scrolling. Each one shows "How to play"
   open when there is room; if the card would overflow it first tightens its
   spacing, then closes "How to play" (one click opens it again). */
CGB.fitSetups = () => requestAnimationFrame(() => {
  document.querySelectorAll('.setup').forEach(card => {
    if (!card.offsetParent) return;
    const how = card.querySelector('details.howto');
    const over = () => card.scrollHeight > card.clientHeight + 1;
    card.classList.remove('tight', 'tighter', 'twocol');
    if (how) how.open = true;
    if (over()) card.classList.add('tight');
    if (over() && how) how.open = false;
    if (over()) card.classList.add('tighter');
    // a portrait tablet with six class teams: two narrower columns instead of one long one
    if (over() && window.innerWidth >= 640) card.classList.add('twocol');
  });
});
window.addEventListener('resize', () => { clearTimeout(CGB.fitSetups.t); CGB.fitSetups.t = setTimeout(CGB.fitSetups, 120); });
// opening "Edit team names" makes the card taller, so fit it again
document.addEventListener('toggle', e => { if (e.target.matches && e.target.matches('.setup details.tnames')) CGB.fitSetups(); }, true);
CGB.settings.onChange(() => {
  CGB.fitSetups();
  document.querySelectorAll('.seg[data-key]').forEach(seg => {
    const v = CGB.settings.get(seg.dataset.key);
    seg.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(JSON.parse(b.dataset.v) === v)));
  });
});

/* ---------- Question bank manager ---------- */
CGB.bankUI = (() => {
  const B = CGB.bank, esc = CGB.escapeHtml;
  let editing = null;   // id of the set being edited, or null when adding
  const $ = id => document.getElementById(id);
  function subjectTag(s) { const k = s.toLowerCase(); return k.startsWith('bio') ? 'b' : k.startsWith('chem') ? 'c' : k.startsWith('phys') ? 'p' : ''; }
  const subjectOptions = (sel) => '<option value="">Any subject</option>' + B.SUBJECTS.map(x => `<option value="${x.id}"${x.id === sel ? ' selected' : ''}>${esc(x.label)}</option>`).join('');
  function renderList() {
    const a = B.active(), sel = B.selection();
    const inUse = id => !!a && (sel.mixed ? (B.get(id) || {}).specCode === sel.mixed : sel.ids.includes(id));
    $('bankIntro').textContent = `${B.subjectLabel()} questions (${B.boardLabel()}). Tick the packs and sets to use: every game uses them together, and each team's wrong answers are shared between games. Change the subject and exam board on the main screen.`;
    const list = B.available();
    const courses = B.courses();
    const specLine = courses.map(c => `${B.boardLabel()}-style GCSE ${c.name || B.subjectLabel()} ${c.specCode}, specification version ${c.version}`).join('; ');
    $('bankList').innerHTML = (specLine ? `<p class="hint bank-spec">Built-in packs follow ${esc(specLine)}. They are practice questions written for Showtime, not produced or endorsed by the exam board.</p>` : '')
      + (list.length ? '' : `<p class="hint">${esc(B.subjectNote())}</p>`) + list.map(s => {
      const on = inUse(s.id);
      const n = s.builtin ? s.count : s.questions.length;
      const tags = s.builtin
        ? `<span class="tag">${n} questions</span><span class="tag">${s.subtopics.length} subtopics</span><span class="tag">Spec ${esc(s.specCode)} ${esc(s.ref)}</span><span class="tag">Built-in</span>`
        : `<span class="tag">${n} questions</span><span class="tag">${B.summary(s).topics.length} topics</span><span class="tag own">Your set</span><label class="tag sj">Subject <select data-subj="${esc(s.id)}" aria-label="Subject for ${esc(s.name)}">${subjectOptions(B.subjectOfSet(s) || '')}</select></label>`;
      const acts = s.builtin
        ? `<button class="btn plain sm" type="button" data-act="copy" data-id="${esc(s.id)}">Copy to edit</button>`
        : `<button class="btn plain sm" type="button" data-act="rename" data-id="${esc(s.id)}">Rename</button>
           <button class="btn plain sm" type="button" data-act="edit" data-id="${esc(s.id)}">Edit questions</button>
           <button class="btn danger sm" type="button" data-act="delete" data-id="${esc(s.id)}">Delete</button>`;
      return `<div class="bank-item${on ? ' active' : ''}" data-id="${esc(s.id)}">
        <input type="checkbox" name="useSet" value="${esc(s.id)}" ${on ? 'checked' : ''} aria-label="Use ${esc(s.name)}">
        <div><div class="nm">${esc(s.builtin ? s.ref + ' ' + s.topic : s.name)}${on ? ' <span class="tag own">In use</span>' : ''}</div><div class="tags">${tags}</div></div>
        <div class="acts">${acts}</div>
      </div>`;
    }).join('');
    const ps = B.players();
    $('bankPlayers').innerHTML = ps.length
      ? ps.map(p => `<span class="tag" style="font-size:0.85rem;padding:5px 10px;">${esc(p.name)}: ${p.count} wrong <button class="linkish" type="button" data-clearp="${esc(p.name)}" style="font-size:0.8rem;padding:0 0 0 6px;">Clear</button></span>`).join('')
      : '<span class="hint">No wrong answers recorded yet.</span>';
  }
  function setStatus(id, text, err) { const el = $(id); el.textContent = text; el.classList.toggle('err', !!err); }
  function startEdit(set) {
    editing = set ? set.id : null;
    $('bankFormTitle').textContent = set ? 'Edit questions' : 'Add a question set';
    $('bankName').value = set ? set.name : '';
    $('bankText').value = set ? B.toText(set.questions) : '';
    $('bankSubject').innerHTML = B.SUBJECTS.map(x => `<option value="${x.id}">${esc(x.label)}</option>`).join('');
    $('bankSubject').value = set && set.subject ? set.subject : B.subject();
    $('bankSubject').disabled = !!set;
    $('bankSave').textContent = set ? 'Save changes' : 'Save as new set';
    $('bankCancel').hidden = !set;
    setStatus('bankFormStatus', '');
    if (set) { $('bankForm').scrollIntoView({ block: 'start', behavior: CGB.settings.reduced() ? 'auto' : 'smooth' }); $('bankName').focus(); }
  }
  function init() {
    $('bankList').addEventListener('change', e => {
      if (e.target.name === 'useSet') B.toggle(e.target.value);
      else if (e.target.dataset.subj) B.setSubjectOf(e.target.dataset.subj, e.target.value);
    });
    $('bankList').addEventListener('click', e => {
      const b = e.target.closest('button[data-act]'); if (!b) return;
      const set = B.get(b.dataset.id); if (!set) return;
      if (b.dataset.act === 'delete') CGB.armButton(b, 'Tap again to delete', () => { B.deleteSet(set.id); if (editing === set.id) startEdit(null); });
      else if (b.dataset.act === 'edit') startEdit(set);
      else if (b.dataset.act === 'copy') {
        const r = B.addSet((set.short || set.name) + ' (my copy)', B.toText(set.questions), B.subject());
        if (r.ok) startEdit(r.set);
      } else if (b.dataset.act === 'rename') {
        const nm = b.closest('.bank-item').querySelector('.nm');
        nm.innerHTML = `<input class="cgb-input" type="text" maxlength="60" value="${esc(set.name)}" aria-label="New name" style="max-width:340px"> <button class="btn ok sm" type="button" data-act="renameSave" data-id="${esc(set.id)}">Save name</button>`;
        nm.querySelector('input').focus(); nm.querySelector('input').select();
        nm.querySelector('input').addEventListener('keydown', ev => { if (ev.key === 'Enter') { ev.preventDefault(); B.updateSet(set.id, ev.target.value, null); } });
      } else if (b.dataset.act === 'renameSave') {
        B.updateSet(set.id, b.parentElement.querySelector('input').value, null);
      }
    });
    $('bankPlayers').addEventListener('click', e => {
      const b = e.target.closest('[data-clearp]'); if (!b) return;
      CGB.armButton(b, 'Tap again', () => B.clearHistory(b.dataset.clearp));
    });
    $('bankSave').addEventListener('click', () => {
      const name = $('bankName').value, text = $('bankText').value;
      const r = editing ? B.updateSet(editing, name, text) : B.addSet(name, text, $('bankSubject').value);
      if (!r.ok) { setStatus('bankFormStatus', r.error, true); return; }
      setStatus('bankFormStatus', `${editing ? 'Saved' : 'Added'} "${r.set.name}" with ${r.set.questions.length} questions. It is now in use in every game for ${B.subjectLabel(r.set.subject || B.subject())}.${CGB.store.available ? '' : ' Storage is blocked, so it will be lost when the page closes.'}`);
      if (!editing) { $('bankName').value = ''; $('bankText').value = ''; }
      else if (!B.selection().ids.includes(editing)) B.setSelection({ ids: [editing] });
    });
    $('bankCancel').addEventListener('click', () => startEdit(null));
    $('bankExport').addEventListener('click', () => {
      const data = JSON.stringify(B.exportData(), null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'showtime-backup-' + new Date().toISOString().slice(0, 10) + '.json';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      setStatus('bankBackupStatus', 'Backup file saved to your downloads.');
    });
    $('bankImport').addEventListener('click', () => $('bankImportFile').click());
    $('bankImportFile').addEventListener('change', e => {
      const f = e.target.files && e.target.files[0]; if (!f) return;
      const rd = new FileReader();
      rd.onload = () => {
        let data = null;
        try { data = JSON.parse(rd.result); } catch (err) { setStatus('bankBackupStatus', 'That file could not be read. Choose a .json backup made by Showtime.', true); return; }
        const r = B.importData(data);
        setStatus('bankBackupStatus', r.ok ? `Imported: ${r.added} new set${r.added === 1 ? '' : 's'}, ${r.updated} updated, ${r.entries} history entr${r.entries === 1 ? 'y' : 'ies'} added.` : r.error, !r.ok);
      };
      rd.readAsText(f);
      e.target.value = '';
    });
    B.onChange(renderList);
    renderList();
  }
  return { init, open() { renderList(); startEdit(null); CGB.modal.open('bankModal'); }, renderList };
})();

/* ---------- Host editor (on the main screen; changes the host in every game) ---------- */
CGB.hostEditor = (() => {
  const $ = id => document.getElementById(id);
  let preview = null;
  const save = () => { CGB.saveHost(); if (preview) { preview.gesture('present', 1200); preview.talk(700); } };
  function swatchRow(id, list, key, label) {
    const el = $(id);
    el.innerHTML = list.map((c, i) => `<button type="button" class="sw" data-i="${i}" style="background:${c}" aria-label="${label} option ${i + 1}"></button>`).join('');
    const paint = () => el.querySelectorAll('.sw').forEach(b => { const on = +b.dataset.i === CGB.hostCfg[key]; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
    el.addEventListener('click', e => { const b = e.target.closest('.sw'); if (!b) return; CGB.hostCfg[key] = +b.dataset.i; paint(); save(); });
    paint();
  }
  function segRow(id, list, key) {
    const el = $(id);
    el.innerHTML = list.map(([v, label]) => `<button type="button" data-v="${v}">${label}</button>`).join('');
    const paint = () => el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === CGB.hostCfg[key])));
    el.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; CGB.hostCfg[key] = b.dataset.v; paint(); save(); });
    paint();
  }
  function init() {
    swatchRow('swSkin', CGB.hostOpts.skin, 'skin', 'Skin');
    swatchRow('swHair', CGB.hostOpts.hair, 'hair', 'Hair colour');
    swatchRow('swSuit', CGB.hostOpts.suit, 'suit', 'Suit colour');
    segRow('segStyle', CGB.hostOpts.style, 'style');
    segRow('segBeard', CGB.hostOpts.beard, 'beard');
    segRow('segGlasses', CGB.hostOpts.glasses, 'glasses');
    $('hostName').value = CGB.hostName();
    // empty means nameless: no name label anywhere
    $('hostName').addEventListener('input', e => { CGB.hostCfg.name = e.target.value.trim(); CGB.store.setJSON('host', CGB.hostCfg); CGB.hostListeners.forEach(fn => { try { fn(); } catch (x) { /* ignore */ } }); });
    $('hostName').addEventListener('change', () => CGB.saveHost());
  }
  function open() {
    if (!preview) preview = CGB.createHost2D($('hostPreview'), { className: 'host-preview-fig' });
    CGB.modal.open('hostModal');
    preview.gesture('wave', 1800); preview.talk(800);
  }
  return { init, open };
})();

/* ---------- The questions line on every setup card ----------
   Shows the subject and set in use, with a link back to the main screen to change them.
   Returns false when the subject has no questions yet, so the game can hold its Start button. */
CGB.renderPackLine = function (el) {
  const B = CGB.bank, esc = CGB.escapeHtml, a = B.active();
  const where = `${esc(B.subjectLabel())} · ${esc(B.boardLabel())}`;
  el.classList.add('packline');
  el.innerHTML = a
    ? `<span class="lab">Questions</span><span class="pk"><b title="${esc(a.name)}">${esc(a.short || a.name)}</b> <span class="n">${where} · ${a.questions.length} questions</span></span><button class="linkish" type="button" data-pack="change">Change</button>`
    : `<span class="lab">Questions</span><span class="pk"><b>No ${esc(B.subjectLabel())} questions yet</b> <span class="n">Add your own in the Question bank, or choose another subject.</span></span><button class="linkish" type="button" data-pack="change">Change</button>`;
  el.classList.toggle('empty', !a);
  if (!el.dataset.wired) { el.dataset.wired = '1'; el.addEventListener('click', e => { if (e.target.closest('[data-pack]')) CGB.app.chooseSubject(); }); }
  return !!a;
};

/* ---------- Launcher ---------- */
CGB.app = (() => {
  const $ = id => document.getElementById(id);
  let current = 'launcher';
  const routes = { 'over-the-edge': 'over-the-edge', 'outpace': 'outpace', 'category-clash': 'category-clash', 'hex-hunt': 'hex-hunt' };
  let mascot = null;

  function show(id) {
    if (id === current) return;
    const prev = current;
    if (prev !== 'launcher' && CGB.games[prev]) { CGB.games[prev].exit(); $('game-' + prev).hidden = true; }
    current = id;
    if (id === 'launcher') {
      $('launcher').hidden = false;
      document.title = 'Showtime: Classroom Gameshows';
      if (mascot) mascot.resume();
      renderSubject();
      const card = prev !== 'launcher' && document.querySelector(`[data-play="${prev}"]`);
      if (card) card.focus();
    } else {
      const g = CGB.games[id];
      $('launcher').hidden = true;
      if (mascot) mascot.pause();
      $('game-' + id).hidden = false;
      if (!g.ready) { g.init($('game-' + id)); g.ready = true; }
      document.title = g.title + ' | Showtime: Classroom Gameshows';
      g.enter();
      CGB.fitSetups();
    }
    const want = id === 'launcher' ? '' : '#' + id;
    if (location.hash !== want) {
      try { history.replaceState(null, '', want || location.pathname + location.search); }
      catch (e) { location.hash = want; }   // some browsers block this on file:// pages
    }
  }
  /* Games call this from their Menu button and Esc key */
  function requestLauncher() {
    const g = CGB.games[current];
    if (g && g.inProgress && g.inProgress()) { CGB.modal.open('leaveModal'); return; }
    show('launcher');
  }
  /* The subject panel: subject, exam board and the question set in use */
  function renderSubject() {
    const B = CGB.bank, esc = CGB.escapeHtml, sj = B.subject(), bd = B.board();
    // compact dropdowns: native selects work with mouse, touch and keyboard everywhere
    $('subjectSelect').innerHTML = B.SUBJECTS.map(x => `<option value="${x.id}"${x.id === sj ? ' selected' : ''}>${esc(x.label)}</option>`).join('');
    $('boardSelect').innerHTML = B.BOARDS.map(x => `<option value="${x.id}"${x.id === bd ? ' selected' : ''}>${esc(x.label)}</option>`).join('');
    renderPacks();
    const note = B.subjectNote();
    $('subjectNote').textContent = note;
    $('subjectNote').hidden = !note;
  }
  /* The question pack dropdown: "Mixed: all topics", the topic packs in specification order,
     then the teacher's own sets, ticked one or more at a time; and the Higher tier switch */
  function renderPacks() {
    const B = CGB.bank, esc = CGB.escapeHtml, a = B.active(), sel = B.selection();
    const packs = B.packs(), own = B.ownSets(), courses = B.courses();
    $('packBtn').textContent = a ? `${a.short} (${a.questions.length})` : 'No questions yet';
    $('packBtn').title = a ? a.name : '';
    const opt = (attrs, on, body, n) => `<label class="pk-opt"><input type="checkbox" ${attrs}${on ? ' checked' : ''}><span>${body}</span><small>${n}</small></label>`;
    let html = courses.map(c => opt(`data-mixed="${esc(c.specCode)}"`, sel.mixed === c.specCode, `Mixed: all ${c.name ? esc(c.name) + ' ' : ''}topics`, c.count).replace('pk-opt', 'pk-opt mixed')).join('');
    courses.forEach(c => {
      const list = packs.filter(p => p.specCode === c.specCode);
      html += `<div class="pk-head">${c.name ? esc(c.name) + ': ' : ''}topics</div>` + list.map(p => opt(`data-id="${esc(p.id)}"`, !sel.mixed && sel.ids.includes(p.id), `<span class="rf">${esc(p.ref)}</span>${esc(p.topic)}`, p.count)).join('');
    });
    if (own.length) html += `<div class="pk-head">Your own sets</div>` + own.map(s => opt(`data-id="${esc(s.id)}"`, !sel.mixed && sel.ids.includes(s.id), esc(s.name), s.questions.length)).join('');
    if (!courses.length && !own.length) html += `<p class="pk-empty">${esc(B.subjectNote())}</p>`;
    html += `<div class="pk-foot"><label><input type="checkbox" id="pkHigher"${B.higher() ? ' checked' : ''}>Include Higher tier only questions</label><button class="btn go sm" type="button" data-done>Done</button></div>`;
    $('packMenu').innerHTML = html;
  }
  function openPacks(open) {
    const menu = $('packMenu'), btn = $('packBtn');
    const on = open === undefined ? menu.hidden : open;
    menu.hidden = !on; btn.setAttribute('aria-expanded', String(on));
    if (on) {
      // open towards the side with room, and bring all of it into view on short screens
      menu.style.left = menu.style.right = '';
      const r = menu.getBoundingClientRect();
      if (r.right > window.innerWidth - 8) { menu.style.left = 'auto'; menu.style.right = '0'; }
      menu.scrollIntoView({ block: 'nearest' });
      const first = menu.querySelector('input:checked') || menu.querySelector('input');
      if (first) first.focus({ preventScroll: true });
    }
  }
  function wirePacks() {
    const menu = $('packMenu'), B = CGB.bank;
    $('packBtn').addEventListener('click', () => openPacks());
    menu.addEventListener('change', e => {
      const t = e.target, keep = t.id || t.dataset.id || t.dataset.mixed;
      if (t.id === 'pkHigher') B.setHigher(t.checked);
      else if (t.dataset.mixed) B.setSelection(t.checked ? { mixed: t.dataset.mixed } : { ids: [] });
      else if (t.dataset.id) B.toggle(t.dataset.id);
      // the menu is drawn again: keep the keyboard on the box just changed
      const again = t.id ? document.getElementById(t.id) : menu.querySelector(t.dataset.id ? `[data-id="${CSS.escape(keep)}"]` : `[data-mixed="${CSS.escape(keep)}"]`);
      if (again) again.focus();
    });
    menu.addEventListener('click', e => { if (e.target.closest('[data-done]')) { openPacks(false); $('packBtn').focus(); } });
    menu.addEventListener('keydown', e => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); openPacks(false); $('packBtn').focus(); } });
    document.addEventListener('pointerdown', e => { if (!menu.hidden && !e.target.closest('#packDrop')) openPacks(false); });
  }
  /* The mascot on the launcher: the host waves hello now and then */
  function createMascot() {
    const box = $('mascot');
    const host = CGB.createHost2D(box, { className: 'host-launcher' });
    let timer = 0;
    const loop = () => { host.gesture(Math.random() < 0.65 ? 'wave' : 'present', 2200); host.talk(900); timer = setTimeout(loop, 7000 + Math.random() * 4000); };
    return {
      resume() { clearTimeout(timer); timer = setTimeout(loop, 600); },
      pause() { clearTimeout(timer); },
      wave() { host.gesture('wave', 2200); host.talk(900); }
    };
  }

  function init() {
    $('wordmark').innerHTML = CGB.brand.wordmark();
    $('artOTE').innerHTML = CGB.brand.oteArt();
    $('artOP').innerHTML = CGB.brand.opArt();
    $('artCC').innerHTML = CGB.brand.ccArt();
    $('artHH').innerHTML = CGB.brand.hhArt();
    CGB.renderSettings($('launcherSettings'), true);
    CGB.renderSettings($('modalSettings'), false);
    CGB.bankUI.init();
    // the greeting never introduces him; his name label shows only if the teacher has given him one
    const paintHello = () => { const n = CGB.hostName(); $('mascotWho').textContent = n; $('mascotWho').hidden = !n; };
    paintHello();
    CGB.hostListeners.push(paintHello);
    document.querySelectorAll('[data-play]').forEach(b => b.addEventListener('click', () => show(b.dataset.play)));
    $('openBank').addEventListener('click', () => CGB.bankUI.open());
    $('subjectSelect').addEventListener('change', e => CGB.bank.setSubject(e.target.value));
    $('boardSelect').addEventListener('change', e => CGB.bank.setBoard(e.target.value));
    wirePacks();
    $('openHost').addEventListener('click', () => CGB.hostEditor.open());
    $('openAbout').addEventListener('click', () => CGB.modal.open('aboutModal'));
    $('leaveConfirm').addEventListener('click', () => { CGB.modal.close('leaveModal'); show('launcher'); });
    $('aboutVersion').textContent = CGB.VERSION;
    $('storageNote').hidden = CGB.store.available;
    CGB.bank.onChange(renderSubject);
    renderSubject();
    CGB.hostEditor.init();
    mascot = createMascot();
    window.addEventListener('hashchange', () => { const r = routes[location.hash.slice(1)]; show(r || 'launcher'); });
    const start = routes[location.hash.slice(1)];
    if (start) show(start);
    else if (mascot) mascot.resume();
  }
  /* "Change" on a setup card: back to the main screen with the subject panel in focus */
  function chooseSubject() {
    show('launcher');
    $('subjectSelect').focus();
  }
  return { init, show, requestLauncher, chooseSubject, get current() { return current; } };
})();
