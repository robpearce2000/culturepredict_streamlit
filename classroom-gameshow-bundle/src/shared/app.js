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
CGB.settings.onChange(() => {
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
  function renderList() {
    const act = B.active().id;
    $('bankList').innerHTML = B.all().map(s => {
      const sum = B.summary(s);
      const tags = sum.subjects.map(x => `<span class="tag ${subjectTag(x)}">${esc(x)}</span>`).join('') +
        `<span class="tag">${sum.count} questions</span><span class="tag">${sum.topics.length} topics</span>` +
        (s.builtin ? '<span class="tag">Built-in</span>' : '<span class="tag own">Your set</span>');
      const acts = s.builtin
        ? `<button class="btn plain sm" type="button" data-act="copy" data-id="${esc(s.id)}">Copy to edit</button>`
        : `<button class="btn plain sm" type="button" data-act="rename" data-id="${esc(s.id)}">Rename</button>
           <button class="btn plain sm" type="button" data-act="edit" data-id="${esc(s.id)}">Edit questions</button>
           <button class="btn danger sm" type="button" data-act="delete" data-id="${esc(s.id)}">Delete</button>`;
      return `<div class="bank-item${s.id === act ? ' active' : ''}" data-id="${esc(s.id)}">
        <input type="radio" name="activeSet" value="${esc(s.id)}" ${s.id === act ? 'checked' : ''} aria-label="Use ${esc(s.name)}">
        <div><div class="nm">${esc(s.name)}${s.id === act ? ' <span class="tag own">In use</span>' : ''}</div><div class="tags">${tags}</div></div>
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
    $('bankSave').textContent = set ? 'Save changes' : 'Save as new set';
    $('bankCancel').hidden = !set;
    setStatus('bankFormStatus', '');
    if (set) { $('bankForm').scrollIntoView({ block: 'start', behavior: CGB.settings.reduced() ? 'auto' : 'smooth' }); $('bankName').focus(); }
  }
  function init() {
    $('bankList').addEventListener('change', e => { if (e.target.name === 'activeSet') { B.setActive(e.target.value); } });
    $('bankList').addEventListener('click', e => {
      const b = e.target.closest('button[data-act]'); if (!b) return;
      const set = B.get(b.dataset.id); if (!set) return;
      if (b.dataset.act === 'delete') CGB.armButton(b, 'Tap again to delete', () => { B.deleteSet(set.id); if (editing === set.id) startEdit(null); });
      else if (b.dataset.act === 'edit') startEdit(set);
      else if (b.dataset.act === 'copy') {
        const r = B.addSet(set.short + ' (my copy)', B.toText(set.questions));
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
      const r = editing ? B.updateSet(editing, name, text) : B.addSet(name, text);
      if (!r.ok) { setStatus('bankFormStatus', r.error, true); return; }
      setStatus('bankFormStatus', `${editing ? 'Saved' : 'Added'} "${r.set.name}" with ${r.set.questions.length} questions. It is now in use in every game.${CGB.store.available ? '' : ' Storage is blocked, so it will be lost when the page closes.'}`);
      if (!editing) { $('bankName').value = ''; $('bankText').value = ''; }
      else B.setActive(editing);
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
      renderActive();
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
  function renderActive() {
    const s = CGB.bank.active();
    $('activeSetName').textContent = s.name;
    $('activeSetCount').textContent = s.questions.length + ' questions';
  }

  /* The mascot on the launcher: Pip waves hello now and then */
  function createMascot() {
    const box = $('mascot');
    const host = CGB.createHost2D(box, { className: 'pip-launcher' });
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
    $('mascotHello').textContent = `Hello, I'm ${CGB.hostCfg.name}! Pick a game.`;
    CGB.hostListeners.push(() => { $('mascotHello').textContent = `Hello, I'm ${CGB.hostCfg.name}! Pick a game.`; });
    document.querySelectorAll('[data-play]').forEach(b => b.addEventListener('click', () => show(b.dataset.play)));
    $('openBank').addEventListener('click', () => CGB.bankUI.open());
    $('openAbout').addEventListener('click', () => CGB.modal.open('aboutModal'));
    $('leaveConfirm').addEventListener('click', () => { CGB.modal.close('leaveModal'); show('launcher'); });
    $('aboutVersion').textContent = CGB.VERSION;
    $('storageNote').hidden = CGB.store.available;
    CGB.bank.onChange(renderActive);
    renderActive();
    mascot = createMascot();
    window.addEventListener('hashchange', () => { const r = routes[location.hash.slice(1)]; show(r || 'launcher'); });
    const start = routes[location.hash.slice(1)];
    if (start) show(start);
    else if (mascot) mascot.resume();
  }
  return { init, show, requestLauncher, get current() { return current; } };
})();
