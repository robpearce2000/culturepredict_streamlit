'use strict';

/* The answer as the games show it: the expected answer, then any other accepted answers and the
   teacher's note in smaller type (built-in packs have both; teachers' own sets may) */
CGB.answerHTML = q => {
  const esc = CGB.escapeHtml;
  if (!q) return '';
  let h = `<span class="ans-main">${esc(q.a)}</span>`;
  if (q.accept && q.accept.length) h += ` <span class="ans-also">Also accept: ${q.accept.map(esc).join('; ')}</span>`;
  if (q.notes) h += ` <span class="ans-note">${esc(q.notes)}</span>`;
  return h;
};
/* =========================================================
   PAUSE (P, the Pause button in every game's top bar, and automatically behind any prompt)
   While paused nothing in a game moves on: the answer countdowns, Outpace's sprint clock, the
   "3, 2, 1", the 3D physics and every big moment. Games read the time from CGB.pause.now() (a
   clock that stops while paused) and schedule game events with CGB.pause.after(); on-screen CSS
   animations are frozen too. A manual pause shows a large banner; a prompt (such as "Leave this
   game?") holds the game paused while it is open without a banner, and lets go when it closes.
   ========================================================= */
CGB.pause = (() => {
  let manual = false, holds = 0, pausedAt = 0, total = 0, seq = 0, ticker = 0, banner = null;
  const pending = [], listeners = [];
  const on = () => manual || holds > 0;
  const now = () => (on() ? pausedAt : performance.now()) - total;
  function changed(was) {
    const is = on();
    if (is !== was) {
      if (is) pausedAt = performance.now(); else total += performance.now() - pausedAt;
      document.documentElement.classList.toggle('cgb-paused', is);
      listeners.forEach(fn => { try { fn(is); } catch (e) { /* a listener must not stop the others */ } });
    }
    paint();
  }
  function paint() {
    if (!banner) {
      banner = document.createElement('div');
      banner.className = 'cgb-pausebar'; banner.hidden = true;
      banner.setAttribute('role', 'dialog'); banner.setAttribute('aria-label', 'Paused');
      banner.innerHTML = '<div class="cgb-pausecard"><b>Paused</b><span>Nothing moves on until you carry on.</span><button class="btn go" type="button" data-resume>Carry on <span class="kbd">P</span></button></div>';
      banner.addEventListener('click', e => { if (e.target.closest('[data-resume]')) set(false); });
      document.body.appendChild(banner);
    }
    banner.hidden = !manual;
    document.querySelectorAll('[data-pause]').forEach(b => { b.innerHTML = `${manual ? 'Carry on' : 'Pause'} <span class="kbd">P</span>`; b.setAttribute('aria-pressed', String(manual)); });
    if (manual) { const r = banner.querySelector('[data-resume]'); if (r) r.focus({ preventScroll: true }); }
  }
  function set(v) { const was = on(); manual = !!v; changed(was); }
  function tick() {
    if (on()) return;
    const t = now();
    for (let i = 0; i < pending.length; i++) {
      if (pending[i].due <= t) { const p = pending.splice(i--, 1)[0]; try { p.fn(); } catch (e) { setTimeout(() => { throw e; }); } }
    }
    if (!pending.length) { clearInterval(ticker); ticker = 0; }
  }
  return {
    on, now, set, toggle() { set(!manual); },
    get manual() { return manual; },
    hold() { const was = on(); holds++; changed(was); },
    release() { const was = on(); holds = Math.max(0, holds - 1); changed(was); },
    reset() { const was = on(); manual = false; holds = 0; changed(was); },
    // a timeout on the game clock: it waits while the game is paused
    after(ms, fn) { const id = ++seq; pending.push({ id, due: now() + ms, fn }); if (!ticker) ticker = setInterval(tick, 40); return id; },
    cancel(id) { const i = pending.findIndex(p => p.id === id); if (i >= 0) pending.splice(i, 1); },
    onChange(fn) { listeners.push(fn); },
    paint
  };
})();

/* There is no answer countdown: the question stays up until the teacher presses Space for "3, 2, 1, show me!" */

/* =========================================================
   THE CLASS ROUND (shared by all four games)
   A whole class of about 30 plays in 2 to 6 teams with mini whiteboards.
   Every team answers every question:
     1. the question shows and the teams write on their whiteboards, as long as they need;
     2. Space brings up "3, 2, 1, show me!";
     3. the teacher marks each team: keys 1 to 6 toggle a team, C marks
        every team correct, W every team wrong, Space or Enter confirms;
     4. the game shows the answer and applies the result;
     5. U undoes the marking until the next question starts.
   This file holds the pieces the games share: the team panels, the
   question round (show me, marking, undo), the class
   misconceptions summary and the host's reactions.
   ========================================================= */
(function () {
  const CGB = window.CGB;
  const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pick = a => a[Math.floor(Math.random() * a.length)];   // caption variety only

  CGB.answerSeconds = () => 0;   // no countdown
  CGB.timeLabel = t => t <= 60 ? `${t} seconds` : t % 60 ? `${Math.floor(t / 60)} minute${t >= 120 ? 's' : ''} ${t % 60} seconds` : `${t / 60} minute${t === 60 ? '' : 's'}`;
  CGB.answerTimeField = after => { const f = after.parentElement.querySelector(':scope > .answer-time'); if (f) f.remove(); };   // no Answer time choice any more
  CGB.TEAM_RULE = 'A team is correct only if its whiteboards agree on a correct answer, or if you judge it correct.';

  /* The host's reaction to a marked question: "Five out of six teams! Brilliant!" */
  const WORDS = ['No', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'];
  CGB.classLine = (c, n) => {
    if (n > 1 && c === n) return pick([`All ${WORDS[n].toLowerCase()} teams! Brilliant!`, 'Every single team! Fantastic!']);
    if (c === 0) return pick(["No team this time. Let's look at the answer together.", 'A tricky one! Here is the answer.']);
    const s = `${WORDS[c]} out of ${WORDS[n].toLowerCase()} teams`;
    return c * 2 >= n ? `${s}! ${pick(['Brilliant!', 'Great work!', 'Lovely!'])}` : `${s}. ${pick(['One to go over.', 'Tricky!', 'Let us look at that one.'])}`;
  };
  CGB.classGesture = (c, n) => c * 2 >= n ? (c === n ? 'cheer' : 'clap') : 'groan';

  /* ---------- Team panels: score and this question's ✓ or ✗, big enough for the back of the room ---------- */
  CGB.createTeamBoard = function (el, opts) {
    opts = opts || {};
    el.classList.add('cm-board');
    if (opts.className) el.classList.add(opts.className);
    let teams = [], marks = [], marking = false, earned = [], turn = -1, badge = [], toggle = null, last = '';
    el.addEventListener('click', e => {
      const b = e.target.closest('.cm-team');
      if (b && marking && toggle) toggle(+b.dataset.i);
    });
    function render() {
      el.style.setProperty('--n', teams.length);
      el.dataset.n = teams.length;
      el.classList.toggle('marking', marking);
      const html = teams.map((t, i) => {
        const T = CGB.TEAMS[i], m = marks[i];
        const state = m === true ? ' ok' : m === false ? ' no' : '';
        const label = `${t.name}: ${t.score}${m === true ? ', correct' : m === false ? ', wrong' : ''}`;
        return `<button type="button" class="cm-team${state}${i === turn ? ' turn' : ''}${earned[i] ? ' earned' : ''}" data-i="${i}" style="--tc:${T.css};--tl:${T.light}" ${marking ? '' : 'tabindex="-1" aria-disabled="true"'} aria-label="${esc(label)}">
          <span class="cm-key" aria-hidden="true">${i + 1}</span>
          <span class="cm-nm">${T.mark} ${esc(t.name)}</span>
          <span class="cm-sc">${esc(t.score)}</span>
          <span class="cm-mark" aria-hidden="true">${m === true ? '✓' : m === false ? '✗' : marking ? '?' : ''}</span>
          ${earned[i] ? `<span class="cm-earn">${esc(earned[i])}</span>` : ''}
          ${badge[i] ? `<span class="cm-badge">${esc(badge[i])}</span>` : ''}
        </button>`;
      }).join('');
      if (html !== last) { el.innerHTML = html; last = html; }   // unchanged panels are left alone, so their animations run once
    }
    return {
      el,
      set(o) {
        if (o.teams) teams = o.teams;
        if ('marks' in o) marks = (o.marks || []).slice();
        if ('marking' in o) marking = !!o.marking;
        if ('earned' in o) earned = o.earned || [];
        if ('turn' in o) turn = o.turn;
        if ('badge' in o) badge = o.badge || [];
        render();
      },
      onToggle(fn) { toggle = fn; }
    };
  };

  /* ---------- One question for the whole class ----------
     o.root      element the "show me" prompt sits in
     o.board     a team board (marking happens on it)
     o.countEl   where the countdown bar goes
     o.btnEl     where the round's buttons go (the game adds its own after marking)
     o.teams()   number of teams
     o.onMark()  marking has started (optional)
     o.onConfirm(results)  results[i] is true for each correct team
     o.onUndo()  put the game back as it was before onConfirm
     o.doneHtml() the game's own buttons after marking (optional)
     o.fast      a shorter "show me" (Outpace's Final Sprint)
     o.marker    optional custom marking (Hex Hunt's two side buttons): { start(), key(k), values(), hint,
                 instant } used instead of the team board's ✓ and ✗. With instant, the marker's own
                 buttons confirm straight away (it calls confirm()), so there is no Confirm button */
  CGB.createClassRound = function (o) {
    let phase = 'idle', marks = [], timer = null, lastSpace = -1e9, steps = [], times = {}, undoable = false;
    const question = () => o.question ? o.question() : null;
    const prompt = document.createElement('div');
    prompt.className = 'cm-showme';
    prompt.setAttribute('aria-hidden', 'true');
    o.root.appendChild(prompt);
    const live = document.createElement('div');
    live.className = 'sr-only';
    live.setAttribute('aria-live', 'polite');
    o.root.appendChild(live);
    function clear() { clearInterval(timer); timer = null; steps.forEach(CGB.pause.cancel); steps = []; prompt.classList.remove('on', 'go'); }
    function paintCount() { if (o.countEl) o.countEl.hidden = true; }
    let lastPhase = null;
    function buttons() {
      if (o.onPhase && phase !== lastPhase) { lastPhase = phase; o.onPhase(phase); }
      if (!o.btnEl) return;
      const k = (key, label, cls, act) => `<button class="btn ${cls}" type="button" data-cm="${act}">${label} <span class="kbd">${key}</span></button>`;
      if (phase === 'think') o.btnEl.innerHTML = `<div class="cm-hint">Everyone writes an answer on their whiteboard. Press Space when every board is ready.</div><div class="cm-btns">${k('Space', '3, 2, 1, show me!', 'go', 'show')}</div>`;
      else if (phase === 'show') o.btnEl.innerHTML = '<div class="cm-hint">Boards up!</div>';
      else if (phase === 'mark' && o.marker) o.btnEl.innerHTML = `<div class="cm-hint">${o.marker.hint}</div>${o.marker.instant ? '' : `<div class="cm-btns">${k('Space', 'Confirm', 'go', 'confirm')}</div>`}`;
      else if (phase === 'mark') o.btnEl.innerHTML = `<div class="cm-hint">Mark each team: tap its panel or press its number, 1 to ${marks.length}.</div>
        <div class="cm-btns">${k('C', '✓ All correct', 'plain', 'all')}${k('W', '✗ All wrong', 'plain', 'none')}${k('Space', 'Confirm', 'go', 'confirm')}</div>`;
      else if (phase === 'done') o.btnEl.innerHTML = `<div class="cm-btns">${o.doneHtml ? o.doneHtml() : ''}${undoable ? k('U', 'Undo marking', 'plain', 'undo') : ''}</div>`;
      else o.btnEl.innerHTML = '';
    }
    if (o.btnEl) o.btnEl.addEventListener('click', e => {
      const b = e.target.closest('button[data-cm]'); if (!b) return;
      const a = b.dataset.cm;
      if (a === 'show') showMe(); else if (a === 'all') setAll(true); else if (a === 'none') setAll(false);
      else if (a === 'confirm') confirm(); else if (a === 'undo') undo();
    });
    function think() {
      clear();
      phase = 'think'; undoable = false;
      marks = new Array(o.teams()).fill(null);
      times = { think: performance.now() };
      if (o.board) o.board.set({ marks: [], marking: false, earned: [] });
      paintCount(); buttons();
    }
    function showMe() {
      if (phase !== 'think') return;
      clear();
      phase = 'show'; times.show = performance.now();
      paintCount(); buttons();
      const words = CGB.settings.reduced() || o.fast ? ['Show me!'] : ['3', '2', '1', 'Show me!'];
      const gap = 600;
      words.forEach((t, k) => steps.push(CGB.pause.after(k * gap, () => {
        prompt.textContent = t;
        prompt.classList.add('on');
        prompt.classList.toggle('go', k === words.length - 1);
        if (k === words.length - 1) live.textContent = 'Show me! Boards up.';
        if (CGB.sfx && CGB.sfx.tick) CGB.sfx.tick(k === words.length - 1);
      })));
      steps.push(CGB.pause.after(words.length * gap + 300, startMarking));
    }
    function startMarking() {
      clear();
      phase = 'mark'; times.mark = performance.now();
      if (o.marker) o.marker.start(); else o.board.set({ marks, marking: true });
      buttons();
      if (o.onMark) o.onMark();
    }
    function toggle(i) {
      if (phase !== 'mark' || i < 0 || i >= marks.length) return;
      marks[i] = marks[i] !== true;
      o.board.set({ marks });
    }
    function setAll(v) { if (phase !== 'mark') return; marks = marks.map(() => v); o.board.set({ marks }); }
    function confirm() {
      if (phase !== 'mark') return;
      phase = 'done'; times.confirm = performance.now(); undoable = true;
      if (o.marker) { o.onConfirm(o.marker.values()); buttons(); return; }
      marks = marks.map(m => m === true);
      o.board.set({ marks, marking: false });
      const c = marks.filter(Boolean).length;
      live.textContent = `${c} of ${marks.length} teams correct.`;
      o.onConfirm(marks.slice());
      buttons();
    }
    function undo() {
      if (phase !== 'done' || !undoable) return;
      undoable = false;
      o.onUndo();
      phase = 'mark';
      if (o.marker) o.marker.start(true); else o.board.set({ marks, marking: true, earned: [] });
      buttons();
    }
    function handleKey(k) {
      // a second Space within half a second is a repeat: it can't skip the countdown and the "3, 2, 1" in one go
      // (the half second counts from the last Space that did something, so holding Space down can't block it)
      if (k === ' ' && (phase === 'show' || phase === 'think' || phase === 'mark')) { const t = performance.now(); if (t - lastSpace < 500) return true; lastSpace = t; }
      if ((phase === 'show' || phase === 'mark') && k === 'a' && o.reveal) { o.reveal(); return true; }
      if (phase === 'think' && (k === ' ' || k === 'enter')) { showMe(); return true; }
      if (phase === 'show' && (k === ' ' || k === 'enter')) { startMarking(); return true; }
      if (phase === 'mark' && o.marker) {
        if ((k === 'enter' || k === ' ') && !o.marker.instant) { confirm(); return true; }
        return o.marker.key(k) || k === ' ' || k === 'enter';
      }
      if (phase === 'mark') {
        if (/^[1-6]$/.test(k)) { toggle(+k - 1); return true; }
        if (k === 'c') { setAll(true); return true; }
        if (k === 'w') { setAll(false); return true; }
        if (k === 'enter' || k === ' ') { confirm(); return true; }
        return false;
      }
      if (phase === 'done' && k === 'u' && undoable) { undo(); return true; }
      return false;
    }
    if (o.board) o.board.onToggle(toggle);
    return {
      think, showMe, confirm, undo, toggle, setAll, handleKey,
      lock() { undoable = false; if (phase === 'done') buttons(); },
      refresh: buttons,
      stop() { clear(); phase = 'idle'; undoable = false; paintCount(); buttons(); },
      get phase() { return phase; },
      get undoable() { return phase === 'done' && undoable; },
      marks: () => marks.slice(),
      times: () => Object.assign({}, times)
    };
  };

  /* ---------- "Start game": shown after setup, before anything starts ----------
     Every game shows the same large button over its game screen when the teacher finishes
     setup. Nothing ticks until it is pressed (Enter, Space, click or tap), which gives the
     teacher a moment to explain the rules and get the whiteboards out. Esc still leaves. */
  // opts.title  a heading above the note; opts.label  the button text (default "Start game"); opts.className  an extra class
  CGB.createStartGate = function (root, note, opts) {
    opts = opts || {};
    const el = document.createElement('div');
    el.className = 'start-gate' + (opts.className ? ' ' + opts.className : '');
    el.hidden = true;
    el.innerHTML = `<div class="start-gate-card" role="dialog" aria-modal="false" aria-label="${opts.title || 'Ready to start'}">
      ${opts.title ? `<h2 class="start-gate-title">${opts.title}</h2>` : ''}
      <p class="start-gate-note">${note}</p>
      <button class="btn go start-gate-btn" type="button">${opts.label || 'Start game'} <span class="kbd">Enter</span></button></div>`;
    root.appendChild(el);
    let go = null;
    function start() { if (!go) return; const f = go; go = null; el.hidden = true; f(); }
    el.querySelector('button').addEventListener('click', start);
    document.addEventListener('keydown', e => {
      if (!go || el.hidden || (CGB.modal && CGB.modal.isOpen()) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopImmediatePropagation(); start(); }
    }, true);
    return {
      // show the button; onStart runs once when it is pressed (note, if given, replaces the text)
      show(onStart, note) {
        if (note) el.querySelector('.start-gate-note').innerHTML = note;
        /* @test-only */
        if (window.__SHOWTIME_NOGATE__) { onStart(); return; }
        /* @end-test-only */
        go = onStart; el.hidden = false;
        el.querySelector('button').focus({ preventScroll: true });
      },
      hide() { go = null; el.hidden = true; },
      get open() { return !!go; }
    };
  };

  /* ---------- What the class found hardest: no individual student data ---------- */
  CGB.createMisconceptions = function () {
    let list = [];
    return {
      reset() { list = []; },
      // share: the fraction of the class (or of the teams) that got it wrong
      add(q, share, detail) { if (q) list.push({ q, share, detail, n: list.length }); },
      remove(q) { const i = list.map(x => x.q).lastIndexOf(q); if (i >= 0) list.splice(i, 1); },
      top(limit) { return list.filter(x => x.share > 0).sort((a, b) => b.share - a.share || a.n - b.n).slice(0, limit || 5); },
      html(limit) {
        const top = this.top(limit);
        const body = top.length
          ? top.map(x => `<div class="cm-mq"><div class="q">${esc(x.q.q)}</div><div class="a">${esc(x.q.a)}</div><div class="t">${esc(x.q.subject)}: ${esc(x.q.topic)} · ${esc(x.detail)}</div></div>`).join('')
          : '<div class="cm-none">No question caught out the class. Impressive!</div>';
        return `<section class="cm-miscon" aria-label="Class misconceptions"><h3>Reteach these</h3><p class="cm-sub">The questions most teams got wrong, with the answers and topics.</p>${body}</section>`;
      }
    };
  };
})();
