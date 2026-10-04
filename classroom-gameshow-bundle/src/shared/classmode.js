'use strict';
/* =========================================================
   THE CLASS ROUND (shared by all four games)
   A whole class of about 30 plays in 2 to 6 teams with mini whiteboards.
   Every team answers every question:
     1. the question shows, with a 20-second countdown;
     2. Space (or the end of the countdown) brings up "3, 2, 1, show me!";
     3. the teacher marks each team: keys 1 to 6 toggle a team, C marks
        every team correct, W every team wrong, Enter confirms;
     4. the game shows the answer and applies the result;
     5. U undoes the marking until the next question starts.
   This file holds the pieces the games share: the team panels, the
   question round (countdown, show me, marking, undo), the class
   misconceptions summary and the host's reactions.
   ========================================================= */
(function () {
  const CGB = window.CGB;
  const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const pick = a => a[Math.floor(Math.random() * a.length)];   // caption variety only

  CGB.COUNTDOWN = 20;    // seconds to think before "3, 2, 1, show me!" (Space skips it)
  CGB.TEAM_RULE = 'A team is correct only if its whiteboards agree on a correct answer, or if you judge it correct.';

  /* Marty's reaction to a marked question: "Five out of six teams! Brilliant!" */
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
     o.seconds() countdown length (0 = off)
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
    let phase = 'idle', marks = [], timer = null, left = 0, steps = [], times = {}, undoable = false;
    const prompt = document.createElement('div');
    prompt.className = 'cm-showme';
    prompt.setAttribute('aria-hidden', 'true');
    o.root.appendChild(prompt);
    const live = document.createElement('div');
    live.className = 'sr-only';
    live.setAttribute('aria-live', 'polite');
    o.root.appendChild(live);
    function clear() { clearInterval(timer); timer = null; steps.forEach(clearTimeout); steps = []; prompt.classList.remove('on', 'go'); }
    function paintCount() {
      if (!o.countEl) return;
      const s = o.seconds();
      const on = phase === 'think' && s > 0;
      o.countEl.hidden = !on;
      if (!on) return;
      o.countEl.innerHTML = `<div class="cm-cbar"><i style="width:${(left / s * 100).toFixed(1)}%"></i></div><b>${Math.ceil(left)} s</b>`;
      o.countEl.classList.toggle('low', left <= 5);
    }
    function buttons() {
      if (!o.btnEl) return;
      const k = (key, label, cls, act) => `<button class="btn ${cls}" type="button" data-cm="${act}">${label} <span class="kbd">${key}</span></button>`;
      if (phase === 'think') o.btnEl.innerHTML = `<div class="cm-hint">Everyone writes an answer on their whiteboard.</div>${k('Space', '3, 2, 1, show me!', 'go', 'show')}`;
      else if (phase === 'show') o.btnEl.innerHTML = '<div class="cm-hint">Boards up!</div>';
      else if (phase === 'mark' && o.marker) o.btnEl.innerHTML = `<div class="cm-hint">${o.marker.hint}</div>${o.marker.instant ? '' : `<div class="cm-btns">${k('Enter', 'Confirm', 'go', 'confirm')}</div>`}`;
      else if (phase === 'mark') o.btnEl.innerHTML = `<div class="cm-hint">Mark each team: tap its panel or press its number, 1 to ${marks.length}.</div>
        <div class="cm-btns">${k('C', '✓ All correct', 'plain', 'all')}${k('W', '✗ All wrong', 'plain', 'none')}${k('Enter', 'Confirm', 'go', 'confirm')}</div>`;
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
      left = o.seconds();
      paintCount(); buttons();
      if (left > 0) timer = setInterval(() => {
        left = Math.max(0, left - 0.1);
        paintCount();
        if (left <= 0) { clearInterval(timer); showMe(); }
      }, 100);
    }
    function showMe() {
      if (phase !== 'think') return;
      clear();
      phase = 'show'; times.show = performance.now();
      paintCount(); buttons();
      const words = CGB.settings.reduced() || o.fast ? ['Show me!'] : ['3', '2', '1', 'Show me!'];
      const gap = 600;
      words.forEach((t, k) => steps.push(setTimeout(() => {
        prompt.textContent = t;
        prompt.classList.add('on');
        prompt.classList.toggle('go', k === words.length - 1);
        if (k === words.length - 1) live.textContent = 'Show me! Boards up.';
        if (CGB.sfx && CGB.sfx.tick) CGB.sfx.tick(k === words.length - 1);
      }, k * gap)));
      steps.push(setTimeout(startMarking, words.length * gap + 300));
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
      if (phase === 'think' && (k === ' ' || k === 'enter')) { showMe(); return true; }
      if (phase === 'show' && (k === ' ' || k === 'enter')) { startMarking(); return true; }
      if (phase === 'mark' && o.marker) {
        if (k === 'enter' && !o.marker.instant) { confirm(); return true; }
        return o.marker.key(k) || k === ' ' || k === 'enter';
      }
      if (phase === 'mark') {
        if (/^[1-6]$/.test(k)) { toggle(+k - 1); return true; }
        if (k === 'c') { setAll(true); return true; }
        if (k === 'w') { setAll(false); return true; }
        if (k === 'enter') { confirm(); return true; }
        return k === ' ';
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
