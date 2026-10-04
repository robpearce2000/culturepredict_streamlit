'use strict';
/* =========================================================
   HEX HUNT
   A 4 × 4 hexagon board for two teams, the two halves of the class.
   Team 1 links left to right, Team 2 links top to bottom. Each hexagon
   is a question whose answer starts with the letter shown. Everyone
   answers on whiteboards; the teacher enters the share of each half that
   got it right, and the higher share wins the hexagon (the choosing half
   on a tie). Winning a hexagon lets that half pick the next one.
   ========================================================= */
(function () {
let game = null;

function createHexHunt() {
const $ = id => document.getElementById('hh-' + id);
const esc = CGB.escapeHtml;
const bank = CGB.bank;
const GAME_NAME = 'Hex Hunt';
const TEAM = CGB.TEAMS.slice(0, 2).map((t, i) => Object.assign({ goal: ['left to right', 'top to bottom'][i] }, t));
const SFX = {
  pick() { CGB.sfx.tone(620, 0.08, 'triangle', 0.06); },
  claim() { CGB.sfx.tone(660, 0.12, 'triangle', 0.08); CGB.sfx.tone(990, 0.16, 'triangle', 0.08, 0.08); },
  wrong: CGB.sfx.wrong,
  win() { [392, 523, 659, 784, 1047, 1319].forEach((f, i) => CGB.sfx.tone(f, 0.22, 'triangle', 0.09, i * 0.09)); }
};

/* Fixed settings: one round on a 4 × 4 board (under 10 minutes with a class) */
const S = {
  phase: 'home', size: 4,
  teams: [], cells: [], picker: 0, cursor: { c: 0, r: 0 }, open: null, step: null, path: null
};

/* ---------- Letter clues ---------- */
/* The clue is the first letter of the answer's first significant word: a leading "a", "an" or
   "the" is skipped, and so is a leading "in", "on", "at" or "from" in a place answer ("In the
   right atrium" is R). Only short answers are used, so the letter really is where the answer
   starts: answers that open with a pronoun or a link word ("It decreases", "To stop..."),
   start with a number, symbol or formula, list options ("Any two: ..."), are yes/no/true/false,
   are explanations (over ten words) or whose main part is over five words are left out. CGB.hexLetter is shared with the tests. */
const SKIP_LEAD = /^((in|into|on|at|from)\s+)?((the|a|an)\s+)?/i;
const NOT_A_CLUE = /^(any|yes|no|true|false|it|its|it's|they|them|their|this|these|those|there|to|so|when|because|by|one|if|both|either|all|each|only|more|less|same|that|which|with|without|after|before|as|for|of|about|roughly|around|over|under|nearly)\b/i;
function firstLetter(answer) {
  const a = String(answer).trim();
  if (NOT_A_CLUE.test(a) || a.split(/\s+/).length > 10 || /[→⇌₀-₉ₙ]/.test(a.split(/[,(]/)[0])) return null;
  const core = a.split(/[.,;:(]|\s[-–]\s/)[0].trim();
  if (core.split(/\s+/).length > 5) return null;
  const s = core.replace(SKIP_LEAD, '');
  if (NOT_A_CLUE.test(s)) return null;
  const m = /^[A-Za-z]/.exec(s);
  return m ? m[0].toUpperCase() : null;
}
CGB.hexLetter = firstLetter;
let deck = [];
function buildDeck() {
  const set = bank.active();
  const eligible = set.questions.filter(q => firstLetter(q.a));
  // topics either half got wrong before come up a little more often
  const weak = new Set(S.teams.flatMap(t => bank.wrongLog(t.name).map(e => e.topic)));
  const shuffled = eligible.map(q => [CGB.random() - (weak.has(q.topic) ? 0.3 : 0), q]).sort((x, y) => x[0] - y[0]).map(v => v[1]);
  deck = shuffled;
  return eligible.length;
}
function drawQuestion(avoid) {
  if (!deck.length) buildDeck();
  if (!deck.length) return null;
  let i = deck.findIndex(q => !avoid || !avoid.has(q.q));
  if (i < 0) i = 0;
  return deck.splice(i, 1)[0];
}

/* ---------- Board geometry (pointy-top hexagons, odd rows shifted right) ---------- */
const R = 50, HW = Math.sqrt(3) * R, PAD = 46;
function centre(c, r) { return { x: PAD + HW / 2 + c * HW + (r % 2 ? HW / 2 : 0), y: PAD + R + r * 1.5 * R }; }
function neighbours(c, r) {
  const odd = r % 2;
  const d = odd ? [[1, 0], [-1, 0], [0, -1], [1, -1], [0, 1], [1, 1]] : [[1, 0], [-1, 0], [-1, -1], [0, -1], [-1, 1], [0, 1]];
  return d.map(([dc, dr]) => [c + dc, r + dr]).filter(([x, y]) => x >= 0 && y >= 0 && x < S.size && y < S.size);
}
const cellAt = (c, r) => S.cells[r * S.size + c];
/* Has team t joined its two edges? Returns the winning path or null. */
function winningPath(t) {
  const n = S.size;
  const starts = [];
  for (let i = 0; i < n; i++) { const [c, r] = t === 0 ? [0, i] : [i, 0]; if (cellAt(c, r).owner === t) starts.push([c, r]); }
  const prev = new Map(), seen = new Set(starts.map(p => p.join(',')));
  const queue = starts.slice();
  while (queue.length) {
    const [c, r] = queue.shift();
    if ((t === 0 && c === n - 1) || (t === 1 && r === n - 1)) {
      const path = []; let k = c + ',' + r;
      while (k) { path.push(k); k = prev.get(k); }
      return path;
    }
    for (const [x, y] of neighbours(c, r)) {
      const k = x + ',' + y;
      if (seen.has(k) || cellAt(x, y).owner !== t) continue;
      seen.add(k); prev.set(k, c + ',' + r); queue.push([x, y]);
    }
  }
  return null;
}

/* ---------- Rendering ---------- */
function renderBoard() {
  const n = S.size;
  const w = PAD * 2 + HW * n + HW / 2, h = PAD * 2 + R * 2 + (n - 1) * 1.5 * R;
  const svg = $('board');
  svg.setAttribute('viewBox', `0 0 ${w.toFixed(0)} ${h.toFixed(0)}`);
  const pathSet = new Set(S.path || []);
  let out = '';
  // team edges: Team 1 owns left and right, Team 2 owns top and bottom
  out += `<rect class="hh-edge-0" x="6" y="${PAD}" width="18" height="${h - PAD * 2}" rx="9"/><rect class="hh-edge-0" x="${w - 24}" y="${PAD}" width="18" height="${h - PAD * 2}" rx="9"/>`;
  out += `<rect class="hh-edge-1" x="${PAD}" y="6" width="${w - PAD * 2}" height="18" rx="9"/><rect class="hh-edge-1" x="${PAD}" y="${h - 24}" width="${w - PAD * 2}" height="18" rx="9"/>`;
  S.cells.forEach(cell => {
    const { x, y } = centre(cell.c, cell.r);
    const cls = ['hh-hex'];
    if (cell.owner >= 0) cls.push('own-' + cell.owner);
    // the gold outline is the keyboard cursor, so it only shows once the arrow keys are used
    if (S.keyboard && S.cursor.c === cell.c && S.cursor.r === cell.r && S.phase === 'board') cls.push('cursor');
    if (pathSet.has(cell.c + ',' + cell.r)) cls.push('path');
    const label = cell.owner >= 0 ? `${cell.letter}, won by ${S.teams[cell.owner].name}` : `Letter ${cell.letter}`;
    out += `<g class="${cls.join(' ')}" data-c="${cell.c}" data-r="${cell.r}" role="gridcell" aria-label="${esc(label)}"><path d="${CGB.brand.hexPath(x, y, R - 3)}"/>`;
    if (cell.owner >= 0) out += `<text class="lt" x="${x}" y="${y + 2}" font-size="38">${cell.letter}</text><text class="mk" x="${x}" y="${y + 30}" font-size="22">${TEAM[cell.owner].mark}</text>`;
    else out += `<text class="lt" x="${x}" y="${y + 17}" font-size="48">${cell.letter}</text>`;
    out += '</g>';
  });
  svg.innerHTML = out;
  $('curHint').hidden = !(S.keyboard && S.phase === 'board');
  renderTeams();
}
function renderTeams() {
  const owned = t => S.cells.filter(c => c.owner === t).length;
  $('teams').innerHTML = S.teams.map((t, i) => `<div class="hh-team${i === S.picker && S.phase === 'board' ? ' turn' : ''}" style="--tc:${TEAM[i].css}"><span class="nm">${TEAM[i].mark} ${esc(t.name)} <small>${TEAM[i].goal} · ${owned(i)} hexagons</small></span></div>`).join('');
  const p = S.teams[S.picker];
  $('turn').innerHTML = p ? `<b style="--tc:${TEAM[S.picker].light}">${TEAM[S.picker].mark} ${esc(p.name)}</b>, pick a hexagon<small>Captain: swap to the next person</small>` : '';
}
$('board').addEventListener('click', e => {
  const g = e.target.closest('.hh-hex'); if (!g) return;
  S.cursor = { c: +g.dataset.c, r: +g.dataset.r };
  S.keyboard = false;
  openHex(S.cursor.c, S.cursor.r);
});
function moveCursor(dc, dr) {
  const n = S.size;
  if (!S.keyboard) { S.keyboard = true; renderBoard(); return; }     // the first press just shows where it is
  S.cursor.c = Math.max(0, Math.min(n - 1, S.cursor.c + dc));
  S.cursor.r = Math.max(0, Math.min(n - 1, S.cursor.r + dr));
  renderBoard();
}

/* ---------- Questions ---------- */
function newBoard() {
  const n = S.size;
  buildDeck();
  S.cells = [];
  const used = new Set();
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const q = drawQuestion(used); if (q) used.add(q.q);
    S.cells.push({ c, r, q, letter: q ? firstLetter(q.a) : '?', owner: -1, tries: 0 });
  }
  S.path = null;
  S.cursor = { c: Math.floor(n / 2), r: Math.floor(n / 2) };
}
function openHex(c, r) {
  if (S.phase !== 'board') return;
  const cell = cellAt(c, r);
  if (!cell || cell.owner >= 0 || !cell.q) return;
  SFX.pick();
  S.phase = 'question'; S.open = cell; S.step = 'ask';
  showQuestion();
}
function showQuestion() {
  const cell = S.open;
  $('qLetter').textContent = cell.letter;
  $('qClue').textContent = `The answer begins with ${cell.letter}`;
  $('qTag').textContent = `${cell.q.subject} · ${cell.q.topic} · picked by ${S.teams[S.picker].name}`;
  $('qText').textContent = cell.q.q;
  $('qAnswer').textContent = cell.q.a;
  $('qAnswer').classList.remove('shown');
  $('qMsg').textContent = 'Everyone answers on their whiteboard.';
  $('q').hidden = false;
  $('share').hidden = true;
  round.think();
  $('qcard').focus({ preventScroll: true });
}
$('qBtns').addEventListener('click', e => {
  const b = e.target.closest('button[data-q]'); if (!b) return;
  const a = b.dataset.q;
  if (a === 'next') backToBoard();
  else if (a === 'again') newQuestionHere();
  else if (a === 'leave') backToBoard();
  else if (a === 'win' && S.step === 'winning') roundWon(S.picker);
});
/* ---------- Whole class: which half had more right answers ----------
   The teacher compares the whiteboards and presses the half that had more correct answers (on a
   level count, the half that chose the hexagon), or Neither when nobody got it. One press marks it;
   Undo puts it back. */
const misc = CGB.createMisconceptions();
let choice = null, undoSnap = null;
function paintChoice() {
  $('share').hidden = false;
  $('share').innerHTML = S.teams.map((t, h) => `<button type="button" class="hh-side" data-side="${h}" style="--tc:${TEAM[h].css}" aria-pressed="${choice === h}">
    <span class="kbd">${h + 1}</span><b>${TEAM[h].mark} ${esc(t.name)}</b><small>had more right</small></button>`).join('')
    + `<button type="button" class="hh-side neither" data-side="none" aria-pressed="${choice === 'none'}"><span class="kbd">N</span><b>Neither</b><small>nobody got it</small></button>`;
}
function choose(c) {
  if (round.phase !== 'mark') return;
  choice = c; paintChoice(); round.confirm();
}
$('share').addEventListener('click', e => {
  const b = e.target.closest('button[data-side]'); if (!b) return;
  choose(b.dataset.side === 'none' ? 'none' : +b.dataset.side);
});
const round = CGB.createClassRound({
  root: document.getElementById('game-hex-hunt'), board: null, countEl: $('count'), btnEl: $('qBtns'),
  seconds: () => CGB.COUNTDOWN, teams: () => 2,
  marker: {
    instant: true,
    hint: 'Compare the whiteboards. Which half had more right answers? If it\'s level, press the half that chose this hexagon.',
    start() { choice = null; paintChoice(); },
    key(k) {
      if (k === '1' || k === '2') { choose(+k - 1); return true; }
      if (k === 'n' || k === '0') { choose('none'); return true; }
      return false;
    },
    values: () => choice
  },
  doneHtml: () => S.step === 'winning' ? `<button class="btn go" type="button" data-q="win">They've joined their edges! <span class="kbd">Enter</span></button>`
    : S.step === 'nobody' ? `<button class="btn go" type="button" data-q="again">New question for this hexagon <span class="kbd">Enter</span></button><button class="btn plain" type="button" data-q="leave">Leave it and pick again</button>`
    : `<button class="btn go" type="button" data-q="next">Back to the board <span class="kbd">Enter</span></button>`,
  onConfirm: classResult, onUndo: undoClassResult
});
function classResult(c) {
  const cell = S.open;
  undoSnap = { picker: S.picker, won: S.teams.map(t => t.won), wrong: S.teams.map(t => t.wrong.length), path: S.path };
  const w = c === 'none' ? -1 : c;
  // the half with fewer right answers (both, if nobody got it) has the topic noted to come round again
  S.teams.forEach((t, h) => { if (h !== w) { t.wrong.push(cell.q); bank.logWrong(t.name, cell.q, GAME_NAME); } });
  $('qAnswer').classList.add('shown');
  $('share').hidden = true;
  if (w < 0) {
    misc.add(cell.q, 1, 'nobody got it');
    SFX.wrong();
    S.step = 'nobody';
    $('qMsg').textContent = 'Nobody got it, so nobody wins this hexagon.';
    hostC.say("Nobody got that one. Here's the answer.", 'shrug', 1500);
    return;
  }
  SFX.claim();
  cell.owner = w; S.teams[w].won++; S.picker = w;
  $('qMsg').textContent = `The hexagon goes to ${S.teams[w].name}.`;
  hostC.say(`${S.teams[w].name} take it!`, 'clap', 1500);
  const path = winningPath(w);
  if (path) { S.path = path; S.step = 'winning'; $('qMsg').textContent += ` ${S.teams[w].name} join their edges!`; }
  else S.step = 'claimed';
  renderBoard();
}
function undoClassResult() {
  const u = undoSnap, cell = S.open; if (!u) return;
  cell.owner = -1; S.picker = u.picker; S.path = u.path;
  S.teams.forEach((t, i) => { t.won = u.won[i]; while (t.wrong.length > u.wrong[i]) { t.wrong.pop(); bank.unlogWrong(t.name, cell.q); } });
  misc.remove(cell.q);
  $('qAnswer').classList.remove('shown');
  $('qMsg').textContent = 'Marking undone. Press the half that had more right answers.';
  S.step = 'ask'; undoSnap = null;
  renderBoard();
}
/* Marty in the corner: short captions at the big moments */
const hostC = CGB.createHostCorner($('host'));
const sumHost = CGB.createHostCorner(document.createElement('div'), { className: 'host-sum' });
function newQuestionHere() {
  if (S.step !== 'nobody') return;
  round.stop();
  const cell = S.open;
  const q = drawQuestion(new Set(S.cells.map(c => c.q && c.q.q)));
  if (q) { cell.q = q; cell.letter = firstLetter(q.a); }
  S.step = 'ask';
  showQuestion();
}
function backToBoard() {
  if (S.step !== 'claimed' && S.step !== 'nobody') return;
  round.stop();
  if (S.step === 'nobody') {
    // swap in a fresh question so the hexagon is not stuck on the one nobody knew
    const q = drawQuestion(new Set(S.cells.map(c => c.q && c.q.q)));
    if (q) { S.open.q = q; S.open.letter = firstLetter(q.a); }
  }
  $('q').hidden = true;
  S.phase = 'board'; S.open = null; S.step = null;
  if (cellAt(S.cursor.c, S.cursor.r).owner >= 0) {
    const free = S.cells.find(c => c.owner < 0); if (free) S.cursor = { c: free.c, r: free.r };
  }
  renderBoard();
}

/* ---------- The result ---------- */
function roundWon(i) {
  round.stop();
  $('q').hidden = true;
  S.phase = 'won';
  S.winner = i;
  SFX.win();
  renderBoard();
  $('winTitle').textContent = `${S.teams[i].name} win!`;
  $('winDetail').textContent = `${TEAM[i].mark} Joined ${TEAM[i].goal} in ${S.path.length} hexagons`;
  $('winBtn').innerHTML = 'See the results <span class="kbd">Enter</span>';
  $('win').hidden = false;
  hostC.say(`${S.teams[i].name} link their edges and win!`, 'cheer', 2200);
  $('winBtn').focus({ preventScroll: true });
}
function continueAfterWin() {
  if (S.phase !== 'won') return;
  $('win').hidden = true;
  showSummary();
}
$('winBtn').addEventListener('click', continueAfterWin);

function startGame() {
  if (!bank.active()) { renderPack(); return; }
  CGB.leaveField();
  const names = [0, 1].map(i => ($('name' + i).value.trim() || 'Team ' + (i + 1)).slice(0, 18));
  CGB.saveTeamNames(names);
  S.teams = names.map(name => ({ name, won: 0, wrong: [] }));
  S.picker = 0; S.winner = -1;
  misc.reset(); round.stop();
  newBoard();
  if (!S.cells.some(c => c.q)) { $('setHint').textContent = 'This set has no answers that start with a letter. Choose another set.'; return; }
  S.phase = 'board';
  $('home').classList.remove('active'); $('summary').classList.remove('active');
  $('win').hidden = true; $('q').hidden = true; $('play').hidden = false;
  renderBoard();
  hostC.say(`Welcome to Hex Hunt! ${S.teams[0].name} go left to right, ${S.teams[1].name} top to bottom.`, 'wave', 2200);
}
$('startBtn').addEventListener('click', startGame);

function showSummary() {
  S.phase = 'summary';
  $('play').hidden = true; $('q').hidden = true; $('win').hidden = true;
  const w = S.teams[S.winner];
  const title = w ? w.name + ' win!' : 'Game over';
  $('sumCard').innerHTML = `<h2>${esc(title)}</h2>${misc.html(5)}
    <div class="hh-sum-btns"><button class="btn go" type="button" id="hh-again">Play again <span class="kbd">Enter</span></button><button class="btn plain" type="button" id="hh-change">Change team names</button><button class="btn plain" type="button" id="hh-menu2">Back to menu</button></div>${CGB.REVIEW_NOTE}`;
  $('sumCard').prepend(sumHost.el);
  sumHost.say(w ? `${title} Fantastic hunting, everyone!` : 'Brilliant hunting, everyone!', 'cheer', 2200);
  hostC.quiet();
  $('summary').classList.add('active');
  $('summary').scrollTop = 0;
  $('again').onclick = startGame;
  $('change').onclick = goHome;
  $('menu2').onclick = () => CGB.app.requestLauncher();
  $('again').focus({ preventScroll: true });
}
function goHome() {
  if (CGB.fitSetups) CGB.fitSetups();
  round.stop();
  S.phase = 'home';
  $('play').hidden = true; $('q').hidden = true; $('win').hidden = true; $('summary').classList.remove('active');
  $('home').classList.add('active');
  renderPack();
}

/* ---------- Setup screen ---------- */
$('logo').innerHTML = CGB.brand.hhLogo();
const fillNames = () => CGB.teamNames(2).forEach((n, i) => { $('name' + i).value = n; });
fillNames();
function renderPack() {
  const ok = CGB.renderPackLine($('pack'));
  $('startBtn').disabled = !ok;
  const usable = ok ? bank.active().questions.filter(q => firstLetter(q.a)).length : 0;
  $('setHint').textContent = ok ? `${usable} of these questions have answers that start with a letter, so they can go on the board.` : '';
}
bank.onChange(() => { if (S.phase === 'home') renderPack(); });
renderPack();

function paintMute() { $('muteBtn').textContent = CGB.settings.get('sound') ? 'Sound on' : 'Sound off'; $('muteBtn').setAttribute('aria-pressed', String(!CGB.settings.get('sound'))); }
$('muteBtn').addEventListener('click', () => { CGB.settings.set('sound', !CGB.settings.get('sound')); CGB.sfx.unlock(); });
$('menuBtn').addEventListener('click', () => CGB.app.requestLauncher());
$('settingsBtn').addEventListener('click', () => CGB.modal.open('settingsModal'));
CGB.settings.onChange(k => { if (k === 'sound') paintMute(); });
paintMute();

let active = false;
document.addEventListener('keydown', e => {
  if (!active || CGB.modal.isOpen() || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { e.preventDefault(); CGB.app.requestLauncher(); return; }
  const k = e.key.toLowerCase();
  const isButton = e.target.matches && e.target.matches('button');
  const inField = e.target.matches && e.target.matches('input, textarea, select');
  if (S.phase === 'home') { if (k === 'enter' && !(e.target.matches && e.target.matches('button, select, textarea, summary'))) { e.preventDefault(); startGame(); } return; }
  if (S.phase === 'summary') { if (k === 'enter' && !isButton) { e.preventDefault(); startGame(); } return; }
  if (S.phase === 'won') { if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); continueAfterWin(); } return; }
  if (inField) return;
  if (S.phase === 'board') {
    const mv = { arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1] }[k];
    if (mv) { e.preventDefault(); moveCursor(mv[0], mv[1]); return; }
    if ((k === 'enter' || k === ' ') && !isButton && S.keyboard) { e.preventDefault(); openHex(S.cursor.c, S.cursor.r); }
    else if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); S.keyboard = true; renderBoard(); }   // shows the outline first
    return;
  }
  if (S.phase === 'question') {
    if (round.handleKey(k)) { e.preventDefault(); return; }
    if ((k === 'enter' || k === ' ') && !isButton) {
      e.preventDefault();
      if (S.step === 'claimed') backToBoard();
      else if (S.step === 'nobody') newQuestionHere();
      else if (S.step === 'winning') roundWon(S.picker);
    }
    return;
  }
});

return {
  enter() { active = true; if (S.phase === 'home') { fillNames(); renderPack(); $('startBtn').focus({ preventScroll: true }); } },
  exit() { active = false; if (S.phase !== 'home') goHome(); },
  inProgress: () => ['board', 'question', 'won'].includes(S.phase),
  _state: () => ({ keyboard: !!S.keyboard, round: round.phase, undoable: round.undoable, times: round.times(), choice, misconceptions: misc.top(5).map(x => x.q.q), phase: S.phase, step: S.step, cursor: Object.assign({}, S.cursor), picker: S.picker, size: S.size, q: S.open ? S.open.q : null, letter: S.open ? S.open.letter : null, winner: S.winner, owners: S.cells.map(c => c.owner), letters: S.cells.map(c => [c.letter, c.q ? c.q.a : null]) })
};
}

CGB.registerGame('hex-hunt', {
  title: 'Hex Hunt',
  init() { game = createHexHunt(); },
  enter() { if (game) game.enter(); },
  exit() { if (game) game.exit(); },
  inProgress() { return !!(game && game.inProgress()); },
  state() { return game && game._state(); }
});
})();
