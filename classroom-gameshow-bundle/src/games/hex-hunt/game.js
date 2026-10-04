'use strict';
/* =========================================================
   HEX HUNT
   A hexagon board for two teams. Team 1 links left to right,
   Team 2 links top to bottom. Each hexagon is a question whose
   answer starts with the letter shown. Winning a hexagon lets
   that team pick the next one.
   ========================================================= */
(function () {
let game = null;

function createHexHunt() {
const $ = id => document.getElementById('hh-' + id);
const esc = CGB.escapeHtml;
const bank = CGB.bank;
const GAME_NAME = 'Hex Hunt';
const TEAM = [{ css: 'var(--ha)', mark: '●', goal: 'left to right' }, { css: 'var(--hb)', mark: '■', goal: 'top to bottom' }];
const SFX = {
  pick() { CGB.sfx.tone(620, 0.08, 'triangle', 0.06); },
  buzz() { CGB.sfx.tone(180, 0.18, 'square', 0.05); },
  claim() { CGB.sfx.tone(660, 0.12, 'triangle', 0.08); CGB.sfx.tone(990, 0.16, 'triangle', 0.08, 0.08); },
  wrong: CGB.sfx.wrong,
  win() { [392, 523, 659, 784, 1047, 1319].forEach((f, i) => CGB.sfx.tone(f, 0.22, 'triangle', 0.09, i * 0.09)); }
};

const S = {
  phase: 'home', size: CGB.store.getJSON('hh.size', 5), best: CGB.store.getJSON('hh.best', 1),
  teams: [], cells: [], picker: 0, firstPicker: 0, cursor: { c: 0, r: 0 }, open: null,
  step: null, answering: -1, tried: [false, false], answerShown: false, round: 1, focusWeak: false, path: null
};
if (![5, 6].includes(S.size)) S.size = 5;
if (![1, 3].includes(S.best)) S.best = 1;

/* ---------- Letter clues ---------- */
/* The clue is the first letter of the answer, ignoring a leading "a", "an" or "the".
   Answers that start with a number or symbol, list several options ("Any two: ...")
   or are just yes/no/true/false do not make good clues, so they are left out. */
function firstLetter(answer) {
  const a = String(answer).trim();
  if (/^(any\b|yes\b|no\b|true\b|false\b)/i.test(a)) return null;
  const s = a.replace(/^(the|a|an)\s+/i, '');
  const m = /^[A-Za-z]/.exec(s);
  return m ? m[0].toUpperCase() : null;
}
let deck = [];
function buildDeck() {
  const set = bank.active();
  const eligible = set.questions.filter(q => firstLetter(q.a));
  const weak = S.focusWeak ? new Set(S.teams.flatMap(t => bank.wrongLog(t.name).map(e => e.topic))) : null;
  const shuffled = eligible.map(q => [Math.random() - (weak && weak.has(q.topic) ? 0.5 : 0), q]).sort((x, y) => x[0] - y[0]).map(v => v[1]);
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
    if (S.cursor.c === cell.c && S.cursor.r === cell.r && S.phase === 'board') cls.push('cursor');
    if (pathSet.has(cell.c + ',' + cell.r)) cls.push('path');
    const label = cell.owner >= 0 ? `${cell.letter}, won by ${S.teams[cell.owner].name}` : `Letter ${cell.letter}`;
    out += `<g class="${cls.join(' ')}" data-c="${cell.c}" data-r="${cell.r}" role="gridcell" aria-label="${esc(label)}"><path d="${CGB.brand.hexPath(x, y, R - 3)}"/>`;
    if (cell.owner >= 0) out += `<text class="lt" x="${x}" y="${y + 2}" font-size="38">${cell.letter}</text><text class="mk" x="${x}" y="${y + 30}" font-size="22">${TEAM[cell.owner].mark}</text>`;
    else out += `<text class="lt" x="${x}" y="${y + 17}" font-size="48">${cell.letter}</text>`;
    out += '</g>';
  });
  svg.innerHTML = out;
  renderTeams();
}
function renderTeams() {
  const owned = t => S.cells.filter(c => c.owner === t).length;
  $('teams').innerHTML = S.teams.map((t, i) => `<div class="hh-team${i === S.picker && S.phase === 'board' ? ' turn' : ''}" style="--tc:${TEAM[i].css}"><span class="nm">${TEAM[i].mark} ${esc(t.name)} <small>${TEAM[i].goal} · ${owned(i)} hexagons</small></span><span class="sc">${S.best > 1 ? t.wins + (t.wins === 1 ? ' round' : ' rounds') : ''}</span></div>`).join('');
  const p = S.teams[S.picker];
  $('turn').innerHTML = p ? `<b style="--tc:${TEAM[S.picker].css}">${TEAM[S.picker].mark} ${esc(p.name)}</b>, pick a hexagon` : '';
  $('round').textContent = S.best > 1 ? `Round ${S.round} · first to ${Math.ceil(S.best / 2)} rounds` : '';
}
$('board').addEventListener('click', e => {
  const g = e.target.closest('.hh-hex'); if (!g) return;
  S.cursor = { c: +g.dataset.c, r: +g.dataset.r };
  openHex(S.cursor.c, S.cursor.r);
});
function moveCursor(dc, dr) {
  const n = S.size;
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
  S.phase = 'question'; S.open = cell; S.step = 'buzz'; S.answering = -1; S.tried = [false, false]; S.answerShown = false;
  showQuestion();
}
function showQuestion() {
  const cell = S.open;
  $('qLetter').textContent = cell.letter;
  $('qClue').textContent = `The answer begins with ${cell.letter}`;
  $('qTag').textContent = `${cell.q.subject} · ${cell.q.topic} · picked by ${S.teams[S.picker].name}`;
  $('qText').textContent = cell.q.q;
  $('qAnswer').textContent = cell.q.a;
  $('qAnswer').classList.toggle('shown', S.answerShown);
  $('qMsg').textContent = S.step === 'buzz' ? 'Hands up or buzz in! Which team answers first?' : '';
  $('q').hidden = false;
  renderQButtons();
}
function renderQButtons() {
  const B = $('qBtns');
  const ans = `<button class="btn plain" type="button" data-q="answer">${S.answerShown ? 'Hide answer' : 'Show answer'} <span class="kbd">A</span></button>`;
  if (S.step === 'buzz') {
    B.innerHTML = S.teams.map((t, i) => S.tried[i] ? '' : `<button class="btn buzz" type="button" data-q="buzz" data-i="${i}" style="--tc:${TEAM[i].css}">${TEAM[i].mark} ${esc(t.name)} answers <span class="kbd">${i + 1}</span></button>`).join('') +
      `<button class="btn plain" type="button" data-q="nobody">Nobody knows <span class="kbd">N</span></button>${ans}`;
  } else if (S.step === 'answer') {
    const t = S.teams[S.answering];
    B.innerHTML = `<button class="btn ok" type="button" data-q="correct">✓ ${esc(t.name)} correct <span class="kbd">C</span></button><button class="btn no" type="button" data-q="wrong">✗ Wrong <span class="kbd">W</span></button>${ans}`;
  } else if (S.step === 'claimed') {
    B.innerHTML = `<button class="btn go" type="button" data-q="next">Back to the board <span class="kbd">Enter</span></button>`;
  } else if (S.step === 'nobody') {
    B.innerHTML = `<button class="btn go" type="button" data-q="again">New question for this hexagon <span class="kbd">Enter</span></button><button class="btn plain" type="button" data-q="leave">Leave it and pick again</button>`;
  }
}
$('qBtns').addEventListener('click', e => {
  const b = e.target.closest('button[data-q]'); if (!b) return;
  const a = b.dataset.q;
  if (a === 'answer') toggleAnswer();
  else if (a === 'buzz') buzz(+b.dataset.i);
  else if (a === 'nobody') nobody();
  else if (a === 'correct') mark(true);
  else if (a === 'wrong') mark(false);
  else if (a === 'next') backToBoard();
  else if (a === 'again') newQuestionHere();
  else if (a === 'leave') backToBoard();
});
function toggleAnswer() { S.answerShown = !S.answerShown; $('qAnswer').classList.toggle('shown', S.answerShown); renderQButtons(); }
function buzz(i) {
  if (S.step !== 'buzz' || S.tried[i] || !S.teams[i]) return;
  SFX.buzz();
  S.step = 'answer'; S.answering = i;
  $('qMsg').innerHTML = `<span style="color:inherit">${TEAM[i].mark} ${esc(S.teams[i].name)}</span>, what's your answer?`;
  renderQButtons();
}
function mark(correct) {
  if (S.step !== 'answer') return;
  const i = S.answering, team = S.teams[i], cell = S.open;
  if (correct) {
    SFX.claim();
    cell.owner = i; team.won++;
    S.answerShown = true; $('qAnswer').classList.add('shown');
    $('qMsg').textContent = `Correct! The hexagon goes to ${team.name}.`;
    S.picker = i;
    S.step = 'claimed';
    const path = winningPath(i);
    if (path) { S.path = path; S.step = 'winning'; $('qBtns').innerHTML = ''; $('qMsg').textContent = `Correct! ${team.name} join their edges!`; setTimeout(() => { if (S.phase === 'question' && S.step === 'winning') roundWon(i); }, 700); return; }
  } else {
    SFX.wrong();
    team.wrong.push(cell.q); bank.logWrong(team.name, cell.q, GAME_NAME);
    S.tried[i] = true;
    const other = 1 - i;
    if (!S.tried[other]) { S.step = 'buzz'; $('qMsg').textContent = `Not quite. ${S.teams[other].name}, can you answer?`; }
    else { S.step = 'nobody'; S.answerShown = true; $('qAnswer').classList.add('shown'); $('qMsg').textContent = 'Nobody got it.'; }
  }
  renderQButtons();
}
function nobody() {
  if (S.step !== 'buzz') return;
  S.step = 'nobody'; S.answerShown = true; $('qAnswer').classList.add('shown');
  $('qMsg').textContent = 'Nobody knew this one.';
  renderQButtons();
}
function newQuestionHere() {
  if (S.step !== 'nobody') return;
  const cell = S.open;
  const q = drawQuestion(new Set(S.cells.map(c => c.q && c.q.q)));
  if (q) { cell.q = q; cell.letter = firstLetter(q.a); }
  S.step = 'buzz'; S.answering = -1; S.tried = [false, false]; S.answerShown = false;
  showQuestion();
}
function backToBoard() {
  if (S.step !== 'claimed' && S.step !== 'nobody') return;
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

/* ---------- Rounds and results ---------- */
function roundWon(i) {
  $('q').hidden = true;
  S.phase = 'won';
  S.teams[i].wins++;
  SFX.win();
  renderBoard();
  const need = Math.ceil(S.best / 2);
  const matchOver = S.teams[i].wins >= need;
  $('winTitle').textContent = `${S.teams[i].name} win${S.best > 1 && !matchOver ? ' the round' : ''}!`;
  $('winDetail').textContent = `${TEAM[i].mark} Joined ${TEAM[i].goal} in ${S.path.length} hexagons` + (S.best > 1 ? ` · ${S.teams[0].name} ${S.teams[0].wins}, ${S.teams[1].name} ${S.teams[1].wins}` : '');
  $('winBtn').innerHTML = (matchOver ? 'See the results' : 'Next round') + ' <span class="kbd">Enter</span>';
  $('win').hidden = false;
  $('winBtn').focus({ preventScroll: true });
}
function continueAfterWin() {
  if (S.phase !== 'won') return;
  $('win').hidden = true;
  const need = Math.ceil(S.best / 2);
  if (S.teams.some(t => t.wins >= need)) { showSummary(); return; }
  S.round++;
  S.firstPicker = 1 - S.firstPicker; S.picker = S.firstPicker;
  newBoard();
  S.phase = 'board';
  renderBoard();
}
$('winBtn').addEventListener('click', continueAfterWin);

function startGame() {
  const names = [0, 1].map(i => ($('name' + i).value.trim() || 'Team ' + (i + 1)).slice(0, 18));
  CGB.store.setJSON('hh.names', names);
  S.teams = names.map(name => ({ name, wins: 0, won: 0, wrong: [] }));
  S.focusWeak = $('focusWeak').checked;
  S.round = 1; S.firstPicker = 0; S.picker = 0;
  newBoard();
  if (!S.cells.some(c => c.q)) { $('setHint').textContent = 'This set has no answers that start with a letter. Choose another set.'; return; }
  S.phase = 'board';
  $('home').classList.remove('active'); $('summary').classList.remove('active');
  $('win').hidden = true; $('q').hidden = true; $('play').hidden = false;
  renderBoard();
}
$('startBtn').addEventListener('click', startGame);

function showSummary() {
  S.phase = 'summary';
  $('play').hidden = true; $('q').hidden = true; $('win').hidden = true;
  const [a, b] = S.teams;
  const title = a.wins === b.wins ? "It's a draw!" : (a.wins > b.wins ? a.name : b.name) + ' win the match!';
  const cols = S.teams.map((t, i) => {
    const missed = t.wrong.length ? t.wrong.map(q => `<div class="hh-missed">${esc(q.q)}<span class="a">${esc(q.a)}</span></div>`).join('') : '<div class="hh-none">No wrong answers this game.</div>';
    const hist = bank.weakTopics(t.name, 4).map(([tp, n]) => `<div class="hh-trow"><span>${esc(tp)}</span><span>${n} wrong</span></div>`).join('') || '<div class="hh-none">No history yet.</div>';
    return `<div class="hh-sum-p" style="--tc:${TEAM[i].css}"><h3>${TEAM[i].mark} ${esc(t.name)}</h3><div class="hint">${t.wins} round${t.wins === 1 ? '' : 's'} won · ${t.won} hexagons claimed</div><div class="hh-sub">Missed this game</div>${missed}<div class="hh-sub">Weakest topics, all games</div>${hist}<button class="linkish" type="button" data-clear="${i}">Clear ${esc(t.name)}'s history</button></div>`;
  }).join('');
  $('sumCard').innerHTML = `<h2>${esc(title)}</h2><div class="hh-sum-grid">${cols}</div>
    <div class="hh-sum-btns"><button class="btn go" type="button" id="hh-again">Play again <span class="kbd">Enter</span></button><button class="btn plain" type="button" id="hh-change">Change teams or settings</button><button class="btn plain" type="button" id="hh-menu2">Back to menu</button></div>`;
  $('summary').classList.add('active');
  $('summary').scrollTop = 0;
  $('again').onclick = startGame;
  $('change').onclick = goHome;
  $('menu2').onclick = () => CGB.app.requestLauncher();
  $('sumCard').querySelectorAll('[data-clear]').forEach(btn => { btn.onclick = () => CGB.armButton(btn, 'Tap again to clear', () => { bank.clearHistory(S.teams[+btn.dataset.clear].name); showSummary(); }); });
  $('again').focus({ preventScroll: true });
}
function goHome() {
  S.phase = 'home';
  $('play').hidden = true; $('q').hidden = true; $('win').hidden = true; $('summary').classList.remove('active');
  $('home').classList.add('active');
}

/* ---------- Setup screen ---------- */
$('logo').innerHTML = CGB.brand.hhLogo();
const savedNames = CGB.store.getJSON('hh.names', null);
if (Array.isArray(savedNames)) savedNames.forEach((n, i) => { if (n && i < 2) $('name' + i).value = n; });
function wireSeg(id, key, store) {
  const seg = $(id);
  const paint = () => seg.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.v === S[key])));
  seg.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; S[key] = +b.dataset.v; CGB.store.setJSON(store, S[key]); paint(); });
  paint();
}
wireSeg('segSize', 'size', 'hh.size');
wireSeg('segBest', 'best', 'hh.best');
$('focusWeak').checked = CGB.store.get('hh.focusWeak') === '1';
$('focusWeak').addEventListener('change', e => CGB.store.set('hh.focusWeak', e.target.checked ? '1' : '0'));
function renderSetSelect() {
  const sel = $('setSelect');
  sel.innerHTML = bank.all().map(s => `<option value="${esc(s.id)}">${esc(s.name)} (${s.questions.length} questions)</option>`).join('');
  sel.value = bank.active().id;
  const usable = bank.active().questions.filter(q => firstLetter(q.a)).length;
  $('setHint').textContent = `${usable} questions in this set have answers that start with a letter, so they can go on the board.`;
}
$('setSelect').addEventListener('change', e => bank.setActive(e.target.value));
bank.onChange(renderSetSelect);
$('openBank').addEventListener('click', () => CGB.bankUI.open());
renderSetSelect();

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
  if (S.phase === 'home') { if (k === 'enter' && !(e.target.matches && e.target.matches('button, select, textarea'))) { e.preventDefault(); startGame(); } return; }
  if (S.phase === 'summary') { if (k === 'enter' && !isButton) { e.preventDefault(); startGame(); } return; }
  if (S.phase === 'won') { if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); continueAfterWin(); } return; }
  if (inField) return;
  if (S.phase === 'board') {
    const mv = { arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1] }[k];
    if (mv) { e.preventDefault(); moveCursor(mv[0], mv[1]); return; }
    if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); openHex(S.cursor.c, S.cursor.r); }
    return;
  }
  if (S.phase === 'question') {
    if (k === 'a') toggleAnswer();
    else if ((k === '1' || k === '2') && S.step === 'buzz') buzz(+k - 1);
    else if (k === 'n' && S.step === 'buzz') nobody();
    else if (k === 'c' && S.step === 'answer') mark(true);
    else if (k === 'w' && S.step === 'answer') mark(false);
    else if ((k === 'enter' || k === ' ') && !isButton) {
      if (S.step === 'claimed') { e.preventDefault(); backToBoard(); }
      else if (S.step === 'nobody') { e.preventDefault(); newQuestionHere(); }
    }
  }
});

return {
  enter() { active = true; if (S.phase === 'home') { renderSetSelect(); $('startBtn').focus({ preventScroll: true }); } },
  exit() { active = false; if (S.phase !== 'home') goHome(); },
  inProgress: () => ['board', 'question', 'won'].includes(S.phase),
  _state: () => ({ phase: S.phase, step: S.step, cursor: Object.assign({}, S.cursor), picker: S.picker, answering: S.answering, size: S.size, q: S.open ? S.open.q : null, letter: S.open ? S.open.letter : null, wins: S.teams.map(t => t.wins), owners: S.cells.map(c => c.owner) })
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
