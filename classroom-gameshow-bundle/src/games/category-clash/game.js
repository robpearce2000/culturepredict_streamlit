'use strict';
/* =========================================================
   CATEGORY CLASH
   A board of categories (topics) and points values.
   Small group: 2 to 4 teams take turns to pick a tile and answer it;
   wrong answers can be stolen.
   Whole class: 2 to 6 teams. The choosing team picks a tile, every team
   answers it on whiteboards, the choosing team wins full points if
   correct and every other correct team half. Harder questions sit lower
   on the board and are worth more.
   ========================================================= */
(function () {
let game = null;

function createCategoryClash() {
const $ = id => document.getElementById('cc-' + id);
const esc = CGB.escapeHtml;
const bank = CGB.bank;
const GAME_NAME = 'Category Clash';
const VALUES = [100, 200, 300, 400, 500];
const TEAM = CGB.TEAMS;
const SFX = {
  open() { CGB.sfx.tone(392, 0.12, 'triangle', 0.07); CGB.sfx.tone(784, 0.14, 'triangle', 0.06, 0.08); },
  star() { [880, 1175, 1568, 2093].forEach((f, i) => CGB.sfx.tone(f, 0.16, 'sine', 0.07, i * 0.07)); },
  correct: CGB.sfx.correct,
  wrong: CGB.sfx.wrong,
  timeUp() { CGB.sfx.tone(260, 0.45, 'square', 0.05, 0, 0.7); },
  win() { [523, 659, 784, 1047, 1319].forEach((f, i) => CGB.sfx.tone(f, 0.24, 'triangle', 0.09, i * 0.1)); }
};

const S = {
  mode: CGB.mode.get('category-clash'), session: CGB.store.get('cc.session') || 'plenary', rows: 5,
  order: [], pos: 0, catchUp: -1,
  phase: 'home', nTeams: CGB.store.getJSON('cc.nTeams', CGB.mode.get('category-clash') === 'class' ? 4 : 2), nCats: CGB.store.getJSON('cc.nCats', 5), timer: CGB.store.getJSON('cc.timer', 0),
  teams: [], cats: [], turn: 0, cursor: { c: 0, r: 0 },
  open: null, step: null, answerShown: false, timeLeft: 0, timerHandle: null, picked: null
};
if (![2, 3, 4, 5, 6].includes(S.nTeams)) S.nTeams = 2;
if (![4, 5, 6].includes(S.nCats)) S.nCats = 5;
if (!CGB.COUNTDOWNS.includes(S.timer)) S.timer = 0;
if (!CGB.SESSIONS.some(x => x.id === S.session)) S.session = 'plenary';
/* Whole-class session lengths: categories × rows (one question per tile, every team answers) */
const SESSION = { starter: { cats: 4, rows: 3 }, plenary: { cats: 5, rows: 4 }, full: { cats: 6, rows: 5 } };
const classMode = () => S.mode === 'class';
const catsWanted = () => classMode() ? SESSION[S.session].cats : S.nCats;
const maxTeams = () => classMode() ? 6 : 4;

/* ---------- Setup ---------- */
$('logo').innerHTML = CGB.brand.ccLogo();
const defaultNames = [0, 1, 2, 3, 4, 5].map(CGB.teamFallback);
const teamCount = () => Math.min(S.nTeams, maxTeams());
function renderNames() {
  const prev = [0, 1, 2, 3, 4, 5].map(i => { const el = $('name' + i); return el ? el.value : null; });
  $('names').innerHTML = [0, 1, 2, 3, 4, 5].slice(0, teamCount()).map(i => `<label style="--tc:${TEAM[i].css}"><span>${TEAM[i].mark} Team ${i + 1}</span><input id="cc-name${i}" type="text" maxlength="18" autocomplete="off" value="${esc(prev[i] || CGB.teamNames(6)[i])}"></label>`).join('');
  $('names').classList.toggle('many', teamCount() > 4);
}
const segPainters = [];
function wireSeg(id, key, store, text) {
  const seg = $(id);
  const val = v => text ? v : +v;
  const paint = () => seg.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(val(b.dataset.v) === (key === 'nTeams' ? teamCount() : S[key]))));
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    S[key] = val(b.dataset.v);
    if (text) CGB.store.set(store, S[key]); else CGB.store.setJSON(store, S[key]);
    if (key === 'mode') CGB.mode.set('category-clash', S.mode);
    applyMode();
    segPainters.forEach(f => f());
    if (key === 'nTeams' || key === 'mode') renderNames();
    if (key === 'nCats' || key === 'session' || key === 'mode') autoPickTopics();
    CGB.fitSetups();
  });
  segPainters.push(paint);
  paint();
}
function applyMode() {
  $('setupCard').dataset.mode = S.mode;
  const sess = SESSION[S.session];
  $('sessionHint').textContent = `${CGB.SESSIONS.find(x => x.id === S.session).note}: ${sess.cats} categories × ${sess.rows} questions, every team answers each one.`;
}
wireSeg('segMode', 'mode', 'category-clash.mode', true);
wireSeg('segSession', 'session', 'cc.session', true);
wireSeg('segTeams', 'nTeams', 'cc.nTeams');
wireSeg('segCats', 'nCats', 'cc.nCats');
wireSeg('segTimer', 'timer', 'cc.timer');
applyMode();
renderNames();
['optSteal', 'optPenalty', 'optStar', 'focusWeak'].forEach(id => {
  const saved = CGB.store.get('cc.' + id);
  if (saved != null) $(id).checked = saved === '1';
  $(id).addEventListener('change', e => CGB.store.set('cc.' + id, e.target.checked ? '1' : '0'));
});

/* Topics in the active set, each a candidate category */
function topicsOf(set) {
  const m = new Map();
  set.questions.forEach(q => { const k = q.topic; if (!m.has(k)) m.set(k, { topic: k, subject: q.subject, qs: [] }); m.get(k).qs.push(q); });
  return Array.from(m.values());
}
let chosen = [];
function autoPickTopics(shuffle) {
  const all = topicsOf(bank.active());
  const full = all.filter(t => t.qs.length >= 5), rest = all.filter(t => t.qs.length < 5);
  const mix = a => a.map(v => [CGB.random(), v]).sort((x, y) => x[0] - y[0]).map(v => v[1]);
  // take topics from each subject in turn, so a mixed set gives a mixed board
  const spread = list => {
    const bySub = new Map(); list.forEach(t => { if (!bySub.has(t.subject)) bySub.set(t.subject, []); bySub.get(t.subject).push(t); });
    const lanes = Array.from(bySub.values()), out = [];
    for (let i = 0; out.length < list.length; i++) lanes.forEach(l => { if (l[i]) out.push(l[i]); });
    return out;
  };
  const pool = spread(shuffle ? mix(full) : full).concat(shuffle ? mix(rest) : rest);
  chosen = pool.slice(0, catsWanted()).map(t => t.topic);
  renderTopics();
}
function renderTopics() {
  const all = topicsOf(bank.active());
  $('topics').innerHTML = all.map(t => `<button type="button" class="cc-topic" data-topic="${esc(t.topic)}" aria-pressed="${chosen.includes(t.topic)}">${esc(t.topic)} <small>${t.qs.length}</small></button>`).join('');
  const few = chosen.filter(c => (all.find(t => t.topic === c) || { qs: [] }).qs.length < 5).length;
  const rows = classMode() ? SESSION[S.session].rows : 5;
  $('topicHint').textContent = `${chosen.length} chosen (up to 6). ` + (all.length < 2 ? 'This set has only one topic. Add Topic lines to make more categories. ' : '') + (few ? `Topics with fewer than ${rows} questions leave gaps on the board.` : `Each category gets ${rows} questions from easiest (100) to hardest (${rows * 100}).`);
}
$('topics').addEventListener('click', e => {
  const b = e.target.closest('[data-topic]'); if (!b) return;
  const t = b.dataset.topic;
  if (chosen.includes(t)) chosen = chosen.filter(x => x !== t);
  else if (chosen.length < 6) chosen.push(t);
  renderTopics();
});
$('shuffle').addEventListener('click', () => autoPickTopics(true));
function renderSetSelect() {
  bank.fillSelect($('setSelect'));
}
$('setSelect').addEventListener('change', e => bank.setActive(e.target.value));
let lastSet = bank.active().id;
bank.onChange(() => { renderSetSelect(); if (bank.active().id !== lastSet) { lastSet = bank.active().id; autoPickTopics(); } });
$('openBank').addEventListener('click', () => CGB.bankUI.open());
renderSetSelect();
autoPickTopics();

/* ---------- Building the board ---------- */
function buildBoard() {
  const all = topicsOf(bank.active());
  const cats = (chosen.length ? chosen : all.slice(0, catsWanted()).map(t => t.topic)).map(name => all.find(t => t.topic === name)).filter(Boolean);
  const weak = $('focusWeak').checked ? new Set(S.teams.flatMap(t => bank.wrongLog(t.name).map(e => e.q))) : null;
  S.cats = cats.map(t => {
    // Row k (100 to 500 points) takes a question tagged Difficulty k+1. Untagged questions get a
    // level from their place in the topic's estimated order. If a row has no question at its
    // level, the nearest level is used, then any question left. Questions a team got wrong
    // before are preferred when that option is on.
    const untagged = t.qs.filter(q => !q.level).sort((a, b) => bank.estimateLevel(a) - bank.estimateLevel(b));
    const levelOf = q => q.level || 1 + Math.floor(untagged.indexOf(q) * 5 / untagged.length);
    // with fewer rows (shorter class sessions) the rows still run from easiest to hardest
    const values = VALUES.slice(0, S.rows);
    const want = k => S.rows === 5 ? k + 1 : 1 + Math.round(k * 4 / (S.rows - 1));
    const left = t.qs.slice();
    const rnd = a => a[Math.floor(CGB.random() * a.length)];
    const take = (k, exact) => {
      if (!left.length) return null;
      const gap = exact ? 0 : Math.min(...left.map(q => Math.abs(levelOf(q) - want(k))));
      const pool = left.filter(q => Math.abs(levelOf(q) - want(k)) === gap);
      if (!pool.length) return null;
      const missed = weak ? pool.filter(q => weak.has(q.q)) : [];
      const q = rnd(missed.length ? missed : pool);
      left.splice(left.indexOf(q), 1);
      return q;
    };
    // exact levels first, so a short row never uses up another row's question
    const picks = values.map((v, k) => take(k, true));
    values.forEach((v, k) => { if (!picks[k]) picks[k] = take(k, false); });
    return { topic: t.topic, subject: t.subject, tiles: picks.map((q, k) => ({ value: values[k], q, used: !q, wonBy: -1, star: false, empty: !q, right: 0 })) };
  });
  if ($('optStar').checked) {
    const live = S.cats.flatMap(c => c.tiles.filter(t => !t.empty && t.value >= 200));
    if (live.length) live[Math.floor(CGB.random() * live.length)].star = true;
  }
  S.cursor = { c: 0, r: 0 };
}

/* ---------- Rendering ---------- */
function renderBoard() {
  const b = $('board');
  b.style.gridTemplateColumns = `repeat(${S.cats.length}, minmax(0, 1fr))`;
  let html = S.cats.map(c => `<div class="cc-cat" role="columnheader"><b>${esc(c.topic)}</b><small>${esc(c.subject)}</small></div>`).join('');
  b.style.gridTemplateRows = `minmax(56px, auto) repeat(${S.rows}, minmax(0, 1fr))`;
  for (let r = 0; r < S.rows; r++) {
    html += S.cats.map((c, ci) => {
      const t = c.tiles[r];
      const cur = S.cursor.c === ci && S.cursor.r === r ? ' cursor' : '';
      if (t.empty) return `<button type="button" class="cc-tile used empty${cur}" disabled aria-label="${esc(c.topic)} ${t.value}: no question">—</button>`;
      if (t.used && classMode()) {
        const lead = t.wonBy >= 0 ? `<span class="mk">${TEAM[t.wonBy].mark}</span>` : `<span class="mk">${t.right ? '✓' : '✗'}</span>`;
        return `<button type="button" class="cc-tile used${cur}" disabled style="--tc:${t.wonBy >= 0 ? TEAM[t.wonBy].light : 'inherit'}" aria-label="${esc(c.topic)} ${t.value}: ${t.right} of ${S.teams.length} teams correct">${lead}${t.right} of ${S.teams.length} correct</button>`;
      }
      if (t.used) {
        const who = t.wonBy >= 0 ? S.teams[t.wonBy] : null;
        return `<button type="button" class="cc-tile used${cur}" disabled style="--tc:${who ? TEAM[t.wonBy].light : 'inherit'}" aria-label="${esc(c.topic)} ${t.value}: ${who ? 'won by ' + esc(who.name) : 'nobody'}"><span class="mk">${who ? TEAM[t.wonBy].mark : '✗'}</span>${who ? esc(who.name) : 'Nobody'}</button>`;
      }
      return `<button type="button" class="cc-tile${cur}" data-c="${ci}" data-r="${r}" aria-label="${esc(c.topic)} for ${t.value} points">${t.value}</button>`;
    }).join('');
  }
  b.innerHTML = html;
  renderTeams();
}
function renderTeams() {
  $('teams').style.setProperty('--n', S.teams.length);
  $('teams').innerHTML = S.teams.map((t, i) => `<div class="cc-team${i === S.turn && S.phase === 'board' ? ' turn' : ''}" style="--tc:${TEAM[i].css}">${i === S.turn && catchUpTurn() ? '<span class="cc-catch">Catch-up pick</span>' : ''}<span class="nm">${TEAM[i].mark} ${esc(t.name)}</span><span class="sc">${t.score}</span></div>`).join('');
  const t = S.teams[S.turn];
  $('turn').innerHTML = t ? `<b style="--turn:${TEAM[S.turn].light}">${TEAM[S.turn].mark} ${esc(t.name)}</b>, pick a category` + (classMode() ? `<small>${catchUpTurn() ? 'Catch-up pick! ' : ''}Captain: swap to the next person</small>` : '') : '';
}
$('board').addEventListener('click', e => {
  const b = e.target.closest('.cc-tile[data-c]'); if (!b) return;
  S.cursor = { c: +b.dataset.c, r: +b.dataset.r };
  openTile(S.cursor.c, S.cursor.r);
});
function moveCursor(dc, dr) {
  const nc = S.cats.length;
  S.cursor.c = (S.cursor.c + dc + nc) % nc;
  S.cursor.r = Math.max(0, Math.min(S.rows - 1, S.cursor.r + dr));
  renderBoard();
  const el = $('board').querySelector('.cursor'); if (el && !el.disabled) el.focus({ preventScroll: true });
}

/* ---------- A question ---------- */
function openTile(c, r) {
  const t = S.cats[c] && S.cats[c].tiles[r];
  if (!t || t.used || S.phase !== 'board') return;
  S.phase = 'question'; S.open = { c, r }; S.step = 'ask'; S.answerShown = false; S.picked = S.turn;
  const cat = S.cats[c];
  $('qCat').textContent = cat.subject + ' · ' + cat.topic;
  $('qVal').textContent = (t.star ? t.value * 2 : t.value) + ' points';
  $('qStar').hidden = !t.star;
  $('qFor').innerHTML = classMode()
    ? `Chosen by <b>${TEAM[S.turn].mark} ${esc(S.teams[S.turn].name)}</b> · every team answers: full points for ${esc(S.teams[S.turn].name)}, half for the others`
    : `For <b style="color:${'inherit'}">${TEAM[S.turn].mark} ${esc(S.teams[S.turn].name)}</b>`;
  $('qText').textContent = t.q.q;
  $('qAnswer').textContent = t.q.a;
  $('qAnswer').classList.remove('shown');
  $('qMsg').textContent = '';
  $('q').hidden = false;
  if (t.star) { SFX.star(); hostC.say('A star tile! This one is worth double.', 'cheer', 1600); } else SFX.open();
  $('qboard').hidden = !classMode();
  if (classMode()) { paintQBoard(); round.think(); $('qcard').focus({ preventScroll: true }); return; }
  startTimer();
  renderQButtons();
  $('qcard').focus({ preventScroll: true });   // keys now go to the question, not the board behind it
}
/* Marty in the corner: short captions at the big moments */
const hostC = CGB.createHostCorner($('host'));
const sumHost = CGB.createHostCorner(document.createElement('div'), { className: 'host-sum' });
const pickLine = a => a[Math.floor(Math.random() * a.length)];   // caption variety only

/* ---------- Whole class: every team answers, the teacher marks each team ---------- */
const misc = CGB.createMisconceptions();
const qBoard = CGB.createTeamBoard($('qboard'));
let undoSnap = null;
const round = CGB.createClassRound({
  root: document.getElementById('game-category-clash'),
  board: qBoard, countEl: $('count'), btnEl: $('qBtns'),
  seconds: () => S.timer, teams: () => S.teams.length,
  doneHtml: () => `<button class="btn go" type="button" data-q="next">${boardDone() ? 'See the results' : 'Back to the board'} <span class="kbd">Enter</span></button>`,
  onConfirm: classResult, onUndo: undoClassResult
});
function paintQBoard(earned) { qBoard.set({ teams: S.teams.map(t => ({ name: t.name, score: t.score })), turn: S.picked, earned: earned || [] }); }
function classResult(res) {
  const t = tileOpen(), base = t.star ? t.value * 2 : t.value;
  undoSnap = { scores: S.teams.map(x => x.score), correct: S.teams.map(x => x.correct), wrong: S.teams.map(x => x.wrong.length), wonBy: t.wonBy, right: t.right };
  const earned = res.map((ok, i) => ok ? (i === S.picked ? base : Math.round(base / 2)) : 0);
  res.forEach((ok, i) => {
    const team = S.teams[i];
    if (ok) { team.score += earned[i]; team.correct++; }
    else { team.wrong.push(t.q); bank.logWrong(team.name, t.q, GAME_NAME); }
  });
  const c = res.filter(Boolean).length, n = res.length;
  t.right = c; t.wonBy = res[S.picked] ? S.picked : -1;
  misc.add(t.q, (n - c) / n, `${n - c} of ${n} teams wrong`);
  showAnswer();
  if (c) SFX.correct(); else SFX.wrong();
  $('qMsg').textContent = `${c} of ${n} teams correct.` + (res[S.picked] ? ` ${S.teams[S.picked].name} win ${base} points; other correct teams win ${Math.round(base / 2)}.` : c ? ` Each correct team wins ${Math.round(base / 2)} points.` : '');
  S.step = 'done';
  paintQBoard(earned.map(e => e ? '+' + e : ''));
  qBoard.set({ marks: res });
  hostC.say(CGB.classLine(c, n), CGB.classGesture(c, n), 1600);
  renderTeams();
}
function undoClassResult() {
  const t = tileOpen(), u = undoSnap; if (!u) return;
  S.teams.forEach((team, i) => {
    team.score = u.scores[i]; team.correct = u.correct[i];
    while (team.wrong.length > u.wrong[i]) { team.wrong.pop(); bank.unlogWrong(team.name, t.q); }
  });
  t.wonBy = u.wonBy; t.right = u.right;
  misc.remove(t.q);
  S.answerShown = false; $('qAnswer').classList.remove('shown');
  $('qMsg').textContent = 'Marking undone. Mark each team again.';
  S.step = 'ask'; undoSnap = null;
  paintQBoard();
  renderTeams();
}
/* Each team chooses once per round. The last-placed team, if it trails the leader by at least
   the value of the top tile, chooses first in the next round: its catch-up pick. */
function startClassRound() {
  const n = S.teams.length, top = S.rows * 100;
  const scores = S.teams.map(t => t.score), lead = Math.max(...scores), last = scores.indexOf(Math.min(...scores));
  S.catchUp = S.order.length && lead - scores[last] >= top ? last : -1;
  const base = Array.from({ length: n }, (x, i) => i);
  S.order = S.catchUp >= 0 ? [S.catchUp].concat(base.filter(i => i !== S.catchUp)) : base;
  S.pos = 0;
}
function nextClassTurn() {
  S.pos++;
  if (S.pos >= S.order.length) startClassRound();
  S.turn = S.order[S.pos];
  if (catchUpTurn()) hostC.say(`${S.teams[S.turn].name}, you get a catch-up pick! Captain: swap to the next person.`, 'point', 2000);
}
const catchUpTurn = () => classMode() && S.pos === 0 && S.catchUp >= 0 && S.turn === S.catchUp;
function tileOpen() { return S.cats[S.open.c].tiles[S.open.r]; }
function points() { const t = tileOpen(); return t.star ? t.value * 2 : t.value; }
function renderQButtons() {
  const B = $('qBtns');
  const ans = `<button class="btn plain" type="button" data-q="answer">${S.answerShown ? 'Hide answer' : 'Show answer'} <span class="kbd">A</span></button>`;
  if (S.step === 'ask') {
    B.innerHTML = `<button class="btn ok" type="button" data-q="correct">✓ Correct <span class="kbd">C</span></button><button class="btn no" type="button" data-q="wrong">✗ Wrong <span class="kbd">W</span></button>${ans}`;
  } else if (S.step === 'steal') {
    B.innerHTML = S.teams.map((t, i) => i === S.picked ? '' : `<button class="btn steal" type="button" data-q="steal" data-i="${i}" style="--tc:${TEAM[i].css}">✓ ${TEAM[i].mark} ${esc(t.name)} <span class="kbd">${i + 1}</span></button>`).join('') +
      `<button class="btn plain" type="button" data-q="nobody">Nobody <span class="kbd">N</span></button>${ans}`;
  } else {
    B.innerHTML = `<button class="btn go" type="button" data-q="next">${boardDone() ? 'See the results' : 'Back to the board'} <span class="kbd">Enter</span></button>`;
  }
}
$('qBtns').addEventListener('click', e => {
  const b = e.target.closest('button[data-q]'); if (!b) return;
  const a = b.dataset.q;
  if (a === 'answer') toggleAnswer();
  else if (a === 'correct') markCorrect();
  else if (a === 'wrong') markWrong();
  else if (a === 'steal') steal(+b.dataset.i);
  else if (a === 'nobody') nobody();
  else if (a === 'next') closeQuestion();
});
function toggleAnswer() { S.answerShown = !S.answerShown; $('qAnswer').classList.toggle('shown', S.answerShown); renderQButtons(); }
function showAnswer() { S.answerShown = true; $('qAnswer').classList.add('shown'); }
function award(i, text) {
  const t = tileOpen(), p = points();
  S.teams[i].score += p; S.teams[i].correct++;
  t.wonBy = i;
  $('qMsg').textContent = `${text} +${p} points for ${S.teams[i].name}.`;
}
function markCorrect() {
  if (S.step !== 'ask') return;
  SFX.correct(); stopTimer(); showAnswer();
  award(S.picked, 'Correct!');
  hostC.say(pickLine([`Correct! ${points()} points to ${S.teams[S.picked].name}.`, `Spot on, ${S.teams[S.picked].name}!`, 'That is right!']), 'clap', 1500);
  finishTile();
}
function markWrong() {
  if (S.step !== 'ask') return;
  SFX.wrong();
  const team = S.teams[S.picked], t = tileOpen();
  team.wrong.push(t.q); bank.logWrong(team.name, t.q, GAME_NAME);
  let msg = `Not quite, ${team.name}.`;
  if ($('optPenalty').checked) { team.score -= points(); msg += ` −${points()} points.`; }
  if ($('optSteal').checked && S.teams.length > 1) { S.step = 'steal'; msg += ' Another team can steal it.'; }
  else { stopTimer(); showAnswer(); S.step = 'done'; }
  $('qMsg').textContent = msg;
  hostC.say(S.step === 'steal' ? 'Not quite. Can another team steal it?' : pickLine(['Not this time.', 'Ooh, not quite.']), 'groan', 1400);
  renderTeams(); renderQButtons();
}
function steal(i) {
  if (S.step !== 'steal' || i === S.picked || !S.teams[i]) return;
  SFX.correct(); stopTimer(); showAnswer();
  award(i, `Stolen by ${S.teams[i].name}!`);
  hostC.say(`Stolen! Well played, ${S.teams[i].name}.`, 'point', 1500);
  finishTile();
}
function nobody() {
  if (S.step !== 'steal' && S.step !== 'ask') return;
  stopTimer(); showAnswer();
  $('qMsg').textContent = 'Nobody wins this one.';
  hostC.say('Nobody wins that one. On we go!', 'shrug', 1400);
  S.step = 'done';
  renderQButtons();
}
function finishTile() { S.step = 'done'; renderTeams(); renderQButtons(); }
function boardDone() { return S.cats.every(c => c.tiles.every(t => t.used || (S.open && c === S.cats[S.open.c] && t === tileOpen()))); }
function closeQuestion() {
  if (S.step !== 'done') return;
  stopTimer();
  tileOpen().used = true;
  $('q').hidden = true;
  S.phase = 'board'; S.open = null;
  if (classMode()) { round.stop(); nextClassTurn(); }
  else S.turn = (S.turn + 1) % S.teams.length;
  if (S.cats.every(c => c.tiles.every(t => t.used))) { showSummary(); return; }
  // move the cursor to the next free tile
  const free = []; S.cats.forEach((c, ci) => c.tiles.forEach((t, r) => { if (!t.used) free.push({ c: ci, r }); }));
  if (!free.some(f => f.c === S.cursor.c && f.r === S.cursor.r)) S.cursor = free[0];
  renderBoard();
  const el = $('board').querySelector('.cursor'); if (el) el.focus({ preventScroll: true });
}

/* ---------- Timer ---------- */
function startTimer() {
  stopTimer();
  $('timer').hidden = !S.timer;
  if (!S.timer) return;
  S.timeLeft = S.timer;
  paintTimer();
  S.timerHandle = setInterval(() => {
    S.timeLeft = Math.max(0, S.timeLeft - 0.1);
    paintTimer();
    if (S.timeLeft <= 0) { stopTimer(); SFX.timeUp(); $('qMsg').textContent = "Time's up! Mark the answer."; }
  }, 100);
}
function stopTimer() { clearInterval(S.timerHandle); S.timerHandle = null; }
function paintTimer() {
  $('timerFill').style.width = (S.timeLeft / S.timer * 100) + '%';
  $('timerText').textContent = Math.ceil(S.timeLeft) + ' s';
  $('timer').classList.toggle('low', S.timeLeft <= 5);
}

/* ---------- Game flow ---------- */
function startGame() {
  const names = [0, 1, 2, 3, 4, 5].slice(0, teamCount()).map(i => (($('name' + i) || {}).value || '').trim().slice(0, 18) || defaultNames[i]);
  CGB.saveTeamNames(names);
  S.teams = names.map(name => ({ name, score: 0, correct: 0, wrong: [] }));
  S.turn = 0;
  S.rows = classMode() ? SESSION[S.session].rows : 5;
  S.order = []; S.catchUp = -1; misc.reset(); round.stop();
  if (classMode()) { startClassRound(); S.turn = S.order[0]; }
  buildBoard();
  if (!S.cats.length || S.cats.every(c => c.tiles.every(t => t.empty))) { $('topicHint').textContent = 'Choose at least one category with questions.'; return; }
  S.phase = 'board';
  $('home').classList.remove('active'); $('summary').classList.remove('active');
  $('play').hidden = false; $('q').hidden = true;
  renderBoard();
  hostC.say(`Welcome to Category Clash! ${S.teams[0].name}, you pick first.`, 'wave', 2000);
  const el = $('board').querySelector('.cursor'); if (el) el.focus({ preventScroll: true });
}
$('startBtn').addEventListener('click', startGame);
$('endBtn').addEventListener('click', () => CGB.armButton($('endBtn'), 'Tap again to end', () => { stopTimer(); $('q').hidden = true; showSummary(); }));

function showSummary() {
  S.phase = 'summary';
  stopTimer();
  $('q').hidden = true; $('play').hidden = true;
  const order = S.teams.map((t, i) => ({ t, i })).sort((a, b) => b.t.score - a.t.score);
  const top = order[0].t.score, winners = order.filter(o => o.t.score === top);
  SFX.win();
  const pos = ['1st', '2nd', '3rd', '4th', '5th', '6th'];
  let place = 0;
  const podium = order.map((o, k) => {
    if (k && o.t.score < order[k - 1].t.score) place = k;
    return `<div class="cc-rank" style="--tc:${TEAM[o.i].css}"><span class="pos">${pos[place]}</span><span class="nm">${TEAM[o.i].mark} ${esc(o.t.name)}<small>${o.t.correct} won · ${o.t.wrong.length} missed</small></span><span class="sc">${o.t.score}</span></div>`;
  }).join('');
  const cols = S.teams.map((t, i) => {
    const missed = t.wrong.length ? t.wrong.map(q => `<div class="cc-missed">${esc(q.q)}<span class="a">${esc(q.a)}</span></div>`).join('') : '<div class="cc-none">No wrong answers this game.</div>';
    const hist = bank.weakTopics(t.name, 4).map(([tp, n]) => `<div class="cc-trow"><span>${esc(tp)}</span><span>${n} wrong</span></div>`).join('') || '<div class="cc-none">No history yet.</div>';
    return `<div class="cc-sum-p" style="--tc:${TEAM[i].css}"><h3>${TEAM[i].mark} ${esc(t.name)}</h3><div class="cc-sub">Missed this game</div>${missed}<div class="cc-sub">Weakest topics, all games</div>${hist}<button class="linkish" type="button" data-clear="${i}">Clear ${esc(t.name)}'s history</button></div>`;
  }).join('');
  $('sumCard').innerHTML = `<h2>${winners.length > 1 ? "It's a draw!" : esc(winners[0].t.name) + ' win!'}</h2>
    <div class="cc-podium">${podium}</div>${classMode() ? misc.html(5) : `<div class="cc-sum-grid">${cols}</div>`}
    <div class="cc-sum-btns"><button class="btn go" type="button" id="cc-again">Play again <span class="kbd">Enter</span></button><button class="btn plain" type="button" id="cc-change">Change teams or categories</button><button class="btn plain" type="button" id="cc-menu2">Back to menu</button></div>${CGB.REVIEW_NOTE}`;
  $('sumCard').prepend(sumHost.el);
  sumHost.say(winners.length > 1 ? "A draw! What a close game. Well done, everyone." : `${winners[0].t.name} win with ${top} points. Great game, everyone!`, 'cheer', 2200);
  hostC.quiet();
  $('summary').classList.add('active');
  $('summary').scrollTop = 0;
  $('again').onclick = startGame;
  $('change').onclick = goHome;
  $('menu2').onclick = () => CGB.app.requestLauncher();
  $('sumCard').querySelectorAll('[data-clear]').forEach(btn => { btn.onclick = () => CGB.armButton(btn, 'Tap again to clear', () => { bank.clearHistory(S.teams[+btn.dataset.clear].name); showSummary(); }); });
  $('again').focus({ preventScroll: true });
}
function goHome() {
  if (CGB.fitSetups) CGB.fitSetups();
  stopTimer(); round.stop();
  S.phase = 'home'; S.open = null;
  $('q').hidden = true; $('play').hidden = true; $('summary').classList.remove('active');
  $('home').classList.add('active');
  renderTopics();
}

/* ---------- Top bar and keys ---------- */
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
  if (inField) return;
  if (S.phase === 'board') {
    const mv = { arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1] }[k];
    if (mv) { e.preventDefault(); moveCursor(mv[0], mv[1]); return; }
    if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); openTile(S.cursor.c, S.cursor.r); }
    return;
  }
  if (S.phase === 'question' && classMode()) {
    if (round.handleKey(k)) { e.preventDefault(); return; }
    if ((k === 'enter' || k === ' ') && S.step === 'done' && !isButton) { e.preventDefault(); closeQuestion(); }
    return;
  }
  if (S.phase === 'question') {
    if (k === 'a') toggleAnswer();
    else if (k === 'c' && S.step === 'ask') markCorrect();
    else if (k === 'w' && S.step === 'ask') markWrong();
    else if (k === 'n') nobody();
    else if (S.step === 'steal' && /^[1-4]$/.test(k)) steal(+k - 1);
    else if ((k === 'enter' || k === ' ') && S.step === 'done' && !isButton) { e.preventDefault(); closeQuestion(); }
  }
});

return {
  enter() { active = true; if (S.phase === 'home') { $('names').innerHTML = ''; renderNames(); renderTopics(); $('startBtn').focus({ preventScroll: true }); } },
  exit() { active = false; if (S.phase !== 'home') goHome(); },
  inProgress: () => S.phase === 'board' || S.phase === 'question',
  _state: () => ({ mode: S.mode, round: round.phase, undoable: round.undoable, times: round.times(), turn: S.turn, catchUp: catchUpTurn(), misconceptions: misc.top(5).map(x => x.q.q), rows: S.rows, phase: S.phase, step: S.step, cursor: Object.assign({}, S.cursor), teams: S.teams.map(t => ({ name: t.name, score: t.score, correct: t.correct })), open: S.open, q: S.open ? tileOpen().q : null, left: S.cats.reduce((n, c) => n + c.tiles.filter(t => !t.used).length, 0) }),
  _board: () => S.cats
};
}

CGB.registerGame('category-clash', {
  title: 'Category Clash',
  init() { game = createCategoryClash(); },
  enter() { if (game) game.enter(); },
  exit() { if (game) game.exit(); },
  inProgress() { return !!(game && game.inProgress()); },
  state() { return game && game._state(); },
  board() { return game && game._board(); }
});
})();
