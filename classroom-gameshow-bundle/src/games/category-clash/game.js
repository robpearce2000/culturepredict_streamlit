'use strict';
/* =========================================================
   CATEGORY CLASH
   A board of four categories (topics) and points values from 100 to 300.
   2 to 6 teams take turns to choose a tile; every team answers it on
   whiteboards. The choosing team wins full points if correct and every
   other correct team half. Harder questions sit lower on the board and
   are worth more; one hidden star tile is worth double.
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
  win() { [523, 659, 784, 1047, 1319].forEach((f, i) => CGB.sfx.tone(f, 0.24, 'triangle', 0.09, i * 0.1)); }
};

/* Fixed settings: four categories of three questions, 100 to 300 points (under 10 minutes
   with a class),
   a 20-second countdown before "show me", and one hidden star tile */
const CATS = 4, ROWS = 3;
const S = {
  order: [], pos: 0, catchUp: -1,
  phase: 'home', nTeams: CGB.store.getJSON('cc.nTeams', 4),
  teams: [], cats: [], turn: 0, cursor: { c: 0, r: 0 },
  open: null, step: null, picked: null
};
if (![2, 3, 4, 5, 6].includes(S.nTeams)) S.nTeams = 4;

/* ---------- Setup: the number of teams, and their names behind "Edit team names" ---------- */
$('logo').innerHTML = CGB.brand.ccLogo();
const defaultNames = [0, 1, 2, 3, 4, 5].map(CGB.teamFallback);
function renderNames() {
  const prev = [0, 1, 2, 3, 4, 5].map(i => { const el = $('name' + i); return el ? el.value : null; });
  $('names').innerHTML = [0, 1, 2, 3, 4, 5].slice(0, S.nTeams).map(i => `<label style="--tc:${TEAM[i].css}"><span>${TEAM[i].mark} Team ${i + 1}</span><input id="cc-name${i}" type="text" maxlength="18" autocomplete="off" value="${esc(prev[i] || CGB.teamNames(6)[i])}"></label>`).join('');
  $('names').classList.toggle('many', S.nTeams > 4);
}
function paintTeams() { $('segTeams').querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.v === S.nTeams))); }
$('segTeams').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  S.nTeams = +b.dataset.v; CGB.store.setJSON('cc.nTeams', S.nTeams);
  paintTeams(); renderNames(); CGB.fitSetups();
});
paintTeams();
renderNames();
function renderPack() { $('startBtn').disabled = !CGB.renderPackLine($('pack')); }
bank.onChange(() => { if (S.phase === 'home') { renderPack(); renderTopics(); } });
renderPack();

/* Topics in the active set, each a candidate category (columns are topics within the subject
   chosen on the main screen). A topic is "full" when it has a question at every tier. */
function topicsOf(set) {
  const m = new Map();
  // one built-in topic chosen: its specification subtopics are the columns, so a one-topic lesson still has a board
  const by = set.groupBy === 'subtopic' ? 'subtopic' : 'topic';
  set.questions.forEach(q => { const k = q[by] || q.topic; if (!m.has(k)) m.set(k, { topic: k, subject: q.subject, qs: [] }); m.get(k).qs.push(q); });
  return Array.from(m.values()).map(t => {
    t.tiers = [1, 2, 3].map(n => t.qs.filter(q => bank.difficultyOf(q) === n).length);
    t.full = t.tiers.every(n => n > 0);
    return t;
  });
}

/* ---------- Choosing the topics: up to four, remembered for each question set. Any not chosen
   (all four by default) are picked for the teacher each game, from topics with a question at every tier ---------- */
const chosenStore = () => CGB.store.getJSON('cc.topics', {});
function chosenTopics(set, all) {
  const names = new Set(all.map(t => t.topic));
  return (chosenStore()[set.id] || []).filter(n => names.has(n)).slice(0, CATS);
}
function saveChosen(set, list) { const m = chosenStore(); if (list.length) m[set.id] = list; else delete m[set.id]; CGB.store.setJSON('cc.topics', m); }
function renderTopics() {
  const set = bank.active();
  if (!set) { $('topicPick').hidden = true; return; }
  const all = topicsOf(set), chosen = chosenTopics(set, all);
  $('topicPick').hidden = false;
  const left = CATS - chosen.length;
  $('topicSum').innerHTML = chosen.length
    ? `Topics: <b>${chosen.map(esc).join(', ')}</b>${left ? ` <small>+ ${left} picked for you</small>` : ''}`
    : `Topics: <b>picked for you each game</b> <small>Choose topics</small>`;
  const multi = new Set(all.map(t => t.subject)).size > 1;
  $('topicHint').textContent = all.length <= CATS
    ? `This set has ${all.length} topic${all.length === 1 ? '' : 's'}, so ${all.length === 1 ? 'it is' : 'they are all'} on the board.`
    : `Choose up to ${CATS} topics for the columns. Any you leave are picked for you, a different mix each game.`;
  const chip = t => {
    const on = chosen.includes(t.topic);
    const note = t.full ? '' : ' <small>(few questions)</small>';
    return `<button type="button" class="chip" data-topic="${esc(t.topic)}" aria-pressed="${on}"${!on && chosen.length >= CATS ? ' disabled' : ''} title="${t.tiers.join(' / ')} questions at tiers 1 / 2 / 3">${esc(t.topic)}${note}</button>`;
  };
  const subjects = Array.from(new Set(all.map(t => t.subject)));
  $('topicChips').classList.toggle('grouped', multi);
  $('topicChips').innerHTML = `<button type="button" class="chip any" data-any aria-pressed="${!chosen.length}">Pick for me</button>` + (multi
    ? subjects.map(sj => `<div class="cc-tgroup"><span class="cc-tsub">${esc(sj)}</span>${all.filter(t => t.subject === sj).map(chip).join('')}</div>`).join('')
    : all.map(chip).join(''));
}
$('topicChips').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b || b.disabled) return;
  const set = bank.active(); if (!set) return;
  let list = chosenTopics(set, topicsOf(set));
  if (b.hasAttribute('data-any')) list = [];
  else { const n = b.dataset.topic; list = list.includes(n) ? list.filter(x => x !== n) : list.concat(n).slice(0, CATS); }
  saveChosen(set, list);
  const key = b.hasAttribute('data-any') ? '[data-any]' : `[data-topic="${CSS.escape(b.dataset.topic)}"]`;
  renderTopics();
  const again = $('topicChips').querySelector(key); if (again) again.focus();
});
$('topicPick').addEventListener('toggle', () => CGB.fitSetups());
renderTopics();

/* Four categories: the teacher's chosen topics, then the rest shuffled each game, full topics
   first, taking topics from each subject in turn so a mixed set gives a mixed board */
function pickTopics(all) {
  const set = bank.active(), chosen = set ? chosenTopics(set, all) : [];
  const fixed = chosen.map(n => all.find(t => t.topic === n));
  const others = all.filter(t => !chosen.includes(t.topic));
  const full = others.filter(t => t.full), rest = others.filter(t => !t.full);
  const mix = a => a.map(v => [CGB.random(), v]).sort((x, y) => x[0] - y[0]).map(v => v[1]);
  const spread = list => {
    const bySub = new Map(); list.forEach(t => { if (!bySub.has(t.subject)) bySub.set(t.subject, []); bySub.get(t.subject).push(t); });
    const lanes = Array.from(bySub.values()), out = [];
    for (let i = 0; out.length < list.length; i++) lanes.forEach(l => { if (l[i]) out.push(l[i]); });
    return out;
  };
  return fixed.concat(spread(mix(full)), mix(rest)).slice(0, CATS);
}

/* ---------- Building the board ---------- */
function buildBoard() {
  const weak = new Set(S.teams.flatMap(t => bank.wrongLog(t.name).map(e => e.q)));
  S.cats = pickTopics(topicsOf(bank.active())).map(t => {
    // Row k (100, 200, 300 points) takes a question at tier 1, 2 or 3 (see TIERS in bank.js;
    // questions without a Tier line get one from their wording). If a row has no question at its
    // tier, the nearest tier is used, then any question left. Among the questions that fit a
    // row, one a playing team got wrong before is preferred, so weak spots come round again.
    const levelOf = q => bank.difficultyOf(q);
    const values = VALUES.slice(0, ROWS);
    const want = k => k + 1;
    const left = t.qs.slice();
    const rnd = a => a[Math.floor(CGB.random() * a.length)];
    const take = (k, exact) => {
      if (!left.length) return null;
      const gap = exact ? 0 : Math.min(...left.map(q => Math.abs(levelOf(q) - want(k))));
      const pool = left.filter(q => Math.abs(levelOf(q) - want(k)) === gap);
      if (!pool.length) return null;
      const missed = pool.filter(q => weak.has(q.q));
      const q = rnd(missed.length ? missed : pool);
      left.splice(left.indexOf(q), 1);
      return q;
    };
    // exact levels first, so a short row never uses up another row's question
    const picks = values.map((v, k) => take(k, true));
    values.forEach((v, k) => { if (!picks[k]) picks[k] = take(k, false); });
    return { topic: t.topic, subject: t.subject, tiles: picks.map((q, k) => ({ value: values[k], q, used: !q, wonBy: -1, star: false, empty: !q, right: 0 })) };
  });
  const live = S.cats.flatMap(c => c.tiles.filter(t => !t.empty && t.value >= 200));
  if (live.length) live[Math.floor(CGB.random() * live.length)].star = true;
  S.cursor = { c: 0, r: 0 };
}

/* ---------- Rendering ---------- */
function renderBoard() {
  const b = $('board');
  b.style.gridTemplateColumns = `repeat(${S.cats.length}, minmax(0, 1fr))`;
  let html = S.cats.map(c => `<div class="cc-cat" role="columnheader"><b>${esc(c.topic)}</b><small>${esc(c.subject)}</small></div>`).join('');
  b.style.gridTemplateRows = `minmax(56px, auto) repeat(${ROWS}, minmax(0, 1fr))`;
  for (let r = 0; r < ROWS; r++) {
    html += S.cats.map((c, ci) => {
      const t = c.tiles[r];
      const cur = S.cursor.c === ci && S.cursor.r === r ? ' cursor' : '';
      if (t.empty) return `<button type="button" class="cc-tile used empty${cur}" disabled aria-label="${esc(c.topic)} ${t.value}: no question">—</button>`;
      if (t.used) {
        const lead = t.wonBy >= 0 ? `<span class="mk">${TEAM[t.wonBy].mark}</span>` : `<span class="mk">${t.right ? '✓' : '✗'}</span>`;
        return `<button type="button" class="cc-tile used${t.wonBy >= 0 ? ' won' : ''}${cur}" disabled style="--tc:${t.wonBy >= 0 ? TEAM[t.wonBy].light : 'inherit'}" aria-label="${esc(c.topic)} ${t.value}: ${t.right} of ${S.teams.length} teams correct">${lead}${t.right} of ${S.teams.length} correct</button>`;
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
  $('turn').innerHTML = t ? `<b style="--turn:${TEAM[S.turn].light}">${TEAM[S.turn].mark} ${esc(t.name)}</b>, pick a category<small>${catchUpTurn() ? 'Catch-up pick! ' : ''}Captain: swap to the next person</small>` : '';
}
$('board').addEventListener('click', e => {
  const b = e.target.closest('.cc-tile[data-c]'); if (!b) return;
  S.cursor = { c: +b.dataset.c, r: +b.dataset.r };
  openTile(S.cursor.c, S.cursor.r);
});
function moveCursor(dc, dr) {
  const nc = S.cats.length;
  S.cursor.c = (S.cursor.c + dc + nc) % nc;
  S.cursor.r = Math.max(0, Math.min(ROWS - 1, S.cursor.r + dr));
  renderBoard();
  const el = $('board').querySelector('.cursor'); if (el && !el.disabled) el.focus({ preventScroll: true });
}

/* ---------- A question ---------- */
function openTile(c, r) {
  const t = S.cats[c] && S.cats[c].tiles[r];
  if (!t || t.used || S.phase !== 'board') return;
  S.phase = 'question'; S.open = { c, r }; S.step = 'ask'; S.picked = S.turn;
  const cat = S.cats[c];
  $('qCat').textContent = cat.subject + ' · ' + cat.topic;
  $('qVal').textContent = (t.star ? t.value * 2 : t.value) + ' points';
  $('qStar').hidden = !t.star;
  $('qFor').innerHTML = `Chosen by <b>${TEAM[S.turn].mark} ${esc(S.teams[S.turn].name)}</b> · every team answers: full points for ${esc(S.teams[S.turn].name)}, half for the others`;
  $('qText').textContent = t.q.q;
  $('qAnswer').innerHTML = CGB.answerHTML(t.q);
  $('qAnswer').classList.remove('shown');
  $('qMsg').textContent = '';
  paintQBoard();
  const tileEl = $('board').querySelector(`.cc-tile[data-c="${c}"][data-r="${r}"]`);
  if (t.star) { SFX.star(); hostC.say('A star tile! This one is worth double.', 'cheer', 1600); starReveal(tileEl); return; }
  SFX.open();
  showQuestionCard(tileEl);
}
/* ---------- 3D moments (presentation only) ----------
   A picked tile flips over and the question card grows out of it while the board pushes in a
   little; the ★ tile gets a moment of its own first. With reduced motion everything fades. */
const playEl = $('play');
function showQuestionCard(tileEl) {
  S.step = 'ask';
  $('q').hidden = false;
  playEl.classList.add('pushed');
  const card = $('qcard');
  if (tileEl && !CGB.settings.reduced()) {
    tileEl.classList.add('flipping');
    // the card starts where the tile is and grows to its place (it is measured, then animated)
    const a = tileEl.getBoundingClientRect(), b = card.getBoundingClientRect();
    if (a.width && b.width) {
      const sx = a.width / b.width, sy = a.height / b.height;
      card.style.transition = 'none';
      card.style.transform = `translate(${a.left + a.width / 2 - (b.left + b.width / 2)}px, ${a.top + a.height / 2 - (b.top + b.height / 2)}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)}) rotateY(90deg)`;
      card.getBoundingClientRect();
      card.style.transition = 'transform 0.42s cubic-bezier(.2,.8,.25,1)';
      card.style.transform = '';
    }
  }
  round.think();
  card.focus({ preventScroll: true });   // keys now go to the question, not the board behind it
}
let revealT = 0, revealTile = null;
function starReveal(tileEl) {
  S.step = 'reveal';
  if (CGB.settings.reduced() || !tileEl) { showQuestionCard(tileEl); return; }
  revealTile = tileEl;
  playEl.classList.add('dim', 'pushed');
  tileEl.classList.add('star-spin');
  revealT = setTimeout(endStarReveal, 1500);
}
// Space or Enter skips the ★ moment
function endStarReveal() {
  if (S.step !== 'reveal') return;
  clearTimeout(revealT);
  playEl.classList.remove('dim');
  if (revealTile) revealTile.classList.remove('star-spin');
  showQuestionCard(revealTile);
  revealTile = null;
}
/* The host in the corner: short captions at the big moments */
const hostC = CGB.createHostCorner($('host'));
const sumHost = CGB.createHostCorner(document.createElement('div'), { className: 'host-sum' });
/* ---------- Whole class: every team answers, the teacher marks each team ---------- */
const misc = CGB.createMisconceptions();
const qBoard = CGB.createTeamBoard($('qboard'));
let undoSnap = null;
const round = CGB.createClassRound({
  root: document.getElementById('game-category-clash'),
  board: qBoard, countEl: $('count'), btnEl: $('qBtns'),
  seconds: () => CGB.COUNTDOWN, teams: () => S.teams.length,
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
  $('qAnswer').classList.remove('shown');
  $('qMsg').textContent = 'Marking undone. Mark each team again.';
  S.step = 'ask'; undoSnap = null;
  paintQBoard();
  renderTeams();
}
/* Each team chooses once per round. The last-placed team, if it trails the leader by at least
   the value of the top tile, chooses first in the next round: its catch-up pick. */
function startClassRound() {
  const n = S.teams.length, top = ROWS * 100;
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
const catchUpTurn = () => S.pos === 0 && S.catchUp >= 0 && S.turn === S.catchUp;
function tileOpen() { return S.cats[S.open.c].tiles[S.open.r]; }
function showAnswer() { $('qAnswer').classList.add('shown'); }
function boardDone() { return S.cats.every(c => c.tiles.every(t => t.used || (S.open && c === S.cats[S.open.c] && t === tileOpen()))); }
function closeQuestion() {
  if (S.step !== 'done') return;
  tileOpen().used = true;
  const justC = S.open.c, justR = S.open.r, winners = S.teams.map((x, i) => (undoSnap && x.score > undoSnap.scores[i]) ? i : -1).filter(i => i >= 0);
  $('q').hidden = true;
  playEl.classList.remove('pushed');
  S.phase = 'board'; S.open = null;
  round.stop(); nextClassTurn();
  if (S.cats.every(c => c.tiles.every(t => t.used))) { showSummary(); return; }
  // move the cursor to the next free tile
  const free = []; S.cats.forEach((c, ci) => c.tiles.forEach((t, r) => { if (!t.used) free.push({ c: ci, r }); }));
  if (!free.some(f => f.c === S.cursor.c && f.r === S.cursor.r)) S.cursor = free[0];
  renderBoard();
  // the answered tile flips back showing the winning team's colour; winners' panels catch the light
  const done = $('board').querySelectorAll('.cc-tile')[justR * S.cats.length + justC];
  if (done) done.classList.add('flip-in');
  const panels = $('teams').querySelectorAll('.cc-team');
  winners.forEach(i => { if (panels[i]) panels[i].classList.add('sweep'); });
  const el = $('board').querySelector('.cursor'); if (el) el.focus({ preventScroll: true });
}

/* ---------- Game flow ---------- */
function startGame() {
  if (!bank.active()) { renderPack(); return; }
  CGB.leaveField();
  const names = [0, 1, 2, 3, 4, 5].slice(0, S.nTeams).map(i => (($('name' + i) || {}).value || '').trim().slice(0, 18) || defaultNames[i]);
  CGB.saveTeamNames(names);
  S.teams = names.map(name => ({ name, score: 0, correct: 0, wrong: [] }));
  S.order = []; S.catchUp = -1; misc.reset(); round.stop();
  startClassRound(); S.turn = S.order[0];
  buildBoard();
  S.phase = 'ready';
  $('home').classList.remove('active'); $('summary').classList.remove('active');
  $('play').hidden = false; $('q').hidden = true;
  renderBoard();
  // the board is on show, with nothing to pick until the teacher presses Start game
  gate.show(() => {
    S.phase = 'board';
    renderBoard();
    hostC.say(`Welcome to Category Clash! ${S.teams[0].name}, you pick first.`, 'wave', 2000);
    const el = $('board').querySelector('.cursor'); if (el) el.focus({ preventScroll: true });
  });
}
const gate = CGB.createStartGate(document.getElementById('game-category-clash'), 'Teams take turns to pick a tile, and every team answers on a whiteboard. Nothing starts until you press Start.');
$('startBtn').addEventListener('click', startGame);
$('endBtn').addEventListener('click', () => CGB.armButton($('endBtn'), 'Tap again to end', () => { round.stop(); $('q').hidden = true; showSummary(); }));

function showSummary() {
  S.phase = 'summary';
  round.stop();
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
  $('sumCard').innerHTML = `<h2>${winners.length > 1 ? "It's a draw!" : esc(winners[0].t.name) + ' win!'}</h2>
    <div class="cc-podium">${podium}</div>${misc.html(5)}
    <div class="cc-sum-btns"><button class="btn go" type="button" id="cc-again">Play again <span class="kbd">Enter</span></button><button class="btn plain" type="button" id="cc-change">Change teams</button><button class="btn plain" type="button" id="cc-menu2">Back to menu</button></div>${CGB.REVIEW_NOTE}`;
  $('sumCard').prepend(sumHost.el);
  sumHost.say(winners.length > 1 ? "A draw! What a close game. Well done, everyone." : `${winners[0].t.name} win with ${top} points. Great game, everyone!`, 'cheer', 2200);
  hostC.quiet();
  $('summary').classList.add('active');
  $('summary').scrollTop = 0;
  $('again').onclick = startGame;
  $('change').onclick = goHome;
  $('menu2').onclick = () => CGB.app.requestLauncher();
  $('again').focus({ preventScroll: true });
}
function goHome() {
  gate.hide();
  clearTimeout(revealT); playEl.classList.remove('dim', 'pushed');
  if (CGB.fitSetups) CGB.fitSetups();
  round.stop();
  S.phase = 'home'; S.open = null;
  $('q').hidden = true; $('play').hidden = true; $('summary').classList.remove('active');
  $('home').classList.add('active');
  renderPack();
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
  if (S.phase === 'home') { if (k === 'enter' && !(e.target.matches && e.target.matches('button, select, textarea, summary'))) { e.preventDefault(); startGame(); } return; }
  if (S.phase === 'summary') { if (k === 'enter' && !isButton) { e.preventDefault(); startGame(); } return; }
  if (inField) return;
  if (S.phase === 'board') {
    const mv = { arrowleft: [-1, 0], arrowright: [1, 0], arrowup: [0, -1], arrowdown: [0, 1] }[k];
    if (mv) { e.preventDefault(); moveCursor(mv[0], mv[1]); return; }
    if ((k === 'enter' || k === ' ') && !isButton) { e.preventDefault(); openTile(S.cursor.c, S.cursor.r); }
    return;
  }
  if (S.phase === 'question') {
    if (S.step === 'reveal') { if (k === 'enter' || k === ' ') { e.preventDefault(); endStarReveal(); } return; }
    if (round.handleKey(k)) { e.preventDefault(); return; }
    if ((k === 'enter' || k === ' ') && S.step === 'done' && !isButton) { e.preventDefault(); closeQuestion(); }
  }
});

return {
  enter() { active = true; if (S.phase === 'home') { $('names').innerHTML = ''; renderNames(); renderPack(); $('startBtn').focus({ preventScroll: true }); } },
  exit() { active = false; if (S.phase !== 'home') goHome(); },
  inProgress: () => S.phase === 'board' || S.phase === 'question',
  _state: () => ({ round: round.phase, undoable: round.undoable, times: round.times(), turn: S.turn, catchUp: catchUpTurn(), misconceptions: misc.top(5).map(x => x.q.q), rows: ROWS, phase: S.phase, step: S.step, cursor: Object.assign({}, S.cursor), teams: S.teams.map(t => ({ name: t.name, score: t.score, correct: t.correct })), open: S.open, q: S.open ? tileOpen().q : null, left: S.cats.reduce((n, c) => n + c.tiles.filter(t => !t.used).length, 0) }),
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
