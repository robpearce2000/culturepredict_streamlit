/* Playtest: one simulated lesson (or several games in a row) on the shipped file, at real speed.
   The teacher only uses visible buttons and the keys in the teacher guide; the read-only
   CGB.games[id].state() is used to know which screen is showing and to log what happened.
   usage: node tools/playtest/lesson.js '<lesson json>' out.json */
const { chromium } = require('@playwright/test');
const path = require('path'), fs = require('fs');
const DIST = 'file://' + path.join(__dirname, '..', '..', 'dist', 'showtime-classroom-gameshows.html');
const L = JSON.parse(process.argv[2]), OUT = process.argv[3];
const SHOTS = path.join(__dirname, '..', '..', 'playtest-screenshots');
fs.mkdirSync(SHOTS, { recursive: true });

// a seeded random source so a lesson can be replayed
let seed = L.seed || 1;
const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const pick = a => a[Math.floor(rnd() * a.length)];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const T0 = Date.now(), now = () => (Date.now() - T0) / 1000;
const log = { lesson: L, events: [], games: [], errors: [], fps: [] };
const ev = (type, data) => { log.events.push(Object.assign({ t: +now().toFixed(2), type }, data || {})); };

/* ---------- the class: each team's chance of a right answer ---------- */
function abilities(n) {
  if (L.abilities) return L.abilities.slice(0, n);
  // one strong team (~85%), one weak (~40%), the rest in between
  const a = [0.85, 0.4, 0.7, 0.55, 0.62, 0.48];
  return a.slice(0, n);
}
function answers(n, q, qi) {
  const ab = abilities(n), d = (q && q.difficulty) || 2;
  if (L.mode === 'allRight') return ab.map(() => true);
  if (L.mode === 'allWrong') return ab.map(() => false);
  if (L.mode === 'tie') return ab.map((x, i) => i < 2 ? qi % 2 === 0 : rnd() < 0.5);   // two teams always level
  const shared = (rnd() - 0.5) * 0.3;                 // some questions trip the whole class up
  return ab.map(p => rnd() < Math.max(0.05, Math.min(0.97, p - 0.12 * (d - 2) + shared)));
}

(async () => {
  const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
  const ctx = L.profile
    ? await chromium.launchPersistentContext(L.profile, { viewport: { width: L.w || 1366, height: L.h || 768 } })
    : await browser.newContext({ viewport: { width: L.w || 1366, height: L.h || 768 } });
  const page = ctx.pages()[0] || await ctx.newPage();
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') log.errors.push(now().toFixed(1) + ' ' + m.type() + ': ' + m.text()); });
  page.on('pageerror', e => log.errors.push(now().toFixed(1) + ' pageerror: ' + e.message));
  await page.addInitScript(([q]) => {
    try { const k = 'cgb.settings', s = JSON.parse(localStorage.getItem(k) || '{}'); s.quality = q; localStorage.setItem(k, JSON.stringify(s)); } catch (e) {}
    // frame timing, read at the end
    window.__frames = []; const f = t => { window.__frames.push(t); if (window.__frames.length > 200000) window.__frames.splice(0, 100000); requestAnimationFrame(f); }; requestAnimationFrame(f);
  }, [L.quality || 'low']);
  await page.goto(DIST);
  await page.waitForFunction(() => window.CGB && CGB.app);
  await sleep(2500);                                   // the teacher looks at the main screen

  // subject, board, packs
  await page.selectOption('#subjectSelect', L.subject); await sleep(800);
  if (L.board) { await page.selectOption('#boardSelect', L.board).catch(() => ev('note', { msg: 'board not offered for ' + L.subject })); await sleep(800); }
  if (L.pack != null) {
    await page.click('#packBtn'); await sleep(1200);
    const boxes = page.locator('#packMenu input[data-id]');
    const n = await boxes.count();
    const idx = typeof L.pack === 'number' ? L.pack % n : 0;
    await boxes.nth(idx).check(); await sleep(600);
    await page.click('#packMenu [data-done]'); await sleep(600);
  }
  ev('packs', { label: await page.textContent('#packBtn') });

  for (const g of L.games) {
    const res = await playGame(page, g);
    log.games.push(res);
    if (L.refresh && g.id === L.games[0].id) break;
    await sleep(3000);
  }
  log.fps = await page.evaluate(() => { const f = window.__frames, out = []; let i0 = 0; for (let i = 1; i < f.length; i++) { if (f[i] - f[i0] >= 5000) { out.push(+((i - i0) * 1000 / (f[i] - f[i0])).toFixed(1)); i0 = i; } } return out; }).catch(() => []);
  log.total = now();
  fs.writeFileSync(OUT, JSON.stringify(log, null, 1));
  await ctx.close(); await browser.close();
})().catch(e => { log.crash = String(e && e.stack || e); fs.writeFileSync(OUT, JSON.stringify(log, null, 1)); process.exit(1); });

/* ---------- one game ---------- */
const P = { 'over-the-edge': 'ote', outpace: 'op', 'category-clash': 'cc', 'hex-hunt': 'hh' };
async function st(page, id) { return page.evaluate(g => CGB.games[g].state(), id); }
async function shot(page, name) { try { await page.screenshot({ path: path.join(SHOTS, name + '.png') }); } catch (e) {} }
async function key(page, k, why) { await page.keyboard.press(k); ev('key', { k, why }); }

async function playGame(page, G) {
  const id = G.id, p = P[id], n = G.teams || 2;
  const R = { game: id, teams: n, questions: [], waits: [], moments: [], scores: [], qids: [] };
  await page.click(`[data-play="${id}"]`); await sleep(2000);
  if (id !== 'hex-hunt') { await page.click(`#${p}-seg${id === 'outpace' ? 'Groups' : 'Teams'} button[data-v="${n}"]`); await sleep(700); }
  if (L.subject === 'maths' && G.mathsTime) {
    const lbl = { 45: '45 sec', 90: '1 min 30', 120: '2 min' }[G.mathsTime];
    await page.locator(`#game-${id} .answer-time button`, { hasText: lbl }).click(); await sleep(600);
  }
  await sleep(1500);
  if (G.shotSetup) await shot(page, G.shotSetup);
  await page.click(`#${p}-startBtn`); ev('start', { game: id }); await sleep(1500);
  // the Start game card
  if (await page.locator(`#game-${id} .start-gate:not([hidden])`).count()) { await sleep(2500); await key(page, 'Enter', 'start game'); }
  const tStart = now(); R.tStart = tStart;
  let qi = 0, lastQ = null, deadline = Date.now() + (G.maxMin || 40) * 60000, lastScoreQ = -1;
  const moments = G.moments || {};
  while (Date.now() < deadline) {
    const s = await st(page, id);
    if (!s) { await sleep(300); continue; }
    if (s.phase === 'summary') break;
    // the game went back to its Start card by itself: a key meant for the last question landed on the results
    if (R.questions.length && (s.phase === 'ready' || s.phase === 'home')) { R.moments.push({ what: 'game restarted: the last key press landed on the results screen (Enter = Play again)', afterQ: R.questions.length }); R.restartedAt = now(); break; }
    if (G.endEarlyAt && now() - tStart > G.endEarlyAt * 60) { R.moments.push(await endEarly(page, id, p)); break; }
    if (L.refresh && qi === 2 && !R.refreshed) { R.refreshed = true; R.moments.push(await refreshMid(page, id)); return R; }
    const qa = await questionStep(page, id, p, s, R, qi, moments);
    if (qa === 'asked') { qi++; continue; }
    if (qa === 'acted') continue;
    await sleep(250);
  }
  R.tEnd = R.restartedAt || now(); R.total = +(R.tEnd - tStart).toFixed(1);
  const s = await st(page, id);
  R.final = summarise(id, s);
  await sleep(4000);
  if (G.shotSummary) await shot(page, G.shotSummary);
  // back to the main screen
  await key(page, 'Escape', 'menu'); await sleep(800);
  if (await page.locator('#leaveModal:not([hidden])').count()) await page.click('#leaveConfirm');
  await sleep(1500);
  return R;
}

function summarise(id, s) {
  if (!s) return null;
  if (id === 'over-the-edge') return { money: s.money, team: s.team, phase: s.phase, correct: s.correct };
  if (id === 'category-clash') return { scores: s.teams.map(t => t.score), correct: s.teams.map(t => t.correct), left: s.left };
  if (id === 'hex-hunt') return { winner: s.winner, owners: s.owners };
  if (id === 'outpace') return { pot: s.pot, net: s.net, target: s.target, runner: s.runner, hunter: s.hunter, phase: s.phase };
}

/* Waits until the game offers the next thing the teacher can do; returns the wait in seconds */
async function waitFor(page, id, test, max) {
  const t = now(), end = Date.now() + (max || 60000);
  while (Date.now() < end) { const s = await st(page, id); if (s && test(s)) return { s, wait: now() - t }; await sleep(150); }
  return { s: await st(page, id), wait: now() - t, timedOut: true };
}

/* One teacher action from the current screen. Returns 'asked' when a whole question was played */
async function questionStep(page, id, p, s, R, qi, M) {
  const n = R.teams;
  // -------- choosing what comes next (board picks, lanes, deals, round ends) --------
  if (id === 'outpace') {
    if (s.roundEnd) { await sleep(4000); await key(page, 'Enter', 'round end'); return 'acted'; }
    if (s.phase === 'deal' && !s.dealReward) { await sleep(6000); await key(page, pick(['2', '2', '1', '3']), 'deal vote'); return 'acted'; }
    if (s.phase === 'deal' && s.round === 'done') { const w = await waitAndNext(page, id, R, s); return w; }
    if (s.phase === 'sprint' && await page.locator('#game-outpace .sprint-gate:not([hidden])').count()) { await sleep(8000); if (L.shotSprintGate) await shot(page, L.shotSprintGate); await key(page, 'Enter', 'start sprint'); return 'acted'; }
  }
  if (id === 'category-clash' && s.phase === 'board') {
    await sleep(3500);                                   // the captain decides
    const free = await page.$$eval('#cc-board .cc-tile[data-c]:not(.used):not(.empty)', els => els.filter(e => !e.disabled).map(e => [e.dataset.c, e.dataset.r]));
    if (!free.length) return 'none';
    const want = rnd() < 0.5 ? free.filter(f => f[1] === String(Math.max(...free.map(x => +x[1])))) : free;   // half the time they go for points
    const [c, r] = pick(want.length ? want : free);
    await page.click(`#cc-board .cc-tile[data-c="${c}"][data-r="${r}"]`); ev('pick', { c, r });
    return 'acted';
  }
  if (id === 'category-clash' && s.phase === 'question' && s.step === 'reveal') { await sleep(500); return 'acted'; }
  if (id === 'hex-hunt') {
    if (s.phase === 'board') {
      await sleep(3500);
      const i = hexChoice(s);
      await page.locator('#hh-board .hh-hex').nth(i).click(); ev('pick', { hex: i });
      return 'acted';
    }
    if (s.phase === 'celebrate') { const w = await waitFor(page, id, x => x.phase !== 'celebrate', 20000); R.waits.push({ what: 'winning chain', s: +w.wait.toFixed(1) }); return 'acted'; }
    if (s.phase === 'won') { await sleep(3000); await key(page, 'Enter', 'see results'); return 'acted'; }
  }
  if (id === 'over-the-edge') {
    const cur = R.ote || (R.ote = {});
    if (s.step === 'lanes' || (s.step === 'dropping' && s.phase !== 'final' && await page.locator('#ote-actions .ote-chute-btn').count())) {
      const t0 = now();
      if (s.phase === 'final') { await sleep(3000); await key(page, 'r', 'random lanes (final)'); }
      else { await sleep(2500); await key(page, String(1 + Math.floor(rnd() * 4)), 'captain picks lane'); }
      cur.human = (cur.human || 0) + (now() - t0);
      return 'acted';
    }
    if (s.step === 'dropping') { await sleep(200); return 'acted'; }
    if (s.step === 'next') {
      // the drop for this question is over: real time and the game's own clock, from marking to settled
      if (s.dropDoneAt > s.markAt && s.dropDoneReal) {
        R.waits.push({ what: 'marking to counters settled', q: qi, sim: +(s.dropDoneAt - s.markAt).toFixed(1), real: +((s.dropDoneReal - s.markReal) / 1000).toFixed(1), humanPicks: +(cur.human || 0).toFixed(1), phase: s.phase });
      }
      R.ote = {};
      await sleep(2500); await key(page, 'Space', 'next question'); return 'acted';
    }
    if (s.step === 'ready') { await sleep(500); return 'acted'; }
  }

  // -------- a question: think, show me, mark, confirm --------
  if (s.round === 'think') {
    const q = s.q || {};
    const t = { qi, id: q.id || null, diff: q.difficulty, len: (q.q || '').length, aLen: (q.a || '').length, tThink: now() };
    R.qids.push(q.id || q.q);
    const secs = await page.evaluate(() => CGB.answerSeconds());
    // reading the question aloud, then writing
    let writeFor = secs;
    if (L.mode !== 'allRight' && rnd() < 0.35) writeFor = Math.max(8, secs * (0.55 + rnd() * 0.3));   // every board up early: Space
    if (M.repeatAt === qi) { R.moments.push({ what: 'a team asks for the question again', q: qi, note: 'teacher reads it again; the countdown keeps running' }); }
    if (M.escAt === qi) {
      await sleep(4000);
      const before = (await st(page, id)).round;
      await key(page, 'Escape', 'accidental Esc');
      await sleep(1500);
      const modal = await page.locator('#leaveModal:not([hidden])').count();
      if (modal) await shot(page, `${L.name}-accidental-esc`);
      const s1 = await st(page, id);
      await sleep(2500);
      const s2 = await st(page, id);
      if (modal) { await page.click('#leaveModal [data-close="leaveModal"]'); }
      R.moments.push({ what: 'Esc pressed by accident mid-question', q: qi, modal: !!modal, roundBefore: before, roundDuringModal: s2.round, countdownKeptRunning: s1.round === 'think' ? 'unknown' : s1.round });
      await sleep(1000);
    }
    if (M.pauseAt === qi) {
      ev('pause', { secs: M.pauseSecs || 120 });
      await sleep(3000);
      const usesP = await page.locator(`#game-${id} [data-pause]`).count();
      if (usesP) await key(page, 'p', 'pause for the interruption');
      const before = await st(page, id);
      await sleep((M.pauseSecs || 120) * 1000);
      const after = await st(page, id);
      if (usesP) { await shot(page, `${L.name}-paused`); await key(page, 'p', 'carry on'); }
      R.moments.push({ what: `interruption: ${M.pauseSecs || 120} s ${usesP ? 'paused with P' : 'pause'} while the class is writing`, q: qi, roundBefore: before.round, roundAfter: after.round, countdownBefore: before.countdown, countdownAfter: after.countdown });
      if (!usesP) await shot(page, `${L.name}-after-pause`);
    }
    // wait for writing, then Space (or the countdown runs out)
    const wEnd = Date.now() + writeFor * 1000;
    while (Date.now() < wEnd) { const x = await st(page, id); if (x.round !== 'think') break; await sleep(250); }
    const x0 = await st(page, id);
    if (x0.round === 'think') await key(page, 'Space', 'show me');
    t.tShow = now();
    // the "3, 2, 1, show me"
    const m = await waitFor(page, id, x => x.round === 'mark' || x.round === 'done' || x.phase === 'summary', 20000);
    t.tMark = now(); t.showAnim = +(t.tMark - t.tShow).toFixed(1);
    if (M.shotQuestionAt === qi) await shot(page, `${L.name}-marking`);
    // the teacher looks along the whiteboards and marks
    if (id === 'hex-hunt') {
      const res = answers(2, q, qi);
      // each half has ~15 students: count right answers in each half
      const right = res.map((r, h) => { const p = abilities(2)[h]; let c = 0; for (let k = 0; k < 15; k++) if (rnd() < p - 0.1 * ((q.difficulty || 2) - 2)) c++; return L.mode === 'allRight' ? 15 : L.mode === 'allWrong' ? 0 : c; });
      await sleep(4000);
      const k = right[0] === 0 && right[1] === 0 ? 'n' : right[0] > right[1] ? '1' : right[1] > right[0] ? '2' : String(s.picker + 1);
      await key(page, k, 'half with more right');
      t.marks = right;
    } else {
      const res = answers(n, q, qi);
      if (L.mode === 'tie' && id === 'category-clash') res.forEach((v, i) => { if (i >= 2) res[i] = false; });
      await sleep(1200 + 900 * n);                      // looking along the boards
      const mis = M.mismarkAt === qi ? res.findIndex(r => !r) : -1;
      for (let i = 0; i < n; i++) if (res[i] || i === mis) { await key(page, String(i + 1), 'mark'); await sleep(450); }
      await sleep(800);
      await key(page, 'Enter', 'confirm');
      t.correct = res.filter(Boolean).length;
      if (mis >= 0) {
        await sleep(2500);                              // "Sir, we got that wrong!"
        const sb = await st(page, id);
        const can = sb.undoable;
        if (can) {
          await key(page, 'u', 'undo the slip');
          await sleep(800);
          await key(page, String(mis + 1), 'unmark'); await sleep(500);
          await key(page, 'Enter', 'confirm again');
        }
        R.moments.push({ what: 'teacher marked a team right by mistake, then undid it', q: qi, undoAvailable: can, step: sb.step || sb.round });
        if (M.shotUndo) await shot(page, `${L.name}-after-undo`);
      }
    }
    t.tConfirm = now(); t.marking = +(t.tConfirm - t.tMark).toFixed(1); t.writing = +(t.tShow - t.tThink).toFixed(1);
    if (M.rapidAt === qi) {
      // a double press on the next button
      await sleep(1500);
      const before = await st(page, id);
      await page.keyboard.press('Enter'); await page.keyboard.press('Enter'); await page.keyboard.press('Space');
      await sleep(1500);
      const after = await st(page, id);
      R.moments.push({ what: 'rapid Enter, Enter, Space after marking', q: qi, before: [before.phase, before.step || before.round, before.qIndex], after: [after.phase, after.step || after.round, after.qIndex] });
    }
    R.questions.push(t);
    R.scores.push(scoreNow(id, await st(page, id)));
    // reading out the answer, then on
    await sleep(4000);
    return 'asked';
  }
  if (s.round === 'done' || (id !== 'over-the-edge' && (s.step === 'done' || s.step === 'claimed' || s.step === 'nobody' || s.step === 'winning' || s.step === 'stalled'))) {
    return waitAndNext(page, id, R, s);
  }
  if (s.round === 'show') { await sleep(300); return 'acted'; }
  if (s.round === 'mark') { await key(page, 'Enter', 'confirm (stray)'); return 'acted'; }
  return 'none';
}

async function waitAndNext(page, id, R, s) {
  const t = now();
  if (id === 'outpace') {
    if (s.phase === 'sprint') { await waitFor(page, id, x => x.round !== 'done' || x.phase !== 'sprint', 15000); return 'acted'; }   // the sprint moves on by itself
    // the Next question button appears once the runner and the Hunter have moved
    const w = await waitFor(page, id, x => x.roundEnd || x.phase !== 'deal' || x.round !== 'done', 1500);
    if (w.s && w.s.phase === 'deal' && w.s.round === 'done' && !w.s.roundEnd) {
      const btn = page.locator('#op-dealBtnRow button[data-op="next"]');
      const t1 = now();
      await btn.waitFor({ state: 'visible', timeout: 20000 }).catch(() => {});
      R.waits.push({ what: 'runner and Hunter moving', s: +(now() - t1 + 1.5).toFixed(1) });
      await sleep(1000); await key(page, 'Enter', 'next question');
    }
    return 'acted';
  }
  await sleep(1000);
  await key(page, 'Enter', 'back to the board / next');
  return 'acted';
}

function scoreNow(id, s) {
  if (!s) return null;
  if (id === 'over-the-edge') return s.money.slice();
  if (id === 'category-clash') return s.teams.map(t => t.score);
  if (id === 'hex-hunt') return [s.owners.filter(o => o === 0).length, s.owners.filter(o => o === 1).length];
  if (id === 'outpace') return [s.runner, s.hunter, s.net, s.target];
}

/* Hex Hunt captains: extend their own chain towards the far edge, otherwise anywhere free */
function hexChoice(s) {
  const n = s.size, free = s.owners.map((o, i) => o < 0 && s.letters[i][1] ? i : -1).filter(i => i >= 0);
  const me = s.picker, mine = s.owners.map((o, i) => o === me ? i : -1).filter(i => i >= 0);
  const nb = i => { const c = i % n, r = Math.floor(i / n); return [[c - 1, r], [c + 1, r], [c, r - 1], [c, r + 1], [c + (r % 2 ? 1 : -1), r - 1], [c + (r % 2 ? 1 : -1), r + 1]].filter(([a, b]) => a >= 0 && b >= 0 && a < n && b < n).map(([a, b]) => b * n + a); };
  const near = free.filter(i => mine.some(m => nb(m).includes(i)));
  return pick(near.length && rnd() < 0.75 ? near : free);
}

async function endEarly(page, id, p) {
  const t = now();
  const end = page.locator(`#game-${id} [data-endgame]`);
  if (await end.count()) {
    await end.click(); await sleep(900); await end.click();
    await sleep(2500);
    const s = await st(page, id);
    await shot(page, `${L.name}-ended-early`);
    return { what: 'time short: teacher ends the game early', how: 'End game in the top bar (tap twice)', took: +(now() - t).toFixed(1), reachedResults: s.phase === 'summary' };
  }
  await key(page, 'Escape', 'end early via Menu'); await sleep(1500);
  const modal = await page.locator('#leaveModal:not([hidden])').count();
  if (modal) { await shot(page, `${L.name}-end-early-modal`); await page.click('#leaveConfirm'); }
  await sleep(1500);
  return { what: 'time short: teacher ends the game early', how: 'Menu (Esc), then Back to menu', took: +(now() - t).toFixed(1), reachedResults: false, note: modal ? 'leave prompt: scores lost, no results screen' : 'went straight to the menu' };
}

async function refreshMid(page, id) {
  const before = await st(page, id);
  await page.reload(); await sleep(4000);
  const where = await page.evaluate(() => CGB.app.current);
  const s = await page.evaluate(g => CGB.games[g] && CGB.games[g].state(), id).catch(() => null);
  await shot(page, `${L.name}-after-refresh`);
  return { what: 'page refreshed mid-game', before: [before.phase, before.step || before.round], after: where, phaseAfter: s && s.phase };
}
