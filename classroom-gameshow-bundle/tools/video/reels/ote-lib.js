/*
 * Over the Edge driver for the reels (recording only). The page runs on Playwright's fake clock and the game's own
 * loop steps the physics (one 1/60 s step per frame): a "take" (seed + the same actions at the same frames) replays
 * exactly, whether it is being searched (tiny window, no pictures) or rendered frame by frame.
 *   prelude: start a 6-team game and play questions (every team right, random lanes); after each drop wave
 *   the shelf keeps pushing for IDLE frames before the next question. Every physics frame is numbered.
 */
'use strict';
const D = 400, R = 18, JR = 34;
const IDLE = 300;
/* the recording stage: only the 3D machine, filling the tall frame (used when searching too, so takes match) */
const STAGE_CSS = `
#game-over-the-edge .ote-panel, #game-over-the-edge .setup, #game-over-the-edge .host-bar, #game-over-the-edge .ote-order { display: none !important; }
#game-over-the-edge #ote-stageWrap { position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important; z-index: 5; }
#game-over-the-edge .ote-host, #game-over-the-edge .ote-bubble, #game-over-the-edge .ote-labels, #game-over-the-edge .flabel, #game-over-the-edge .ote-banner,
#game-over-the-edge .ote-lightnote, #game-over-the-edge .ote-overlay, #game-over-the-edge .cgb-banner, .flabel { display: none !important; }
`;
   // frames of pushing after each drop wave

/* time: real (searching) or the page's fake clock (rendering); the same actions happen either way */
function timer(page, fake) {
  const wait = fake ? ms => page.clock.runFor(ms) : ms => page.waitForTimeout(ms);
  const until = async (fn, ms) => { for (let t = 0; t < (ms || 20000); t += 50) { if (await page.evaluate(fn)) return true; await wait(50); } return false; };
  return { wait, until };
}
async function setup(page, T) {
  await page.evaluate(() => { window.__reelSkip = true; });
  await page.evaluate(() => { CGB.bank.setBoard('aqa'); CGB.bank.setSubject('biology'); });
  await page.evaluate(() => CGB.app.show('over-the-edge'));
  // DOM clicks: the recording stage hides the setup card. The physics then steps only when this script says
  // (the game's test-only manual clock): exactly one 1/60 s step per frame, whatever the browser's frame timing
  await page.evaluate(() => { CGB.test.ote.manual(true); document.querySelector('#ote-segTeams button[data-v="6"]').click(); document.getElementById('ote-startBtn').click(); CGB.test.ote.manual(true); });
  await T.until(() => CGB.games['over-the-edge'].state().step === 'ask', 60000);
}
const st = page => page.evaluate(() => { const s = CGB.games['over-the-edge'].state(); return { phase: s.phase, step: s.step, round: s.round, q: s.qIndex }; });

/* mark every team right for the question on screen (keys, spaced like a teacher's) */
async function markAll(page, T) {
  for (let k = 0; k < 6; k++) {
    const s = await st(page);
    if (s.round === 'mark') break;
    await page.keyboard.press('Space'); await T.wait(560);
  }
  for (let i = 1; i <= 6; i++) await page.keyboard.press(String(i));
  await page.keyboard.press('Enter');
  await T.until(() => { const s = CGB.games['over-the-edge'].state(); return s.step === 'lanes' || s.step === 'dropping'; });
}

/* measure the tray now: lower-tray count, the counter nearest the edge, the counters in the front band */
const MEASURE = ([D, R, JR]) => {
  const info = CGB.test.ote.info();
  const bodies = info.tray ? info.tray.split(';').map(s => { const [k, xy] = s.split(':'); const [x, y] = xy.split(',').map(Number); return { k, x, y }; }) : [];
  let best = null;
  for (const b of bodies) { const r = b.k === 'jackpot' ? JR : R; const over = b.y - (D - r * 0.6); if (!best || over > best.over) best = { over, x: b.x, y: b.y, k: b.k }; }
  return { n: bodies.length, best, jf: info.jackpotFell, sim: CGB.test.ote.simTime(), edge: bodies.filter(b => b.y > D - 3 * R).map(b => [b.x, b.y, b.k === 'jackpot' ? 1 : 0]) };
};
/* advance n frames: one physics step and 1/60 s of the page clock each, measuring after each when asked */
async function advance(page, n, measure) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const m = await page.evaluate(([args, measure]) => { CGB.test.ote.advance(1 / 60); return measure ? (0, eval)('(' + args.fn + ')')(args.c) : null; }, [{ fn: MEASURE.toString(), c: [D, R, JR] }, measure !== false]);
    await page.clock.runFor(1000 / 60);
    out.push(m);
  }
  return out;
}

/* play: up to maxQ questions. step(n) advances n physics frames and returns their measurements (searching:
   all at once; rendering: one picture per frame inside the recorded window). Returns false to stop early. */
async function play(page, T, maxQ, step) {
  let frame = 0;
  const go = async n => { const f = await step(n, frame); if (!f) return false; frame += f.length; return true; };
  for (let q = 0; q < maxQ; q++) {
    let s = await st(page);
    if (s.phase !== 'r1') break;
    if (s.step !== 'ask') { await T.until(() => CGB.games['over-the-edge'].state().step === 'ask'); s = await st(page); if (s.step !== 'ask') break; }
    await markAll(page, T);
    if ((await st(page)).step === 'lanes') await page.keyboard.press('r');
    for (let guard = 0; guard < 400; guard++) {
      if (!await go(30)) return frame;
      if ((await st(page)).step !== 'dropping') break;
    }
    if (!await go(IDLE)) return frame;
    if ((await st(page)).step === 'next') await page.keyboard.press('Space');
  }
  return frame;
}

/* the final: straight to the jackpot round, the gold counter put `near` of its radius short of falling; every
   team right on each question and random lanes (each right team drops two counters) */
async function playFinal(page, T, near, maxQ, step) {
  await page.evaluate(() => CGB.test.ote.toFinal());
  await T.until(() => CGB.games['over-the-edge'].state().phase === 'final');
  await page.evaluate(n => CGB.test.ote.jackpotNear(n), near);
  let frame = 0;
  const go = async n => { const f = await step(n, frame); if (!f) return false; frame += f.length; return true; };
  for (let q = 0; q < maxQ; q++) {
    if (!await T.until(() => CGB.games['over-the-edge'].state().step === 'ask', 30000)) break;
    await markAll(page, T);
    if ((await st(page)).step === 'lanes') await page.keyboard.press('r');
    for (let guard = 0; guard < 400; guard++) {
      if (!await go(30)) return frame;
      const s = await st(page);
      if (s.step !== 'dropping') break;
    }
    if (!await go(IDLE)) return frame;
    if ((await page.evaluate(() => CGB.test.ote.info().jackpotFell))) break;
    if ((await st(page)).step === 'next') await page.keyboard.press('Space');
  }
  return frame;
}

module.exports = { STAGE_CSS, playFinal, timer, setup, play, advance, st, markAll, IDLE, D, R, JR };
