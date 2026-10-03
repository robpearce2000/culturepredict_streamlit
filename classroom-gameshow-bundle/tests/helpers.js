// Shared helpers for the bundle tests.
const path = require('path');
const { expect } = require('@playwright/test');

const DIST = path.join(__dirname, '..', 'dist', 'classroom-gameshow-bundle.html');
const URL = 'file://' + DIST;

/* Open the bundle and record anything that should never happen: network requests and console errors */
async function openBundle(page, hash) {
  const log = { requests: [], errors: [] };
  page.on('request', r => { const u = r.url(); if (!/^(file|data|blob|about):/.test(u)) log.requests.push(u); });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') log.errors.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', e => log.errors.push('pageerror: ' + e.message));
  await page.goto(URL + (hash || ''));
  await page.waitForFunction(() => window.CGB && CGB.app);
  return log;
}
const state = (page, id) => page.evaluate(g => CGB.games[g].state(), id);

/* Play Over the Edge to the summary using only keyboard shortcuts.
   plan(step, qIndex) returns the key to press for a question. */
async function playOverTheEdge(page, opts) {
  opts = opts || {};
  const deadline = Date.now() + (opts.timeout || 200000);
  let correctLeft = opts.correct == null ? 2 : opts.correct;
  while (Date.now() < deadline) {
    const s = await state(page, 'over-the-edge');
    if (s.phase === 'summary') return s;
    if (s.step === 'ask') { await page.keyboard.press(correctLeft > 0 ? 'c' : 'w'); if (correctLeft > 0) correctLeft--; }
    else if (s.step === 'steal') await page.keyboard.press('n');
    else if (s.step === 'chute') await page.keyboard.press(String(1 + Math.floor(Math.random() * 4)));
    else if (s.step === 'next') await page.keyboard.press('Space');
    await page.waitForTimeout(s.step === 'dropping' ? 400 : 120);
  }
  throw new Error('Over the Edge did not reach the summary in time');
}

/* Play Outpace to the summary with keyboard shortcuts. In each Deal Round
   pick the bold deal and answer wrong, so the Hunter catches quickly. */
async function playOutpace(page, opts) {
  opts = opts || {};
  const deadline = Date.now() + (opts.timeout || 200000);
  let n = 0;
  while (Date.now() < deadline) {
    const s = await state(page, 'outpace');
    if (s.phase === 'summary') return s;
    if (s.roundEnd) await page.keyboard.press('Enter');
    else if (s.phase === 'deal' && !s.dealReward) await page.keyboard.press('3');
    else if ((s.phase === 'deal' || s.phase === 'sprint') && !s.awaitingNext) {
      if (!s.answerShown) await page.keyboard.press('a');
      else { await page.keyboard.press(s.phase === 'sprint' && n++ % 3 ? 'c' : 'w'); }
      if (s.phase === 'sprint' && opts.shortSprint && s.timeLeft > 3) await page.evaluate(() => CGB.games.outpace.setTime(2.5));
    }
    await page.waitForTimeout(150);
  }
  throw new Error('Outpace did not reach the summary in time');
}

async function setNames(page, prefix, a, b) {
  await page.fill(`#${prefix}-name0`, a);
  await page.fill(`#${prefix}-name1`, b);
}

module.exports = { URL, DIST, openBundle, state, playOverTheEdge, playOutpace, setNames, expect };
