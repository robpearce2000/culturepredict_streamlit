// Shared helpers for the bundle tests.
const path = require('path');
const { expect, test } = require('@playwright/test');

// The product, exactly as shipped
const DIST = path.join(__dirname, '..', 'dist', 'showtime-classroom-gameshows.html');
const URL = 'file://' + DIST;
// The test build: the same file plus the test-only blocks (seeding and shortcuts).
// build.js makes the product by removing those blocks, so both run the same game code.
const TEST_BUILD = path.join(__dirname, '..', 'test-build', 'showtime-test.html');

// Every run uses the same random sequence for question order, boards and physics.
// Run with SEED=<number> to try another sequence; the seed is printed with any failure.
const SEED = process.env.SEED ? Number(process.env.SEED) : 20261004;

/* Open Showtime and record anything that should never happen: network requests and console errors.
   opts.product  open the shipped file instead of the test build (no shortcuts, no seeding)
   opts.quality  'low' (default for gameplay tests: rules don't depend on shadows or glow) or 'high' */
async function openBundle(page, hash, opts) {
  opts = opts || {};
  const log = { requests: [], errors: [] };
  page.on('request', r => { const u = r.url(); if (!/^(file|data|blob|about):/.test(u)) log.requests.push(u); });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') log.errors.push(m.type() + ': ' + m.text()); });
  page.on('pageerror', e => log.errors.push('pageerror: ' + e.message));
  try { test.info().annotations.push({ type: 'seed', description: String(SEED) }); } catch (e) { /* outside a test */ }
  await page.addInitScript(([seed, quality]) => {
    window.__SHOWTIME_SEED__ = seed;
    try {
      if (quality) {
        const k = 'cgb.settings', cur = JSON.parse(localStorage.getItem(k) || '{}');
        if (!cur.quality) { cur.quality = quality; localStorage.setItem(k, JSON.stringify(cur)); }
      }
    } catch (e) { /* storage blocked: the game falls back to its defaults */ }
  }, [SEED, opts.quality || 'low']);
  await page.goto('file://' + (opts.product ? DIST : TEST_BUILD) + (hash || ''));
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
    if (s.step === 'dropping' && opts.manual) await page.evaluate(() => CGB.test.ote.advance(0.5));
    else await page.waitForTimeout(s.step === 'dropping' ? 400 : 120);
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
      // mark straight away most of the time; now and then show the answer first (optional)
      if (!s.answerShown && n % 5 === 4) { n++; await page.keyboard.press('a'); }
      else { await page.keyboard.press(s.phase === 'sprint' && n % 3 ? 'c' : 'w'); n++; }
      if (s.phase === 'sprint' && opts.shortSprint && s.timeLeft > 3) await page.evaluate(() => CGB.test.outpace.setTime(2.5));
    }
    await page.waitForTimeout(150);
  }
  throw new Error('Outpace did not reach the summary in time');
}

async function setNames(page, prefix, a, b) {
  await page.fill(`#${prefix}-name0`, a);
  await page.fill(`#${prefix}-name1`, b);
}

module.exports = { URL, DIST, TEST_BUILD, SEED, openBundle, state, playOverTheEdge, playOutpace, setNames, expect };
