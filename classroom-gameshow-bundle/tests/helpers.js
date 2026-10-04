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

/* One question in any game: skip the countdown and the "3, 2, 1" with Space, then mark the
   teams whose index passes ok() and confirm with Enter. Returns the marking time in ms. */
async function mark(page, id, ok) {
  await expect.poll(async () => (await state(page, id)).round, { timeout: 15000 }).toMatch(/think|show|mark/);
  if ((await state(page, id)).round === 'think') await page.keyboard.press('Space');
  if ((await state(page, id)).round === 'show') await page.keyboard.press('Space');   // skip the 3, 2, 1
  await expect.poll(async () => (await state(page, id)).round).toBe('mark');
  const s = await state(page, id);
  const n = s.teams ? s.teams.length : s.players ? s.players.length : 6;
  for (let i = 0; i < n; i++) if (ok(i)) await page.keyboard.press(String(i + 1));
  await page.keyboard.press('Enter');
  const t = (await state(page, id)).times;
  return t.confirm - t.mark;
}

/* Play Over the Edge to the summary with keyboard shortcuts. ok(i, q) says which teams are
   right on question q; captains' lanes are picked with R. With opts.manual the physics
   clock only moves when the test advances it. */
async function playOverTheEdge(page, opts) {
  opts = opts || {};
  const ok = opts.ok || ((i, q) => (i + q) % 3 === 0);
  const deadline = Date.now() + (opts.timeout || 200000);
  let q = 0;
  while (Date.now() < deadline) {
    const s = await state(page, 'over-the-edge');
    if (s.phase === 'summary') return s;
    if (s.step === 'ask') { const k = q++; await mark(page, 'over-the-edge', i => ok(i, k)); continue; }
    if (s.step === 'lanes') await page.keyboard.press('r');
    else if (s.step === 'next') await page.keyboard.press('Space');
    if (s.step === 'dropping' && opts.manual) await page.evaluate(() => CGB.test.ote.advance(0.5));
    else await page.waitForTimeout(s.step === 'dropping' ? 400 : 120);
  }
  throw new Error('Over the Edge did not reach the summary in time');
}

/* Play Outpace to the summary with keyboard shortcuts: vote the Bold deal and mark most
   teams wrong in the Deal Rounds, so the Hunter catches quickly; with opts.shortSprint the
   sprint clock is cut to a few seconds. */
async function playOutpace(page, opts) {
  opts = opts || {};
  const deadline = Date.now() + (opts.timeout || 200000);
  let n = 0;
  while (Date.now() < deadline) {
    const s = await state(page, 'outpace');
    if (s.phase === 'summary') return s;
    if (s.roundEnd) await page.keyboard.press('Enter');
    else if (s.phase === 'deal' && !s.dealReward) await page.keyboard.press('3');
    else if (s.phase === 'deal' && s.round === 'done') await page.keyboard.press('Enter');
    else if ((s.phase === 'deal' || s.phase === 'sprint') && /think|show|mark/.test(s.round)) {
      if (s.phase === 'sprint' && opts.shortSprint && s.timeLeft > 3 && n > 2) await page.evaluate(() => CGB.test.outpace.setTime(2.5));
      const k = n++;
      await mark(page, 'outpace', i => s.phase === 'sprint' ? (i + k) % 2 === 0 : i === 0 && k % 4 === 3);
      continue;
    }
    await page.waitForTimeout(120);
  }
  throw new Error('Outpace did not reach the summary in time');
}

/* Play Category Clash to the results: every tile, ok(i, q) says which teams are right */
async function playCategoryClash(page, opts) {
  opts = opts || {};
  const ok = opts.ok || ((i, q) => (i + q) % 3 !== 0);
  const end = Date.now() + (opts.timeout || 200000);
  let q = 0;
  while (Date.now() < end) {
    const s = await state(page, 'category-clash');
    if (s.phase === 'summary') return s;
    if (s.phase === 'board') await page.keyboard.press('Enter');
    else if (s.phase === 'question' && s.step === 'done') await page.keyboard.press('Enter');
    else if (s.phase === 'question') { const k = q++; await mark(page, 'category-clash', i => ok(i, k)); continue; }
    await page.waitForTimeout(60);
  }
  throw new Error('Category Clash did not finish');
}

/* Play Hex Hunt to the results: picks[q % picks.length] is the key pressed when marking:
   '1' or '2' for the half with more right answers, 'n' for Neither */
async function playHexHunt(page, opts) {
  opts = opts || {};
  const picks = opts.picks || ['1', '2', '1', 'n', '2'];
  const end = Date.now() + (opts.timeout || 200000);
  let q = 0;
  while (Date.now() < end) {
    const s = await state(page, 'hex-hunt');
    if (s.phase === 'summary') return s;
    if (s.phase === 'board' || s.phase === 'won' || s.round === 'done') await page.keyboard.press('Enter');
    else if (s.round === 'think' || s.round === 'show') await page.keyboard.press('Space');
    else if (s.round === 'mark') { await page.keyboard.press(picks[q++ % picks.length]); continue; }
    await page.waitForTimeout(60);
  }
  throw new Error('Hex Hunt did not finish');
}

/* Team names sit behind "Edit team names" on every setup card */
const NAME_INPUT = { 'over-the-edge': 'ote-cname', outpace: 'op-gname', 'category-clash': 'cc-name', 'hex-hunt': 'hh-name' };
async function setNames(page, game, names) {
  const card = page.locator(`#game-${game} .setup`);
  const box = card.locator('details.tnames').first();   // the first is the team names
  if (!(await box.evaluate(d => d.open))) await box.locator('summary').click();
  for (let i = 0; i < names.length; i++) await page.fill(`#${NAME_INPUT[game]}${i}`, names[i]);
}
const PICK = (seg, v) => `${seg} button[data-v="${v}"]`;

module.exports = { URL, DIST, TEST_BUILD, SEED, openBundle, state, mark, playOverTheEdge, playOutpace, playCategoryClash, playHexHunt, setNames, PICK, expect };
