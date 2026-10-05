// Over the Edge fairness: whole games with six teams, all correct every time, the real physics.
// Teams take turns and keep whatever comes over the edge on their own drop. Which team wins
// most is down to luck, but dropping first (above all on the first drop of the game) must not
// be an advantage.
const { test } = require('@playwright/test');
const { openBundle, expect } = require('./helpers');
const note = (name, value) => test.info().annotations.push({ type: name, description: String(value) });

async function playRound1(browser, seed) {
  const ctx = await browser.newContext({ viewport: { width: 960, height: 600 } });
  const page = await ctx.newPage();
  await page.addInitScript(() => { window.__SHOWTIME_MANUAL__ = true; });
  await openBundle(page, '#over-the-edge');
  await page.evaluate(sd => CGB.test.seed(sd), seed);          // a different game each time (the shelf is filled on Start)
  await page.click('#ote-segTeams button[data-v="6"]', { timeout: 60000 });
  await page.click('#ote-startBtn');
  let s = seed * 31 + 7; const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  for (let guard = 0; guard < 4000; guard++) {
    const st = await page.evaluate(() => CGB.games['over-the-edge'].state());
    if (st.phase !== 'r1') break;
    if (st.step === 'ask') {
      if (st.round === 'think' || st.round === 'show') { await page.keyboard.press('Space'); continue; }
      if (st.round === 'mark') { await page.keyboard.press('c'); await page.keyboard.press('Enter'); continue; }
    }
    if (st.step === 'lanes') await page.keyboard.press(String(1 + Math.floor(rnd() * 4)));   // captains pick any lane
    else if (st.step === 'dropping') await page.evaluate(() => CGB.test.ote.advance(0.5));
    else if (st.step === 'next') { if (st.qIndex >= st.qTotal) break; await page.keyboard.press('Space'); }
    else await page.waitForTimeout(10);
  }
  const log = (await page.evaluate(() => CGB.games['over-the-edge'].state())).dropLog.filter(d => d.phase === 'r1' && d.order.length === 6);
  await ctx.close();
  return log.map((d, q) => Object.assign(d, { q }));
}

test('@slow Over the Edge: dropping first is no advantage, on the first question or after (54 questions)', async ({ browser }) => {
  test.setTimeout(900000);
  const GAMES = 9, logs = []; let next = 1;
  await Promise.all([0, 1, 2].map(async () => { while (next <= GAMES) { const sd = next++; logs.push(...await playRound1(browser, sd)); } }));
  expect(logs.length).toBeGreaterThanOrEqual(50);
  const team = [0, 0, 0, 0, 0, 0], pos = [0, 0, 0, 0, 0, 0];
  let total = 0, q1First = 0, q1All = 0, q1n = 0;
  logs.forEach(d => {
    d.won.forEach((w, i) => { team[i] += w; total += w; });
    d.order.forEach((t, k) => { pos[k] += d.won[t]; });
    if (d.q === 0) { q1n++; q1First += d.won[d.order[0]]; q1All += d.won.reduce((a, b) => a + b, 0) / 6; }
  });
  const avg = total / 6;
  const teamDiff = team.map(w => w / avg - 1), posDiff = pos.map(w => w / avg - 1);
  note('questions', logs.length);
  note('counters won per question', (total / logs.length).toFixed(2));
  note('each team vs the average', teamDiff.map(d => (d * 100).toFixed(0) + '%').join(' '));
  note('by place in the drop order vs the average', posDiff.map(d => (d * 100).toFixed(0) + '%').join(' '));
  note('first question: first to drop vs average team', `${(q1First / q1n).toFixed(2)} vs ${(q1All / q1n).toFixed(2)}`);
  expect(total / logs.length, 'counters are being won').toBeGreaterThan(2);
  // each team's total is recorded (above) but not tested: with "your drop, your counters" it is
  // down to where the counters land. Dropping first wins no more than the others, on average
  // and on the first question of a game
  expect(posDiff[0], 'first to drop is not ahead').toBeLessThanOrEqual(0.15);
  expect(q1First / q1n, 'first to drop on question 1 is not ahead').toBeLessThanOrEqual(q1All / q1n * 1.15 + 0.1);
});

test('Over the Edge drop order is random, shown on screen, and the set has no flickering faces', async ({ page }) => {
  await page.addInitScript(() => { window.__SHOWTIME_MANUAL__ = true; });
  const log = await openBundle(page, '#over-the-edge');
  // no two fixed set pieces share a face plane where they overlap (that is what flickers)
  expect(await page.evaluate(() => CGB.test.ote.coplanar())).toEqual([]);
  await page.click('#ote-segTeams button[data-v="4"]', { timeout: 60000 });
  await page.click('#ote-startBtn');
  const orders = []; let seen = -1;
  for (let guard = 0; guard < 2000 && orders.length < 4; guard++) {
    const st = await page.evaluate(() => CGB.games['over-the-edge'].state());
    if (st.step === 'ask') {
      if (st.round === 'think' || st.round === 'show') { await page.keyboard.press('Space'); continue; }
      if (st.round === 'mark') { await page.keyboard.press('c'); await page.keyboard.press('Enter'); continue; }
    }
    if (st.step === 'lanes') {
      if (seen !== st.qIndex) {
        seen = st.qIndex;
        // the order is on screen: a numbered list, and each team's card says when it drops
        const shown = await page.locator('.ote-order li').allTextContents();
        expect(shown).toHaveLength(4);
        await expect(page.locator('#game-over-the-edge')).toContainText('Drops 1st');
        orders.push(st.dropOrder.slice());
      }
      await page.keyboard.press('r');
      continue;
    }
    if (st.step === 'dropping') await page.evaluate(() => CGB.test.ote.advance(0.5));
    else if (st.step === 'next') await page.keyboard.press('Space');
    else await page.waitForTimeout(10);
  }
  expect(orders.length).toBe(4);
  orders.forEach(o => expect([...o].sort()).toEqual([0, 1, 2, 3]));
  // not the same team first every time
  expect(new Set(orders.map(o => o[0])).size).toBeGreaterThan(1);
  expect(log.errors).toEqual([]);
});
