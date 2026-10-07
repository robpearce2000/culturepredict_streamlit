// Fixes after Rob's classroom use (version 1.5.1): Space confirms marking, a lost Deal Round still offers the
// Final Sprint, the escape can be seen, Esc and full screen, "Change" topic in place, Hex Hunt's top bar.
const { test } = require('@playwright/test');
const { openBundle, state, mark, playOutpace, expect } = require('./helpers');

test('Space confirms the marking, as Enter does, in every game', async ({ page }) => {
  await openBundle(page, '#category-clash');
  await page.click('#cc-startBtn');
  await page.locator('#cc-board .cc-tile[data-c="0"][data-r="0"]').click();
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('think');
  for (let k = 0; k < 4 && (await state(page, 'category-clash')).round === 'think'; k++) { await page.keyboard.press('Space'); await page.waitForTimeout(550); }
  if ((await state(page, 'category-clash')).round === 'show') await page.keyboard.press('Space');
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('mark');
  await page.keyboard.press('1'); await page.keyboard.press('2');
  await page.waitForTimeout(600);
  await page.keyboard.press('Space');                        // confirms
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('done');
  const s = await state(page, 'category-clash');
  expect(s.teams[0].score + s.teams[1].score).toBeGreaterThan(0);
});

test('a lost Deal Round still offers the Final Sprint, and the class may stop there', async ({ page }) => {
  test.setTimeout(240000);
  await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  for (let i = 0; i < 300; i++) {
    const s = await state(page, 'outpace');
    if (s.roundEnd) break;
    if (s.phase === 'deal' && !s.dealReward) await page.keyboard.press('3');
    else if (s.phase === 'deal' && s.round === 'done') await page.keyboard.press('Enter');
    else if (s.phase === 'deal' && /think|show|mark/.test(s.round)) { await mark(page, 'outpace', () => false); continue; }
    await page.waitForTimeout(150);
  }
  await expect(page.locator('#op-roundEndVerdict')).toHaveText('Caught!');
  await expect(page.locator('#op-roundEndContinue')).toContainText('Play the Final Sprint');
  await expect(page.locator('#op-roundEndStop')).toBeVisible();
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await state(page, 'outpace')).phase).toBe('sprint');
  // and the other choice: stop with the points banked
  await page.evaluate(() => CGB.app.show('launcher'));
  await page.click('[data-play="outpace"]');
  await page.click('#op-startBtn');
  for (let i = 0; i < 300; i++) {
    const s = await state(page, 'outpace');
    if (s.roundEnd) break;
    if (s.phase === 'deal' && !s.dealReward) await page.keyboard.press('3');
    else if (s.phase === 'deal' && s.round === 'done') await page.keyboard.press('Enter');
    else if (s.phase === 'deal' && /think|show|mark/.test(s.round)) { await mark(page, 'outpace', () => false); continue; }
    await page.waitForTimeout(150);
  }
  await page.click('#op-roundEndStop');
  await expect.poll(async () => (await state(page, 'outpace')).phase).toBe('summary');
});

test('the Final Sprint escape stays in view: the runner is on screen while it dashes home', async ({ page }) => {
  test.setTimeout(240000);
  await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  await page.keyboard.press('Enter');
  let seen = 0, total = 0;
  for (let i = 0; i < 500 && total < 12; i++) {
    const s = await state(page, 'outpace');
    const m = await page.evaluate(() => CGB.test.outpace.scene3d().mode);
    if (m === 'escape') {
      const mo = await page.evaluate(() => CGB.test.outpace.motion());
      total++;
      if (mo.rs[0] > 10 && mo.rs[0] < 950 && mo.rs[1] > 10 && mo.rs[1] < 590) seen++;   // on screen (960 × 600)
      await page.waitForTimeout(200); continue;
    }
    if (s.phase === 'summary') break;
    if (/think|show/.test(s.round)) { await page.keyboard.press('Space'); await page.waitForTimeout(600); }
    else if (s.round === 'mark') { await page.keyboard.press('c'); await page.keyboard.press('Space'); await page.waitForTimeout(300); }
    else await page.waitForTimeout(150);
  }
  expect(total).toBeGreaterThan(3);
  expect(seen / total, 'the runner is in the picture for most of the escape').toBeGreaterThan(0.8);
});

test('Esc opens "Leave this game?" and Esc again goes back to the menu', async ({ page }) => {
  await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await page.keyboard.press('Escape');
  await expect(page.locator('#leaveModal')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#launcher')).toBeVisible();
  await expect(page.locator('#leaveModal')).toBeHidden();
});

test('in full screen the Esc key is kept for the game and only F leaves full screen (where the browser can lock it)', async ({ page }) => {
  await page.addInitScript(() => { window.__locked = []; try { Object.defineProperty(navigator, 'keyboard', { value: { lock: async k => { window.__locked.push(k); }, unlock() { window.__locked.push('unlock'); } }, configurable: true }); } catch (e) { /* ignore */ } });
  await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await page.keyboard.press('f');
  await expect.poll(() => page.evaluate(() => !!document.fullscreenElement)).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__locked.some(k => Array.isArray(k) && k[0] === 'Escape'))).toBe(true);
  await page.keyboard.press('Escape');                          // reaches the game: the Leave prompt
  await expect(page.locator('#leaveModal')).toBeVisible();
  await page.keyboard.press('Escape');                          // and again: back to the menu, still in full screen
  await expect(page.locator('#launcher')).toBeVisible();
  expect(await page.evaluate(() => !!document.fullscreenElement)).toBe(true);
  await page.keyboard.press('f');
  await expect.poll(() => page.evaluate(() => !!document.fullscreenElement)).toBe(false);
  expect(await page.evaluate(() => window.__locked.includes('unlock'))).toBe(true);
});

test('"Change" on a game\'s setup card opens the topic dropdown in place instead of going to the main screen', async ({ page }) => {
  await openBundle(page, '#category-clash');
  await expect(page.locator('#cc-home')).toBeVisible();
  await page.click('#cc-pack [data-pack="change"]');
  await expect(page.locator('#gamePackMenu')).toBeVisible();
  await expect(page.locator('#cc-home')).toBeVisible();          // still on the game's setup card
  await expect(page.locator('#launcher')).toBeHidden();
  const before = await page.evaluate(() => CGB.bank.active().short);
  await page.locator('#gamePackMenu input[data-id]').first().click();
  const after = await page.evaluate(() => CGB.bank.active().short);
  expect(after).not.toBe(before);
  await expect(page.locator('#cc-pack')).toContainText(after);   // the setup card follows
  await page.click('#gamePackMenu [data-done]');
  await expect(page.locator('#gamePackMenu')).toBeHidden();
  await expect(page.locator('#cc-home')).toBeVisible();
  // Esc closes it too, without leaving the game
  await page.click('#cc-pack [data-pack="change"]');
  await page.keyboard.press('Escape');
  await expect(page.locator('#gamePackMenu')).toBeHidden();
  await expect(page.locator('#cc-home')).toBeVisible();
});

test('Hex Hunt: while playing, the top bar buttons stack down the side and never overlap the turn and round lines', async ({ page }) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await expect(page.locator('#hh-turn')).toBeVisible();
  const r = await page.evaluate(() => {
    const box = e => { const b = e.getBoundingClientRect(); return { l: b.left, t: b.top, r: b.right, b: b.bottom }; };
    const bar = Array.from(document.querySelectorAll('#game-hex-hunt .topbar .btn')).filter(b => b.offsetParent).map(box);
    return { bar, turn: box(document.getElementById('hh-turn')), round: box(document.getElementById('hh-round')), board: box(document.getElementById('hh-board')) };
  });
  expect(r.bar.length).toBeGreaterThanOrEqual(5);
  // one column: all the buttons share the same left edge, one under the other
  expect(new Set(r.bar.map(b => Math.round(b.l))).size).toBe(1);
  const overlap = (a, b) => a.l < b.r && a.r > b.l && a.t < b.b && a.b > b.t;
  for (const b of r.bar) { expect(overlap(b, r.turn)).toBe(false); expect(overlap(b, r.round)).toBe(false); expect(overlap(b, r.board)).toBe(false); }
});
