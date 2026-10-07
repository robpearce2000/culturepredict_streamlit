// The 3D upgrade is presentation only: the moments play, can be skipped with Space or Enter,
// fit the rules (no rule changes), and turn into simple changes with reduced motion.
const { test } = require('@playwright/test');
const { openBundle, state, mark, expect } = require('./helpers');

async function reduce(page) { await page.addInitScript(() => { try { localStorage.setItem('cgb.settings', JSON.stringify({ reducedMotion: true })); } catch (e) { /* no storage */ } }); }

test('Outpace: the set is built, the Final Sprint has no track or set clock, the lights turn red in the last ten seconds, and a catch can be skipped', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await page.click('#op-startBtn', { timeout: 60000 });
  let s3 = await page.evaluate(() => CGB.test.outpace.scene3d());
  expect(s3.arch).toBe(true);
  // no step-counter strip in the Deal Round: the track shows the gap, and the status line says it in words
  await page.keyboard.press('2');
  await expect(page.locator('#game-outpace .op-gap, #game-outpace .op-gapbar')).toHaveCount(0);
  await expect(page.locator('#op-dealStatus')).toContainText('Hunter 3 steps behind');
  // the Final Sprint: no track and no clock in the set, just the racers (the timer is on screen)
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  s3 = await page.evaluate(() => CGB.test.outpace.scene3d());
  expect(s3.sprintTrack).toBe(false); expect(s3.clock).toBe(false);
  await expect(page.locator('#game-outpace .op-gap')).toHaveCount(0);
  await expect(page.locator('#op-sprintTarget')).toContainText('of'); expect(s3.mood).toBe('normal');
  await page.evaluate(() => CGB.test.outpace.setTime(8));
  await expect.poll(async () => (await page.evaluate(() => CGB.test.outpace.scene3d())).mood).toBe('red');
  // time runs out: the question on screen is still answered and marked, then the catch plays, and Enter skips straight to the result
  await page.evaluate(() => CGB.test.outpace.setTime(0.15));
  await expect(page.locator('#op-sprintFinal')).toContainText("Time's up");
  await mark(page, 'outpace', () => false);
  await expect.poll(async () => (await page.evaluate(() => CGB.test.outpace.scene3d())).mode).toMatch(/buildup|explode/);
  const t0 = Date.now();
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await state(page, 'outpace')).phase, { timeout: 1500 }).toBe('summary');
  expect(Date.now() - t0).toBeLessThan(1500);
  expect(log.errors).toEqual([]);
});

test('Outpace: an escape bursts through the arch with confetti, and Space skips it', async ({ page }) => {
  await openBundle(page, '#outpace');
  await page.click('#op-startBtn', { timeout: 60000 });
  await page.evaluate(() => CGB.test.outpace.toSprint(300));
  for (let i = 0; i < 80; i++) {
    const s = await state(page, 'outpace');
    if (s.frozen || s.phase !== 'sprint') break;
    if (s.round === 'think' || s.round === 'show') { await page.keyboard.press('Space'); await page.waitForTimeout(550); }
    else if (s.round === 'mark') { await page.keyboard.press('c'); await page.keyboard.press('Enter'); }
    else await page.waitForTimeout(100);
  }
  await expect.poll(async () => (await page.evaluate(() => CGB.test.outpace.scene3d())).mode, { timeout: 8000 }).toBe('escape');
  await expect.poll(async () => (await page.evaluate(() => CGB.test.outpace.scene3d())).confetti, { timeout: 6000 }).toBe(true);
  await page.keyboard.press(' ');
  await expect.poll(async () => (await state(page, 'outpace')).phase, { timeout: 1500 }).toBe('summary');
});

async function starTile(page) {
  return page.evaluate(() => { const cats = CGB.games['category-clash'].board(); for (let c = 0; c < cats.length; c++) for (let r = 0; r < cats[c].tiles.length; r++) if (cats[c].tiles[r].star) return [c, r]; });
}
test('Category Clash: a picked tile flips into the question, the ★ moment can be skipped, and answered tiles show the winner', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await page.click('#cc-startBtn');
  // an ordinary tile: the question and its countdown start at once (the flip is only for show)
  await page.keyboard.press('Enter');
  let s = await state(page, 'category-clash');
  expect(s.round).toBe('think');
  // the team panels show in the question card, and a tap on one marks that team
  await expect(page.locator('#cc-qboard .cm-team')).toHaveCount(4);
  await expect(page.locator('#cc-qboard')).toBeVisible();
  await page.keyboard.press('Space'); await page.keyboard.press('Space');
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('mark');
  await page.locator('#cc-qboard .cm-team').first().click();
  await expect(page.locator('#cc-qboard .cm-team').first()).toHaveClass(/ok/);
  await page.locator('#cc-qboard .cm-team').first().click();     // back to unmarked, then mark by key below
  await mark(page, 'category-clash', i => i === 0);
  await page.keyboard.press('Enter');
  await expect(page.locator('.cc-tile.used.won')).toHaveCount(1);
  await expect(page.locator('.cc-team.sweep')).toHaveCount(1);
  // the ★ tile: the lights dim and it spins; Enter skips to the question
  const [c, r] = await starTile(page);
  await page.click(`.cc-tile[data-c="${c}"][data-r="${r}"]`);
  s = await state(page, 'category-clash');
  expect(s.step).toBe('reveal');
  expect(s.round).toBe('idle');                              // no countdown during the moment
  await expect(page.locator('#cc-play')).toHaveClass(/dim/);
  await page.keyboard.press('Enter');
  s = await state(page, 'category-clash');
  expect(s.step).toBe('ask'); expect(s.round).toBe('think');
  await expect(page.locator('#cc-play')).not.toHaveClass(/dim/);
  expect(log.errors).toEqual([]);
});

test('Category Clash with reduced motion: the ★ tile goes straight to its question', async ({ page }) => {
  await reduce(page);
  await openBundle(page, '#category-clash');
  await page.click('#cc-startBtn');
  const [c, r] = await starTile(page);
  await page.click(`.cc-tile[data-c="${c}"][data-r="${r}"]`);
  const s = await state(page, 'category-clash');
  expect(s.step).toBe('ask'); expect(s.round).toBe('think');
});

async function hexWin(page) {
  for (let c = 0; c < 4; c++) {
    await page.click(`#hh-board .hh-hex[data-c="${c}"][data-r="1"]`);
    await page.keyboard.press('Space'); await page.keyboard.press('Space');
    await expect.poll(async () => (await state(page, 'hex-hunt')).round).toBe('mark');
    await page.keyboard.press('1');
    if (c < 3) {
      // a claimed hexagon pops up with a burst of light; the team's edges glow more as it grows
      await page.keyboard.press('Enter');
      await expect(page.locator(`#hh-board .hh-hex[data-c="${c}"][data-r="1"].own-0`)).toHaveCount(1);
    }
  }
  await page.keyboard.press('Enter');                          // "They've joined their edges!"
}
test('Hex Hunt: chunky hexes, claims pop, edges glow, and the winning chain lights up (Enter skips it)', async ({ page }) => {
  const log = await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await expect(page.locator('#hh-board .hh-hex .side')).toHaveCount(16);
  await expect(page.locator('#hh-board .hh-hex .face')).toHaveCount(16);
  await hexWin(page);
  expect((await state(page, 'hex-hunt')).phase).toBe('celebrate');
  await expect(page.locator('#hh-board .hh-hex.chain')).toHaveCount(4);
  const g0 = await page.locator('#hh-board .hh-edges').evaluate(e => +getComputedStyle(e).getPropertyValue('--g0'));
  expect(g0).toBe(1);                                          // edge to edge
  await page.keyboard.press('Enter');
  expect((await state(page, 'hex-hunt')).phase).toBe('won');
  await expect(page.locator('#hh-win')).toBeVisible();
  expect(log.errors).toEqual([]);
});

test('Hex Hunt with reduced motion: no sweep, straight to the result', async ({ page }) => {
  await reduce(page);
  await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await hexWin(page);
  expect((await state(page, 'hex-hunt')).phase).toBe('won');
});
