// Core behaviour of the shipped single file.
const fs = require('fs');
const { test } = require('@playwright/test');
const { openBundle, state, mark, playOverTheEdge, playOutpace, setNames, PICK, expect, DIST, TEST_BUILD } = require('./helpers');

test('loads from file:// with no network requests and no console errors', async ({ page }) => {
  const log = await openBundle(page, '', { product: true });
  await expect(page.locator('#launcher')).toBeVisible();
  await expect(page.locator('#wordmark svg')).toBeVisible();
  await expect(page.locator('[data-play="over-the-edge"]')).toBeVisible();
  await expect(page.locator('[data-play="outpace"]')).toBeVisible();
  await expect(page.getByText('More shows coming soon')).toBeVisible();
  // open every panel on the launcher too
  await page.click('#openBank'); await expect(page.locator('#bankModal')).toBeVisible(); await page.keyboard.press('Escape');
  await page.click('#openAbout'); await expect(page.locator('#aboutModal')).toContainText('MIT'); await page.keyboard.press('Escape');
  await page.click('#openHost'); await expect(page.locator('#hostModal')).toBeVisible(); await page.keyboard.press('Escape');
  await page.waitForTimeout(1500);
  expect(log.requests).toEqual([]);
  expect(log.errors).toEqual([]);
});

test('the dist file references nothing outside itself', async () => {
  const html = fs.readFileSync(require('./helpers').DIST, 'utf8');
  expect(html).not.toMatch(/(src|href)\s*=\s*["']\s*(https?:)?\/\//i);
  expect(html).not.toMatch(/@import/i);
  expect(html).not.toMatch(/fonts\.googleapis|cdnjs|jsdelivr|unpkg/i);
  expect(html).toContain("connect-src 'none'");
});

test('the shipped file has no test shortcuts, and is the test build minus its test-only blocks', async () => {
  const product = fs.readFileSync(DIST, 'utf8'), testBuild = fs.readFileSync(TEST_BUILD, 'utf8');
  expect(product).not.toMatch(/@test-only|CGB\.test\b|__SHOWTIME_/);
  const stripped = testBuild.replace(/[ \t]*\/\* @test-only[\s\S]*?\/\* @end-test-only \*\/[ \t]*\n?/g, '');
  expect(stripped === product, 'product equals the test build without its test-only blocks').toBe(true);
});

test('launcher → each game → back to launcher', async ({ page }) => {
  const log = await openBundle(page, '', { product: true });
  // Over the Edge, back with the Menu button
  await page.click('[data-play="over-the-edge"]');
  await expect(page.locator('#game-over-the-edge')).toBeVisible();
  await expect(page.locator('#ote-home')).toBeVisible();
  await page.click('#ote-menuBtn');
  await expect(page.locator('#launcher')).toBeVisible();
  // Outpace, back with Esc
  await page.click('[data-play="outpace"]');
  await expect(page.locator('#game-outpace')).toBeVisible();
  await expect(page.locator('#op-home')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#launcher')).toBeVisible();
  // Leaving mid-game asks first
  await page.click('[data-play="over-the-edge"]');
  await page.click('#ote-startBtn');
  await page.keyboard.press('Escape');
  await expect(page.locator('#leaveModal')).toBeVisible();
  await page.click('#leaveModal [data-close]');
  await expect(page.locator('#leaveModal')).toBeHidden();
  expect(await page.evaluate(() => CGB.app.current)).toBe('over-the-edge');
  await page.keyboard.press('Escape');
  await page.click('#leaveConfirm');
  await expect(page.locator('#launcher')).toBeVisible();
  // back in again: the game has reset to its setup screen
  await page.click('[data-play="over-the-edge"]');
  await expect(page.locator('#ote-home')).toBeVisible();
  expect((await state(page, 'over-the-edge')).phase).toBe('home');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('a full game of Over the Edge with keyboard shortcuts reaches the summary', async ({ page }) => {
  const log = await openBundle(page, '#over-the-edge');
  await page.click(PICK('#ote-segTeams', 2));
  await setNames(page, 'over-the-edge', ['Ada', 'Ben']);
  await page.locator('#ote-cname1').press('Enter');          // Enter starts the game
  await expect(page.locator('#ote-home')).toBeHidden();
  await page.evaluate(() => CGB.test.ote.manual(true));      // physics only moves when the test says so
  const s = await playOverTheEdge(page, { manual: true });
  expect(s.phase).toBe('summary');
  expect(s.qTotal).toBe(8);                                   // 14 questions in Round 1, then 8 in the final
  await expect(page.locator('#ote-summary')).toBeVisible();
  await expect(page.locator('#ote-sumCard')).toContainText('Ada');
  await expect(page.locator('#ote-sumCard .cm-miscon')).toContainText('Reteach these');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('a full game of Outpace with keyboard shortcuts reaches the summary', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await page.click(PICK('#op-segGroups', 3));
  await setNames(page, 'outpace', ['Cara', 'Dev', 'Eli']);
  await page.fill('#op-className', '10B');
  await page.locator('#op-className').press('Enter');
  await expect(page.locator('#op-hud-deal')).toBeVisible();
  const s = await playOutpace(page, { shortSprint: true });
  expect(s.phase).toBe('summary');
  expect(s.dealRound).toBe(2);                               // two Deal Rounds before the sprint
  await expect(page.locator('#op-summary')).toBeVisible();
  await expect(page.locator('#op-summaryPlayers')).toContainText('10B');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('every game counts down 20 seconds before "show me", and Space skips it', async ({ page }) => {
  await openBundle(page);
  const games = [
    ['over-the-edge', '#ote-count', null, null],
    ['outpace', '#op-dealCount', 'deal', '2'],
    ['category-clash', '#cc-count', 'board', 'Enter'],
    ['hex-hunt', '#hh-count', 'board', 'Enter']
  ];
  for (const [id, count, ready, key] of games) {
    await page.click(`[data-play="${id}"]`, { timeout: 60000 });
    await page.click(`#game-${id} .setup-go .btn`);
    if (ready) { await expect.poll(async () => (await state(page, id)).phase).toBe(ready); await page.keyboard.press(key); }
    await expect.poll(async () => (await state(page, id)).round).toBe('think');
    await expect(page.locator(count)).toBeVisible();
    await expect(page.locator(count)).toContainText(/^(20|19)/);
    await page.keyboard.press('Space');
    await expect.poll(async () => (await state(page, id)).round).toMatch(/show|mark/);
    await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
    await expect(page.locator('#launcher')).toBeVisible();
  }
});

/* ---------- Full-quality playthroughs (pre-push run only: npm test). Shadows and glow on,
   real time, full-length settings, on the shipped file, so errors or slowdowns that only
   happen on those code paths are caught. ---------- */
test('@hq a full-length game of Over the Edge at High graphics, on the shipped file', async ({ page }) => {
  test.setTimeout(600000);
  await page.setViewportSize({ width: 1366, height: 768 });
  const log = await openBundle(page, '#over-the-edge', { product: true, quality: 'high' });
  expect(await page.evaluate(() => CGB.settings.get('quality'))).toBe('high');
  await page.click('#ote-startBtn');
  const s = await playOverTheEdge(page, { timeout: 540000 });
  expect(s.phase).toBe('summary');
  await expect(page.locator('#ote-summary')).toBeVisible();
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('@hq a full game of Outpace at High graphics, with the full 60-second sprint, on the shipped file', async ({ page }) => {
  test.setTimeout(600000);
  await page.setViewportSize({ width: 1366, height: 768 });
  const log = await openBundle(page, '#outpace', { product: true, quality: 'high' });
  expect(await page.evaluate(() => CGB.settings.get('quality'))).toBe('high');
  await page.click('#op-startBtn');
  const s = await playOutpace(page, { timeout: 540000 });
  expect(s.phase).toBe('summary');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

/* ---------- Shortcuts: start where the test needs to be (test build only) ---------- */
async function oteFinal(page) {
  await page.addInitScript(() => { window.__SHOWTIME_MANUAL__ = true; });   // the clock only moves when the test says
  await openBundle(page, '#over-the-edge');
  await page.click('#ote-startBtn');
  await page.evaluate(() => { CGB.test.ote.manual(true); CGB.test.ote.toFinal(); });
  await expect.poll(async () => (await state(page, 'over-the-edge')).phase).toBe('final');
}
/* One final question: team 1 alone is right and drops its two counters down this lane */
async function dropFinalCounter(page, lane) {
  await expect.poll(async () => (await state(page, 'over-the-edge')).step).toBe('ask');
  await mark(page, 'over-the-edge', i => i === 0);
  await expect.poll(async () => (await state(page, 'over-the-edge')).step).toBe('lanes');
  await page.keyboard.press(String(lane));
  for (let i = 0; i < 40 && (await state(page, 'over-the-edge')).step === 'dropping'; i++) await page.evaluate(() => CGB.test.ote.advance(0.5));
}

test('Over the Edge: pushing the jackpot counter over the edge wins the jackpot', async ({ page }) => {
  await oteFinal(page);
  await page.evaluate(() => CGB.test.ote.jackpotNear(0.15));
  for (let k = 0; k < 8; k++) {
    const info = await page.evaluate(() => CGB.test.ote.info());
    if (process.env.DEBUG_OTE) console.log(k, info.progress.toFixed(3), info.jackpotFell, (await state(page, 'over-the-edge')).step);
    if (info.jackpotFell) break;
    await dropFinalCounter(page, [2, 3, 1, 4][k % 4]);
    const s = await state(page, 'over-the-edge');
    if (process.env.DEBUG_OTE) console.log('after drop', k, s.phase, s.step, (await page.evaluate(() => CGB.test.ote.info())).progress.toFixed(3));
    if (s.step === 'next') await page.keyboard.press('Space');
  }
  const info = await page.evaluate(() => CGB.test.ote.info());
  expect(info.jackpotFell, 'the jackpot counter fell over the edge').toBe(true);
  expect(info.jackpotWon).toBe(true);
});

test('Over the Edge physics is the same every run with the same seed', async ({ browser }) => {
  const run = async () => {
    const page = await browser.newPage();
    await oteFinal(page);
    await dropFinalCounter(page, 2);
    await page.evaluate(() => CGB.test.ote.advance(3));
    const info = await page.evaluate(() => CGB.test.ote.info());
    await page.close();
    return info.tray;
  };
  const a = await run(), b = await run();
  expect(a.length).toBeGreaterThan(100);
  expect(b).toBe(a);
});

test('Outpace: the Final Sprint ends when time runs out and shows the result', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  await expect(page.locator('#op-hud-sprint')).toBeVisible();
  await mark(page, 'outpace', () => true);
  expect((await state(page, 'outpace')).net).toBe(4);       // one step for each correct team
  await page.evaluate(() => CGB.test.outpace.setTime(1));
  await expect.poll(async () => (await state(page, 'outpace')).phase, { timeout: 20000 }).toBe('summary');
  await expect(page.locator('#op-finalVerdict')).not.toBeEmpty();
  expect(log.errors).toEqual([]);
});
