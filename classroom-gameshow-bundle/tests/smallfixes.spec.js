// Small fixes: Hex Hunt claims seen on the board, full screen, and Maths answer time.
const { test } = require('@playwright/test');
const { openBundle, state, expect } = require('./helpers');
const GAMES = { 'over-the-edge': 'ote', outpace: 'op', 'category-clash': 'cc', 'hex-hunt': 'hh' };

test('Hex Hunt: a won hexagon changes colour with its burst only once the board is back in view', async ({ page }) => {
  const log = await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await page.locator('#hh-board .hh-hex').first().click();
  await page.locator('#game-hex-hunt [data-cm="show"]').click();
  await expect.poll(async () => (await state(page, 'hex-hunt')).round, { timeout: 10000 }).toBe('mark');
  await page.locator('#game-hex-hunt .hh-side').first().click();
  // still on the question: the hexagon is won, but on the board it has not changed yet
  expect((await state(page, 'hex-hunt')).owners.filter(o => o >= 0)).toHaveLength(1);
  await expect(page.locator('#hh-board .hh-hex.own-0, #hh-board .hh-hex.own-1')).toHaveCount(0);
  await expect(page.locator('#hh-board .hh-hex.claim')).toHaveCount(0);
  await page.locator('#hh-qBtns button.go').click();                      // Back to the board
  await expect(page.locator('#hh-q')).toBeHidden();
  await expect(page.locator('#hh-board .hh-hex.claim')).toHaveCount(1);   // now it pops, in the team's colour
  await expect(page.locator('#hh-board .hh-hex.claim.own-0, #hh-board .hh-hex.claim.own-1')).toHaveCount(1);
  await expect(page.locator('#hh-board .burst')).toHaveCount(1);
  expect(log.errors).toEqual([]);
});

test('Hex Hunt: the winning hexagon pops and the chain lights up after the question closes', async ({ page }) => {
  await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  // win every hexagon of one row for the first half with the keyboard until they join their edges
  for (let k = 0; k < 120; k++) {
    const s = await state(page, 'hex-hunt');
    if (s.step === 'winning') break;
    if (s.phase === 'board') { const i = s.owners.findIndex((o, j) => o < 0 && Math.floor(j / s.size) === Math.floor(s.size / 2)); await page.locator('#hh-board .hh-hex').nth(i).click(); continue; }
    if (s.round === 'think' || s.round === 'show') { await page.keyboard.press('Space'); await page.waitForTimeout(550); continue; }   // (a second Space within half a second is a repeat)
    if (s.round === 'mark') { await page.keyboard.press('1'); continue; }
    if (s.step === 'claimed') { await page.keyboard.press('Enter'); await page.waitForTimeout(550); continue; }
    await page.waitForTimeout(60);
  }
  expect((await state(page, 'hex-hunt')).step).toBe('winning');
  await expect(page.locator('#hh-board .hh-hex.chain, #hh-board .hh-hex.path')).toHaveCount(0);   // not behind the question
  await page.locator('#hh-qBtns button.go').click();                       // They've joined their edges!
  await expect(page.locator('#hh-q')).toBeHidden();
  await expect(page.locator('#hh-board .hh-hex.claim')).toHaveCount(1);
  expect(await page.locator('#hh-board .hh-hex.chain').count()).toBeGreaterThan(2);
});

test('Full screen: a button on the launcher and in every game, F toggles it', async ({ page }) => {
  const log = await openBundle(page);
  const on = () => page.evaluate(() => !!document.fullscreenElement);
  const lb = page.locator('#launcher [data-fullscreen]');
  await expect(lb).toHaveText(/^Full screen\s*F$/);
  await lb.click();
  await expect.poll(on).toBe(true);
  await expect(lb).toHaveText(/^Exit full screen\s*F$/);
  await lb.click();
  await expect.poll(on).toBe(false);
  const settle = () => page.waitForFunction(() => new Promise(r => { const h = innerHeight; setTimeout(() => r(innerHeight === h), 250); }));
  await settle();                                                            // the window has finished resizing
  for (const [g, p] of Object.entries(GAMES)) {
    await page.click(`[data-play="${g}"]`);
    const b = page.locator(`#game-${g} [data-fullscreen]`);
    await expect(b).toBeVisible();
    // it sits in the top bar with Menu and Sound
    expect(await b.evaluate((el, p) => !!el.parentElement.querySelector('#' + p + '-muteBtn'), p)).toBe(true);
    await page.click(`#${p}-startBtn`);
    await page.keyboard.press('f');
    await expect.poll(on).toBe(true);
    await expect(b).toHaveText(/^Exit full screen/);
    await page.keyboard.press('f');                                          // F leaves full screen (Esc is the game's own key there: tests/uifixes.spec.js)
    await expect.poll(on).toBe(false);
    await page.waitForTimeout(600);
    await expect(page.locator('#leaveModal')).toBeHidden();
    await expect(page.locator(`#game-${g}`)).toBeVisible();
    await page.keyboard.press('Escape');                                     // now Esc is the Menu key again
    await expect(page.locator('#leaveModal:not([hidden]), #launcher:not([hidden])')).toHaveCount(1);
    const leave = page.locator('#leaveConfirm'); if (await leave.isVisible()) await leave.click();
    await expect(page.locator('#launcher')).toBeVisible();
    await settle();
  }
  // F typed into a text box is just a letter
  await page.click(`[data-play="over-the-edge"]`);
  const names = page.locator('#game-over-the-edge details.tnames').first();
  await names.locator('summary').click();
  await page.locator('#ote-cname0').fill('');
  await page.locator('#ote-cname0').type('Foxes');
  expect(await on()).toBe(false);
  await expect(page.locator('#ote-cname0')).toHaveValue('Foxes');
  expect(log.errors).toEqual([]);
});

test('there is no answer countdown and no Answer time choice, in any subject; the Maths Final Sprint is 5 minutes', async ({ page }) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  const log = await openBundle(page);
  for (const sj of ['biology', 'maths']) {
    await page.selectOption('#subjectSelect', sj);
    for (const g of Object.keys(GAMES)) {
      await page.click(`[data-play="${g}"]`);
      await expect(page.locator(`#game-${g} .answer-time`)).toHaveCount(0);
      expect(await page.evaluate(() => CGB.answerSeconds())).toBe(0);
      await page.keyboard.press('Escape'); await expect(page.locator('#launcher')).toBeVisible();
    }
  }
  await page.click('[data-play="outpace"]');
  await page.click('#op-startBtn');
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  await expect(page.locator('#op-sprintTimer')).toHaveText('5:00');
  expect(log.errors).toEqual([]);
});
