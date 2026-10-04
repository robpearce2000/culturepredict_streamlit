// Core behaviour of the shipped single file.
const fs = require('fs');
const { test } = require('@playwright/test');
const { openBundle, state, playOverTheEdge, playOutpace, setNames, expect } = require('./helpers');

test('loads from file:// with no network requests and no console errors', async ({ page }) => {
  const log = await openBundle(page);
  await expect(page.locator('#launcher')).toBeVisible();
  await expect(page.locator('#wordmark svg')).toBeVisible();
  await expect(page.locator('[data-play="over-the-edge"]')).toBeVisible();
  await expect(page.locator('[data-play="outpace"]')).toBeVisible();
  await expect(page.getByText('More shows coming soon')).toBeVisible();
  // open every panel on the launcher too
  await page.click('#openBank'); await expect(page.locator('#bankModal')).toBeVisible(); await page.keyboard.press('Escape');
  await page.click('#openAbout'); await expect(page.locator('#aboutModal')).toContainText('MIT'); await page.keyboard.press('Escape');
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

test('launcher → each game → back to launcher', async ({ page }) => {
  const log = await openBundle(page);
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
  await page.click('#ote-segR1 button[data-v="4"]');
  await page.click('#ote-segF button[data-v="8"]');
  await setNames(page, 'ote', 'Ada', 'Ben');
  await page.locator('#ote-name1').press('Enter');           // Enter starts the game
  await expect(page.locator('#ote-home')).toBeHidden();
  const s = await playOverTheEdge(page, { correct: 3 });
  expect(s.phase).toBe('summary');
  await expect(page.locator('#ote-summary')).toBeVisible();
  await expect(page.locator('#ote-sumCard')).toContainText('Ada');
  await expect(page.locator('#ote-sumCard')).toContainText('Weakest topics');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('a full game of Outpace with keyboard shortcuts reaches the summary', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await setNames(page, 'op', 'Cara', 'Dev');
  await page.locator('#op-name1').press('Enter');
  await expect(page.locator('#op-hud-deal')).toBeVisible();
  const s = await playOutpace(page);      // includes the full 60-second Final Sprint
  expect(s.phase).toBe('summary');
  await expect(page.locator('#op-summary')).toBeVisible();
  await expect(page.locator('#op-summaryPlayers')).toContainText('Cara');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('Outpace: C marks a question straight away and shows the answer', async ({ page }) => {
  await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  await page.keyboard.press('2');
  await expect(page.locator('#op-qcard')).toBeVisible();
  await expect(page.locator('#op-dealBtnRow [data-a="correct"]')).toBeVisible();
  await expect(page.locator('#op-dealQAnswer')).toBeHidden();
  await page.keyboard.press('c');
  await expect(page.locator('#op-dealQAnswer')).toBeVisible();
  await expect(page.locator('#op-dealBtnRow .op-marked.ok')).toBeVisible();
  expect((await state(page, 'outpace')).awaitingNext).toBe(true);
});
