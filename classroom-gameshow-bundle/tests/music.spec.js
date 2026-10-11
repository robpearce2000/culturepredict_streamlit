// Launcher music: starts only after a first interaction, follows Sound and Music, fades out for a game, back on return.
const { test, expect } = require('@playwright/test');
const { openBundle } = require('./helpers');

test.describe.configure({ retries: 1 });
const info = page => page.evaluate(() => CGB.music.test.info());
const settle = (page, ms) => page.waitForTimeout(ms);

test('music: no audio errors on load, silent until the first click, then plays; no network', async ({ page }) => {
  const log = await openBundle(page);
  await settle(page, 500);
  let m = await info(page);
  expect(m.playing).toBe(false); expect(m.gestured).toBe(false); expect(m.gain).toBe(0);
  await page.mouse.click(5, 5);
  await expect.poll(async () => (await info(page)).playing, { timeout: 15000 }).toBe(true);
  await expect.poll(async () => (await info(page)).gain, { timeout: 8000 }).toBeGreaterThan(0.1);
  m = await info(page);
  expect(m.gain).toBeLessThan(0.3);                                   // gentle
  expect(m.loopSeconds).toBeGreaterThan(48); expect(m.loopSeconds).toBeLessThan(54);
  expect(log.errors).toEqual([]); expect(log.requests).toEqual([]);
});

test('music: the loop is exactly 24 bars long, so it repeats with no gap', async ({ page }) => {
  await openBundle(page);
  await page.keyboard.press('Shift');
  await expect.poll(async () => (await info(page)).loaded, { timeout: 15000 }).toBe(true);
  const m = await info(page);
  expect(Math.abs(m.loopSeconds - 24 * 240 / 112)).toBeLessThan(0.01);   // 51.43 s at 112 bpm, to within 10 ms
});

test('music: Music off and Sound off both silence it at once, and back on resumes', async ({ page }) => {
  await openBundle(page);
  await page.mouse.click(5, 5);
  await expect.poll(async () => (await info(page)).playing, { timeout: 15000 }).toBe(true);
  await page.evaluate(() => CGB.settings.set('music', false));
  await expect.poll(async () => (await info(page)).playing).toBe(false);
  expect((await info(page)).gain).toBe(0);
  await page.evaluate(() => CGB.settings.set('music', true));
  await expect.poll(async () => (await info(page)).playing, { timeout: 8000 }).toBe(true);
  await page.evaluate(() => CGB.settings.set('sound', false));
  await expect.poll(async () => (await info(page)).playing).toBe(false);
  expect((await info(page)).gain).toBe(0);
  await page.evaluate(() => CGB.settings.set('sound', true));
  await expect.poll(async () => (await info(page)).playing, { timeout: 8000 }).toBe(true);
});

test('music: the launcher has a Music On/Off control next to Sound, remembered between visits', async ({ page }) => {
  await openBundle(page);
  const seg = page.locator('#launcherSettings .seg[data-key="music"]');
  await expect(seg).toBeVisible();
  await expect(page.locator('#launcherSettings .seg[data-key="sound"]')).toBeVisible();
  await seg.locator('button', { hasText: 'Off' }).click();
  expect(await page.evaluate(() => CGB.settings.get('music'))).toBe(false);
  await page.reload();
  await expect(page.locator('#launcherSettings .seg[data-key="music"] button[aria-pressed="true"]')).toHaveText('Off');
});

test('music: choosing a game plays the fanfare and fades the music out; returning fades it back in', async ({ page }) => {
  await openBundle(page);
  await page.mouse.click(5, 5);
  await expect.poll(async () => (await info(page)).gain, { timeout: 15000 }).toBeGreaterThan(0.1);
  await page.click('[data-play="hex-hunt"]');
  await expect.poll(async () => (await info(page)).fanfares).toBe(1);
  await expect.poll(async () => (await info(page)).playing, { timeout: 5000 }).toBe(false);   // gone after about a second
  expect((await info(page)).gain).toBeLessThan(0.02);
  await page.waitForTimeout(1500);
  expect((await info(page)).playing).toBe(false);                                           // silent in the game
  await page.evaluate(() => CGB.app.show('launcher'));
  await expect.poll(async () => (await info(page)).playing, { timeout: 8000 }).toBe(true);
  await expect.poll(async () => (await info(page)).gain, { timeout: 8000 }).toBeGreaterThan(0.1);
});

test('music: no music or fanfare when a game is opened straight from its link, and Sound off means no fanfare', async ({ page }) => {
  await openBundle(page, '#outpace');
  await page.mouse.click(5, 5);
  await page.waitForTimeout(800);
  expect((await info(page)).playing).toBe(false);
  expect((await info(page)).fanfares).toBe(0);
  await page.evaluate(() => { CGB.app.show('launcher'); CGB.settings.set('sound', false); });
  await page.click('[data-play="category-clash"]');
  await page.waitForTimeout(500);
  expect((await info(page)).fanfares).toBe(0);
});

test('music: the shipped file has it too, with no test code', async ({ page }) => {
  const log = await openBundle(page, undefined, { product: true });
  expect(await page.evaluate(() => typeof CGB.music.test)).toBe('undefined');
  expect(await page.evaluate(() => !!CGB.MUSIC_DATA && CGB.MUSIC_DATA.loop.length > 100000)).toBe(true);
  expect(log.errors).toEqual([]);
});

test('music: pauses while the tab is hidden and resumes when it is back', async ({ page }) => {
  await openBundle(page);
  await page.mouse.click(5, 5);
  await expect.poll(async () => (await info(page)).playing, { timeout: 15000 }).toBe(true);
  const hide = v => page.evaluate(h => { Object.defineProperty(document, 'hidden', { configurable: true, get: () => h }); document.dispatchEvent(new Event('visibilitychange')); }, v);
  await hide(true);
  await expect.poll(async () => (await info(page)).playing).toBe(false);
  await hide(false);
  await expect.poll(async () => (await info(page)).playing, { timeout: 8000 }).toBe(true);
});
