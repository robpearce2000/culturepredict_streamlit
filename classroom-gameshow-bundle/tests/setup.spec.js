// The main screen and the setup cards: subject and exam board first, then a game whose only
// choice is the number of teams; the host is customised from the main screen.
const { test } = require('@playwright/test');
const { openBundle, state, mark, setNames, PICK, expect } = require('./helpers');

const GAMES = ['over-the-edge', 'outpace', 'category-clash', 'hex-hunt'];
const card = (page, id) => page.locator(`#game-${id} .setup`);
// the first visit to a 3D game builds its scene, which takes a while on the software renderer
const openGame = (page, id) => page.click(`[data-play="${id}"]`, { timeout: 60000 });
/* "Close the file and open it again". Showtime keeps everything in localStorage, so a new browser
   session that starts with exactly what this page stored is the same as reopening the file. (Two
   tabs on a file:// page don't reliably share storage in Chromium straight away, and reloading a
   tab straight after a save can drop it, so neither is used here.) */
async function reopen(page) {
  const saved = await page.evaluate(() => Object.entries(localStorage));
  const url = page.url().replace(/#.*$/, '');
  const ctx = await page.context().browser().newContext({ viewport: page.viewportSize() });
  await ctx.addInitScript(entries => { if (!sessionStorage.getItem('seeded')) { localStorage.clear(); entries.forEach(([k, v]) => localStorage.setItem(k, v)); sessionStorage.setItem('seeded', '1'); } }, saved);
  const fresh = await ctx.newPage();
  await fresh.goto(url);
  await fresh.waitForFunction(() => window.CGB && CGB.app);
  return fresh;
}

test('each setup card offers only the number of teams, with names tucked away and the questions shown', async ({ page: first }) => {
  let page = first;
  const log = await openBundle(page);
  for (const id of GAMES) {
    await openGame(page, id);
    const c = card(page, id);
    await expect(c).toBeVisible();
    // no other choices: no dropdowns, checkboxes, mode toggle, timer, session or difficulty
    await expect(c.locator('select')).toHaveCount(0);
    await expect(c.locator('input[type="checkbox"]')).toHaveCount(0);
    await expect(c.locator('.seg')).toHaveCount(id === 'hex-hunt' ? 0 : 1);
    if (id !== 'hex-hunt') await expect(c.locator('.seg button')).toHaveText(['2', '3', '4', '5', '6']);
    else await expect(c).toContainText('Two halves of the class');
    await expect(c).not.toContainText(/Whole class|Small group|Session length|Countdown|Jackpot difficulty|Customise the host|got wrong before/);
    // team names are behind "Edit team names"
    const names = c.locator('details.tnames').first();
    await expect(names.locator('input').first()).toBeHidden();
    await names.locator('summary').click();
    await expect(names.locator('input').first()).toBeVisible();
    // the subject and set in use, with a way back to change them
    await expect(c.locator('.packline')).toContainText('Combined Science: mixed');
    await expect(c.locator('.packline')).toContainText('Combined Science · AQA');
    await page.keyboard.press('Escape');
    await expect(page.locator('#launcher')).toBeVisible();
  }
  expect(log.errors).toEqual([]);
});

test('the number of teams and team names are remembered, and two teams works as a pair', async ({ page: first }) => {
  let page = first;
  await openBundle(page, '#category-clash');
  await page.click(PICK('#cc-segTeams', 2));
  await setNames(page, 'category-clash', ['Pip', 'Sam']);
  await page.click('#cc-startBtn');
  let s = await state(page, 'category-clash');
  expect(s.teams.map(t => t.name)).toEqual(['Pip', 'Sam']);
  // a pair plays exactly like a class: the chooser full points, the other half
  await page.keyboard.press('Enter');
  await mark(page, 'category-clash', () => true);
  s = await state(page, 'category-clash');
  expect(s.teams.map(t => t.score).sort((a, b) => a - b)).toEqual([50, 100]);
  page = await reopen(page);
  await openGame(page, 'category-clash');
  await expect(page.locator(`${PICK('#cc-segTeams', 2)}[aria-pressed="true"]`)).toHaveCount(1);
  // names are shared by every game
  await page.click('#cc-menuBtn');
  await openGame(page, 'over-the-edge');
  await page.click(PICK('#ote-segTeams', 2));
  await page.locator('#game-over-the-edge details.tnames summary').click();
  await expect(page.locator('#ote-cname0')).toHaveValue('Pip');
});

test('choosing a subject filters the question sets everywhere and is remembered', async ({ page: first }) => {
  let page = first;
  const log = await openBundle(page);
  // default: Combined Science, AQA
  await expect(page.locator('#subjectChips .chip.on')).toHaveText(/Combined Science/);
  await expect(page.locator('#boardSeg [aria-pressed="true"]')).toHaveText(/AQA/);
  const combined = await page.locator('#launcherSet option').allTextContents();
  expect(combined.length).toBe(5);
  // Biology: only the Biology packs
  await page.click('#subjectChips [data-subject="biology"]');
  const bio = await page.locator('#launcherSet option').allTextContents();
  expect(bio.map(t => t.replace(/ \(\d+\)$/, ''))).toEqual(['Combined Science: Biology', 'Homeostasis L1']);
  await page.selectOption('#launcherSet', { label: await page.locator('#launcherSet option').nth(1).textContent() });
  await page.click('#openBank');
  await expect(page.locator('#bankList .bank-item')).toHaveCount(2);
  await expect(page.locator('#bankIntro')).toContainText('Biology sets (AQA)');
  await page.keyboard.press('Escape');
  // a game uses it, and its questions come from it
  await openGame(page, 'category-clash');
  await expect(page.locator('#cc-pack')).toContainText('Homeostasis L1');
  await expect(page.locator('#cc-pack')).toContainText('Biology · AQA');
  await page.click('#cc-startBtn');
  await page.keyboard.press('Enter');
  const q = (await state(page, 'category-clash')).q;
  const inSet = await page.evaluate(q => CGB.bank.get('homeostasis-l1').questions.some(x => x.q === q.q), q);
  expect(inSet).toBe(true);
  // remembered after closing the page, along with the set chosen for that subject
  page = await reopen(page);
  await expect(page.locator('#subjectChips .chip.on')).toHaveText(/Biology/);
  expect(await page.evaluate(() => CGB.bank.active().id)).toBe('homeostasis-l1');
  // other games can read the choice (Outpace and Category Clash will follow it)
  expect(await page.evaluate(() => [CGB.bank.subject(), CGB.bank.board()])).toEqual(['biology', 'aqa']);
  expect(log.errors).toEqual([]);
});

test('a subject with no built-in pack says so, holds the Start buttons, and takes the teacher\'s own set', async ({ page: first }) => {
  let page = first;
  const log = await openBundle(page);
  await page.click('#subjectChips [data-subject="maths"]');
  await expect(page.locator('#subjectNote')).toContainText('Built-in Maths packs are coming soon');
  await expect(page.locator('#launcherSet')).toBeDisabled();
  for (const id of GAMES) {
    await openGame(page, id);
    await expect(card(page, id).locator('.packline')).toContainText('No Maths questions yet');
    await expect(card(page, id).locator('.setup-go .btn')).toBeDisabled();
    await page.keyboard.press('Enter');                       // Enter does not start it either
    expect((await state(page, id)).phase).toBe('home');
    // "Change" goes back to the subject choice
    await card(page, id).locator('.packline [data-pack]').click();
    await expect(page.locator('#launcher')).toBeVisible();
    await expect(page.locator('#subjectChips .chip.on')).toBeFocused();
  }
  // the teacher adds a Maths set: the subject is filled in for them
  await page.click('#openBank');
  await expect(page.locator('#bankSubject')).toHaveValue('maths');
  await page.fill('#bankName', 'Fractions');
  await page.fill('#bankText', 'Subject: Maths\nTopic: Fractions\nQ: What is half of three quarters?\nA: Three eighths\nQ: Simplify 6/8\nA: 3/4');
  await page.click('#bankSave');
  await expect(page.locator('#bankFormStatus')).toContainText('in every game for Maths');
  await page.keyboard.press('Escape');
  await expect(page.locator('#subjectNote')).toContainText('coming soon');   // still no built-in pack
  await expect(page.locator('#launcherSet')).toBeEnabled();
  await openGame(page, 'over-the-edge');
  await expect(page.locator('#ote-startBtn')).toBeEnabled();
  await page.click('#ote-startBtn');
  expect((await state(page, 'over-the-edge')).q.topic).toBe('Fractions');
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  // it only shows under Maths
  await page.click('#subjectChips [data-subject="physics"]');
  await expect(page.locator('#launcherSet option', { hasText: 'Fractions' })).toHaveCount(0);
  // and can be moved to another subject from the Question bank
  await page.click('#subjectChips [data-subject="maths"]');
  await page.click('#openBank');
  await page.selectOption('#bankList select[data-subj]', 'physics');
  await expect(page.locator('#bankList .bank-item')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await page.click('#subjectChips [data-subject="physics"]');
  await expect(page.locator('#launcherSet option', { hasText: 'Fractions' })).toHaveCount(1);
  expect(log.errors).toEqual([]);
});

test('the exam board choice is shown, remembered, and notes that the built-in packs are AQA-style', async ({ page: first }) => {
  let page = first;
  await openBundle(page);
  await page.click('#boardSeg [data-board="edexcel"]');
  await expect(page.locator('#boardSeg [aria-pressed="true"]')).toHaveText(/Edexcel/);
  await expect(page.locator('#subjectNote')).toContainText('AQA-style. Edexcel packs are coming soon');
  expect(await page.locator('#launcherSet option').count()).toBe(5);   // nothing hidden until Edexcel packs exist
  await openGame(page, 'hex-hunt');
  await expect(page.locator('#hh-pack')).toContainText('Combined Science · Edexcel');
  page = await reopen(page);
  await expect(page.locator('#boardSeg [aria-pressed="true"]')).toHaveText(/Edexcel/);
});

test('sets saved before subjects existed show under every subject', async ({ page: first }) => {
  let page = first;
  await openBundle(page);
  await page.evaluate(() => {
    localStorage.setItem('cgb.sets', JSON.stringify([{ id: 'set-old', name: 'Old set', questions: [{ subject: 'Biology', topic: 'Cells', q: 'Q?', a: 'A' }] }]));
    localStorage.setItem('cgb.activeSet', 'set-old');
  });
  page = await reopen(page);
  expect(await page.evaluate(() => CGB.bank.active().id)).toBe('set-old');   // the set in use is kept
  for (const sj of ['history', 'chemistry']) {
    await page.click(`#subjectChips [data-subject="${sj}"]`);
    await expect(page.locator('#launcherSet option', { hasText: 'Old set' })).toHaveCount(1);
  }
});

test('the host is customised from the main screen and changes everywhere', async ({ page: first }) => {
  let page = first;
  const log = await openBundle(page);
  await page.click('#openHost');
  await expect(page.locator('#hostModal')).toBeVisible();
  await expect(page.locator('#hostPreview svg')).toBeVisible();
  await page.fill('#hostName', 'Ms Quiz');
  await page.locator('#hostName').press('Tab');
  await page.click('#swSuit .sw[data-i="3"]');
  await page.click('#segGlasses button[data-v="no"]');
  await expect(page.locator('#swSuit .sw[data-i="3"]')).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('Escape');
  await expect(page.locator('#mascotHello')).toContainText("Hello, I'm Ms Quiz!");
  // gone from Over the Edge's setup; the game's host uses the new name
  await openGame(page, 'over-the-edge');
  await expect(page.locator('#game-over-the-edge')).not.toContainText('Customise the host');
  await expect(page.locator('#ote-bubbleWho')).toHaveText('Ms Quiz', { timeout: 5000 });
  await page.keyboard.press('Escape');
  // remembered
  page = await reopen(page);
  const cfg = await page.evaluate(() => CGB.hostCfg);
  expect(cfg.name).toBe('Ms Quiz'); expect(cfg.suit).toBe(3); expect(cfg.glasses).toBe('no');
  expect(log.errors).toEqual([]);
});
