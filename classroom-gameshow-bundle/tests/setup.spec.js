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
  await expect(page.locator('#subjectSelect option:checked')).toHaveText(/Combined Science/);
  await expect(page.locator('#boardSelect option:checked')).toHaveText(/AQA/);
  const combined = await page.locator('#launcherSet option').allTextContents();
  expect(combined.length).toBe(5);
  // Biology: only the Biology packs
  await page.selectOption('#subjectSelect', 'biology');
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
  await expect(page.locator('#subjectSelect option:checked')).toHaveText(/Biology/);
  expect(await page.evaluate(() => CGB.bank.active().id)).toBe('homeostasis-l1');
  // other games can read the choice (Outpace and Category Clash will follow it)
  expect(await page.evaluate(() => [CGB.bank.subject(), CGB.bank.board()])).toEqual(['biology', 'aqa']);
  expect(log.errors).toEqual([]);
});

test('a subject with no built-in pack says so, holds the Start buttons, and takes the teacher\'s own set', async ({ page: first }) => {
  let page = first;
  const log = await openBundle(page);
  await page.selectOption('#subjectSelect', 'maths');
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
    await expect(page.locator('#subjectSelect')).toBeFocused();
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
  await page.selectOption('#subjectSelect', 'physics');
  await expect(page.locator('#launcherSet option', { hasText: 'Fractions' })).toHaveCount(0);
  // and can be moved to another subject from the Question bank
  await page.selectOption('#subjectSelect', 'maths');
  await page.click('#openBank');
  await page.selectOption('#bankList select[data-subj]', 'physics');
  await expect(page.locator('#bankList .bank-item')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await page.selectOption('#subjectSelect', 'physics');
  await expect(page.locator('#launcherSet option', { hasText: 'Fractions' })).toHaveCount(1);
  expect(log.errors).toEqual([]);
});

test('the exam board choice is shown, remembered, and notes that the built-in packs are AQA-style', async ({ page: first }) => {
  let page = first;
  await openBundle(page);
  await page.selectOption('#boardSelect', 'edexcel');
  await expect(page.locator('#boardSelect option:checked')).toHaveText(/Edexcel/);
  await expect(page.locator('#subjectNote')).toContainText('AQA-style. Edexcel packs are coming soon');
  expect(await page.locator('#launcherSet option').count()).toBe(5);   // nothing hidden until Edexcel packs exist
  await openGame(page, 'hex-hunt');
  await expect(page.locator('#hh-pack')).toContainText('Combined Science · Edexcel');
  page = await reopen(page);
  await expect(page.locator('#boardSelect option:checked')).toHaveText(/Edexcel/);
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
    await page.selectOption('#subjectSelect', sj);
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
  await expect(page.locator('#mascotWho')).toHaveText('Ms Quiz');                 // the name label appears once he has a name
  await expect(page.locator('#mascotLine')).toHaveText('Choose your subject, then pick a game!');
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

test('every game waits on a Start game button after setup, with no countdown until it is pressed', async ({ page }) => {
  const log = await openBundle(page, '', { gate: true });
  const after = { 'over-the-edge': s => s.round === 'think', outpace: s => s.phase === 'deal', 'category-clash': s => s.phase === 'board', 'hex-hunt': s => s.phase === 'board' };
  for (const id of GAMES) {
    await openGame(page, id);
    await page.keyboard.press('Enter');                           // Enter on the setup card starts the game...
    const gate = page.locator(`#game-${id} .start-gate`);
    await expect(gate).toBeVisible();                             // ...and the same Start game button appears in every game
    await expect(gate.locator('.start-gate-btn')).toHaveText(/^Start game\s*Enter$/);
    await expect(card(page, id)).toBeHidden();
    await page.waitForTimeout(1500);
    const s = await state(page, id);
    expect(after[id](s)).toBe(false);                             // nothing has started, so nothing is ticking
    expect(s.round === 'think').toBe(false);
    await expect(page.locator(`#game-${id} .cm-count:visible, #game-${id} [id$="count"]:visible`)).toHaveCount(0);
    if (id === 'hex-hunt') await gate.locator('.start-gate-btn').click(); else await page.keyboard.press('Enter');
    await expect(gate).toBeHidden();
    await expect.poll(async () => after[id](await state(page, id))).toBe(true);
    await page.keyboard.press('Escape'); const leave = page.locator('#leaveConfirm'); if (await leave.isVisible()) await leave.click();
    await expect(page.locator('#launcher')).toBeVisible();
  }
  expect(log.errors).toEqual([]);
});

test('the Over the Edge camera holds still while the setup card is showing', async ({ page }) => {
  await openBundle(page);
  await openGame(page, 'over-the-edge');
  await expect(card(page, 'over-the-edge')).toBeVisible();
  await page.waitForTimeout(3000);                                // let it settle on the setup view
  const seen = [];
  for (let i = 0; i < 8; i++) { seen.push((await state(page, 'over-the-edge')).camera); await page.waitForTimeout(500); }
  // through four seconds of decorative drops, the camera does not move
  seen.forEach(p => p.forEach((v, k) => expect(Math.abs(v - seen[0][k])).toBeLessThan(0.002)));
});

test('the host has no name unless the teacher gives him one, and clearing it makes him nameless again', async ({ page: first }) => {
  let page = first;
  const log = await openBundle(page);
  // nameless by default: the greeting doesn't introduce him and there is no name label
  await expect(page.locator('#mascotLine')).toHaveText('Choose your subject, then pick a game!');
  await expect(page.locator('#mascotWho')).toBeHidden();
  await expect(page.locator('#mascotHello')).not.toContainText("I'm");
  // no name label in Over the Edge's bubble or the corner host of the other games
  await openGame(page, 'over-the-edge');
  await expect(page.locator('#ote-bubble')).toHaveClass(/show/, { timeout: 8000 });
  await expect(page.locator('#ote-bubbleWho')).toBeHidden();
  await expect(page.locator('#ote-bubbleText')).not.toContainText("I'm");
  await page.keyboard.press('Escape');
  await openGame(page, 'category-clash');
  await page.click('#cc-startBtn');
  await expect(page.locator('#game-category-clash .hc-bubble')).toHaveClass(/show/);
  await expect(page.locator('#game-category-clash .hc-name')).toBeHidden();
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  // the customiser shows an empty name box
  await page.click('#openHost');
  await expect(page.locator('#hostName')).toHaveValue('');
  // a name: shown as the label on his bubbles
  await page.fill('#hostName', 'Mr Pearce');
  await page.keyboard.press('Escape');
  await expect(page.locator('#mascotWho')).toHaveText('Mr Pearce');
  await openGame(page, 'hex-hunt');
  await page.click('#hh-startBtn');
  await expect(page.locator('#game-hex-hunt .hc-name')).toHaveText('Mr Pearce');
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  // cleared: nameless again
  await page.click('#openHost');
  await page.fill('#hostName', '');
  await page.keyboard.press('Escape');
  await expect(page.locator('#mascotWho')).toBeHidden();
  // a name saved by an earlier version that is just the old default counts as no name
  await page.evaluate(() => localStorage.setItem('cgb.host', JSON.stringify({ name: ['Marty', 'Marquee'].join(' '), suit: 1 })));
  page = await reopen(page);
  expect(await page.evaluate(() => CGB.hostName())).toBe('');
  await expect(page.locator('#mascotWho')).toBeHidden();
  // and no default name appears anywhere in the shipped file
  const fs = require('fs');
  const shipped = fs.readFileSync(require('./helpers').DIST, 'utf8');
  expect(shipped).not.toMatch(/Marty|Marquee|Professor Pip/);
  expect(log.errors).toEqual([]);
});

test('launcher: subject and exam board dropdowns, the bubble clear of the host, and four equal Play buttons', async ({ page }) => {
  const log = await openBundle(page);
  for (const [w, h] of [[1920, 1080], [1366, 768], [768, 1024]]) {
    await page.setViewportSize({ width: w, height: h });
    await page.waitForTimeout(300);
    // the speech bubble never overlaps the host
    const [bub, fig] = await Promise.all([page.locator('#mascotHello').boundingBox(), page.locator('#mascot .host-wrap').boundingBox()]);
    const svg = await page.locator('#mascot .host-wrap svg').boundingBox();
    const overlap = (a, b) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
    expect(overlap(bub, fig), `bubble over host at ${w}x${h}`).toBe(false);
    if (svg) expect(overlap(bub, svg), `bubble over host drawing at ${w}x${h}`).toBe(false);
    // the four Play buttons say Play and are exactly the same size
    const play = page.locator('.gcard [data-play]');
    await expect(play).toHaveText(['Play', 'Play', 'Play', 'Play']);
    const boxes = await play.evaluateAll(bs => bs.map(b => { const r = b.getBoundingClientRect(); return [Math.round(r.width * 10), Math.round(r.height * 10)]; }));
    boxes.forEach(b => expect(b).toEqual(boxes[0]));
    // the wordmark sits in the middle and the subject panel to its left on wide screens
    if (w >= 1366) {
      const [word, subj] = await Promise.all([page.locator('#wordmark').boundingBox(), page.locator('#subjectPanel').boundingBox()]);
      expect(Math.abs(word.x + word.width / 2 - w / 2)).toBeLessThan(w * 0.08);
      expect(subj.x + subj.width).toBeLessThanOrEqual(word.x + 1);
    }
  }
  // the subject picker is a compact dropdown showing the chosen subject: keyboard...
  await expect(page.locator('#subjectSelect option:checked')).toHaveText('Combined Science');
  await page.focus('#subjectSelect');
  await page.keyboard.press('ArrowDown');
  await expect.poll(() => page.evaluate(() => CGB.bank.subject())).not.toBe('combined');
  // ...and mouse or touch
  await page.selectOption('#subjectSelect', 'physics');
  expect(await page.evaluate(() => CGB.bank.subject())).toBe('physics');
  await page.selectOption('#boardSelect', 'edexcel');
  expect(await page.evaluate(() => CGB.bank.board())).toBe('edexcel');
  // the sign's bulbs twinkle on varied timings, and are still with reduced motion
  const anim = await page.locator('#wordmark .bulb').evaluateAll(bs => bs.map(b => getComputedStyle(b).animationName + ' ' + getComputedStyle(b).animationDuration + ' ' + getComputedStyle(b).animationDelay));
  expect(anim.length).toBeGreaterThan(20);
  expect(anim.every(a => a.startsWith('bulb-twinkle'))).toBe(true);
  expect(new Set(anim).size).toBeGreaterThan(10);
  await page.evaluate(() => CGB.settings.set('reducedMotion', true));
  const still = await page.locator('#wordmark .bulb').evaluateAll(bs => bs.map(b => getComputedStyle(b).animationName + ' ' + getComputedStyle(b).opacity));
  expect(still.every(a => a === 'none 1')).toBe(true);
  expect(log.errors).toEqual([]);
});
