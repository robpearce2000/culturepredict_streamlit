// Category Clash and Hex Hunt: navigation, full keyboard games, shared history.
const { test } = require('@playwright/test');
const { openBundle, state, expect } = require('./helpers');

async function playCategoryClash(page) {
  const end = Date.now() + 150000;
  let n = 0;
  while (Date.now() < end) {
    const s = await state(page, 'category-clash');
    if (s.phase === 'summary') return s;
    if (s.phase === 'board') await page.keyboard.press('Enter');
    else if (s.phase === 'question') {
      if (s.step === 'ask') await page.keyboard.press(n++ % 3 === 0 ? 'w' : 'c');
      else if (s.step === 'steal') await page.keyboard.press(n % 2 ? '2' : 'n');
      else if (s.step === 'done') await page.keyboard.press('Enter');
    }
    await page.waitForTimeout(60);
  }
  throw new Error('Category Clash did not finish');
}
async function playHexHunt(page) {
  const end = Date.now() + 150000;
  let n = 0;
  while (Date.now() < end) {
    const s = await state(page, 'hex-hunt');
    if (s.phase === 'summary') return s;
    if (s.phase === 'board') await page.keyboard.press('Enter');
    else if (s.phase === 'won') await page.keyboard.press('Enter');
    else if (s.phase === 'question') {
      if (s.step === 'buzz') await page.keyboard.press(n++ % 4 === 3 ? 'n' : (n % 2 ? '1' : '2'));
      else if (s.step === 'answer') await page.keyboard.press(n++ % 5 === 0 ? 'w' : 'c');
      else if (s.step === 'claimed' || s.step === 'nobody') await page.keyboard.press('Enter');
    }
    await page.waitForTimeout(60);
  }
  throw new Error('Hex Hunt did not finish');
}

test('launcher → Category Clash and Hex Hunt → back to launcher', async ({ page }) => {
  const log = await openBundle(page);
  await page.click('[data-play="category-clash"]');
  await expect(page.locator('#cc-home')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#launcher')).toBeVisible();
  await page.click('[data-play="hex-hunt"]');
  await expect(page.locator('#hh-home')).toBeVisible();
  await page.click('#hh-startBtn');
  await page.keyboard.press('Escape');
  await expect(page.locator('#leaveModal')).toBeVisible();
  await page.click('#leaveConfirm');
  await expect(page.locator('#launcher')).toBeVisible();
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('a full game of Category Clash with keyboard shortcuts reaches the results', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await page.click('#cc-segTeams button[data-v="3"]');
  await page.click('#cc-segCats button[data-v="4"]');
  await page.fill('#cc-name0', 'Owls'); await page.fill('#cc-name1', 'Foxes'); await page.fill('#cc-name2', 'Hawks');
  await page.locator('#cc-name2').press('Enter');
  await expect(page.locator('#cc-board .cc-tile')).toHaveCount(20);
  // harder rows are worth more: the 500 row holds questions at least as hard as the 100 row
  const s = await playCategoryClash(page);
  expect(s.left).toBe(0);
  await expect(page.locator('#cc-summary')).toBeVisible();
  await expect(page.locator('#cc-sumCard')).toContainText('Owls');
  const total = s.teams.reduce((a, t) => a + t.score, 0);
  expect(total).toBeGreaterThan(0);
  const games = await page.evaluate(() => ['owls', 'foxes', 'hawks'].flatMap(n => CGB.bank.wrongLog(n).map(e => e.game)));
  expect(games).toContain('Category Clash');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('a full match of Hex Hunt with keyboard shortcuts reaches the results', async ({ page }) => {
  const log = await openBundle(page, '#hex-hunt');
  await page.click('#hh-segBest button[data-v="3"]');
  await page.fill('#hh-name0', 'Reds'); await page.fill('#hh-name1', 'Blues');
  await page.locator('#hh-name1').press('Enter');
  await expect(page.locator('#hh-board .hh-hex')).toHaveCount(25);
  const s = await playHexHunt(page);
  expect(Math.max(...s.wins)).toBe(2);
  await expect(page.locator('#hh-summary')).toBeVisible();
  await expect(page.locator('#hh-sumCard')).toContainText('win the match');
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('Hex Hunt letters match the first letter of each answer', async ({ page }) => {
  await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  for (let i = 0; i < 6; i++) {
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Enter');
    const s = await state(page, 'hex-hunt');
    if (s.phase !== 'question') continue;
    const a = s.q.a.replace(/^(the|a|an)\s+/i, '');
    expect(a[0].toUpperCase()).toBe(s.letter);
    await page.keyboard.press('1'); await page.keyboard.press('c'); await page.keyboard.press('Enter');
    if ((await state(page, 'hex-hunt')).phase !== 'board') break;
  }
});

test('Difficulty lines order Category Clash values', async ({ page }) => {
  await openBundle(page);
  await page.evaluate(() => {
    const lines = ['Subject: Test', 'Topic: Graded'];
    [5, 4, 3, 2, 1].forEach(d => { lines.push('Difficulty: ' + d, `Q: Level ${d} question?`, `A: Answer ${d}`); });
    CGB.bank.addSet('Graded set', lines.join('\n'));
  });
  await page.click('[data-play="category-clash"]');
  await page.click('#cc-startBtn');
  for (let r = 0; r < 5; r++) {
    await page.keyboard.press('Enter');
    const s = await state(page, 'category-clash');
    expect(s.q.q).toBe(`Level ${r + 1} question?`);
    await page.keyboard.press('c'); await page.keyboard.press('Enter');
  }
});

test('Category Clash fills a row from the nearest difficulty when a level is missing', async ({ page }) => {
  await openBundle(page);
  await page.evaluate(() => {
    const lines = ['Subject: Test', 'Topic: Lopsided', 'Difficulty: 1'];
    for (let i = 1; i <= 4; i++) lines.push(`Q: Easy ${i}?`, `A: E${i}`);
    lines.push('Difficulty: 5', 'Q: Hard one?', 'A: H');
    CGB.bank.addSet('Lopsided set', lines.join('\n'));
  });
  await page.click('[data-play="category-clash"]');
  await page.click('#cc-startBtn');
  const tiles = await page.evaluate(() => CGB.games['category-clash'].board().map(c => c.tiles.map(t => t.q && t.q.q)));
  expect(tiles[0].filter(Boolean).length).toBe(5);
  expect(tiles[0][4]).toBe('Hard one?');
  expect(tiles[0][0]).toMatch(/^Easy/);
});

test('every built-in question has a difficulty from 1 to 5, and each topic spans the range', async ({ page }) => {
  await openBundle(page);
  const r = await page.evaluate(() => {
    const out = { untagged: 0, narrow: [] };
    CGB.bank.all().filter(s => s.builtin).forEach(s => {
      const topics = {};
      s.questions.forEach(q => { if (!(q.level >= 1 && q.level <= 5)) out.untagged++; (topics[q.topic] = topics[q.topic] || new Set()).add(q.level); });
      Object.entries(topics).forEach(([t, l]) => { if (l.size < 5) out.narrow.push(s.id + ': ' + t); });
    });
    return out;
  });
  expect(r.untagged).toBe(0);
  expect(r.narrow).toEqual([]);
});
