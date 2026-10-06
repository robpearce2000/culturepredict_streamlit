// Category Clash and Hex Hunt: navigation, full keyboard games, shared history.
const { test } = require('@playwright/test');
const { openBundle, state, mark, playCategoryClash, playHexHunt, setNames, PICK, expect } = require('./helpers');

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

// Every button after marking must work with the mouse too, not only its keyboard shortcut
for (const [g, tile] of [['category-clash', '#cc-board .cc-tile[data-c]'], ['hex-hunt', '#hh-board .hh-hex']]) {
  test(`${g}: one question played with the mouse only goes back to the board`, async ({ page }) => {
    const log = await openBundle(page, '#' + g);
    const p = g === 'category-clash' ? 'cc' : 'hh';
    await page.click(`#${p}-startBtn`);
    await page.locator(tile).first().click();
    await page.locator(`#game-${g} [data-cm="show"]`).click();
    await expect.poll(async () => (await state(page, g)).round, { timeout: 10000 }).toBe('mark');
    if (g === 'category-clash') { await page.locator('#game-category-clash .cm-team').first().click(); await page.locator('#game-category-clash [data-cm="confirm"]').click(); }
    else await page.locator('#game-hex-hunt .hh-side').first().click();
    const next = page.locator(`#${p}-qBtns button.go`);
    await expect(next).toHaveText(/Back to the board|joined their edges/);
    await next.click();
    await expect.poll(async () => (await state(page, g)).phase).toBe('board');
    expect(log.errors).toEqual([]);
  });
}

test('a full game of Category Clash with keyboard shortcuts reaches the results', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await page.click(PICK('#cc-segTeams', 3));
  await setNames(page, 'category-clash', ['Owls', 'Foxes', 'Hawks']);
  await page.locator('#cc-name2').press('Enter');
  await expect(page.locator('#cc-board .cc-tile')).toHaveCount(12);    // four categories of three
  await expect(page.locator('#cc-board .cc-tile:not(.empty)')).toHaveCount(12);
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

test('a full round of Hex Hunt with keyboard shortcuts reaches the results', async ({ page }) => {
  const log = await openBundle(page, '#hex-hunt');
  await setNames(page, 'hex-hunt', ['Reds', 'Blues']);
  await page.locator('#hh-name1').press('Enter');
  await expect(page.locator('#hh-board .hh-hex')).toHaveCount(16);      // one round on a 4 × 4 board
  const s = await playHexHunt(page);
  expect(s.winner).toBeGreaterThanOrEqual(0);
  await expect(page.locator('#hh-summary')).toBeVisible();
  await expect(page.locator('#hh-sumCard')).toContainText(/(Reds|Blues) win!/);
  expect(log.errors).toEqual([]);
  expect(log.requests).toEqual([]);
});

test('Hex Hunt letters really are where each answer starts, for every question that can go on the board', async ({ page }) => {
  await openBundle(page, '#hex-hunt');
  const r = await page.evaluate(() => {
    const out = { bad: [], used: 0, perSet: {} };
    const STOP = /^(it|its|it's|they|them|their|this|these|those|there|to|so|when|because|by|one|if|any|yes|no|true|false|about|roughly|around)$/i;
    CGB.bank.all().filter(s => s.builtin).forEach(s => {
      let n = 0;
      s.questions.forEach(q => {
        const L = CGB.hexLetter(q.a); if (!L) return;
        n++;
        // the first significant word: skip "the", "a", "an", and "in"/"on"/"at"/"from" before a place
        const words = q.a.split(/[.,;:(]/)[0].trim().split(/\s+/);
        while (words.length > 1 && /^(in|into|on|at|from|the|a|an)$/i.test(words[0])) words.shift();
        const w = words[0];
        if (STOP.test(w) || w[0].toUpperCase() !== L || !/^[A-Za-z]/.test(w) || q.a.split(/\s+/).length > 10) out.bad.push(`${L} | ${q.a}`);
      });
      out.perSet[s.id] = n; out.used += n;
    });
    // answers that start with a little word or aren't the answer itself never get a letter
    out.none = ['It decreases', 'To stop blood flowing backwards', 'They have a full outer shell of electrons', 'About 80% (78%)', 'Any two: wind, solar', 'H⁺ + OH⁻ → H₂O', 'True'].filter(a => CGB.hexLetter(a));
    out.cases = ['In the right atrium', 'The pancreas', 'A virus', 'An electrical impulse', 'In the CNS'].map(a => CGB.hexLetter(a));
    return out;
  });
  expect(r.bad).toEqual([]);
  expect(r.none).toEqual([]);
  expect(r.cases).toEqual(['R', 'P', 'V', 'E', 'C']);
  // every built-in set still fills a 4 × 4 board
  Object.values(r.perSet).forEach(n => expect(n).toBeGreaterThanOrEqual(16));
  // and the board uses that letter
  await page.click('#hh-startBtn');
  const cells = await page.evaluate(() => CGB.games['hex-hunt'].state().letters.map(([L, a]) => [L, CGB.hexLetter(a)]));
  expect(cells).toHaveLength(16);
  cells.forEach(([L, want]) => expect(L).toBe(want));
});

test('Difficulty lines put Category Clash questions on the matching row, and older Tier and Level lines still work', async ({ page }) => {
  await openBundle(page);
  await page.evaluate(() => {
    const lines = ['Subject: Test', 'Topic: Graded'];
    [3, 2, 1].forEach(d => { lines.push('Difficulty: ' + d, `Q: Tier ${d} question?`, `A: Answer ${d}`); });
    // "Tier: 1-3" (versions 1.0.x) and "Level: 1-5" (the first stand-alone games)
    lines.push('Topic: Older', 'Level: 5', 'Q: Old hard?', 'A: H', 'Tier: 2', 'Q: Old middle?', 'A: M', 'Level: 2', 'Q: Old easy?', 'A: E');
    CGB.bank.addSet('Graded set', lines.join('\n'));
  });
  await page.click('[data-play="category-clash"]');
  await page.click('#cc-startBtn');
  const tiles = await page.evaluate(() => CGB.games['category-clash'].board().map(c => [c.topic, c.tiles.map(t => t.q && t.q.q)]));
  const by = Object.fromEntries(tiles);
  expect(by.Graded).toEqual(['Tier 1 question?', 'Tier 2 question?', 'Tier 3 question?']);   // 100, 200, 300
  expect(by.Older).toEqual(['Old easy?', 'Old middle?', 'Old hard?']);
  // the difficulty is kept when the set is written back out as text
  const text = await page.evaluate(() => CGB.bank.toText(CGB.bank.active().questions));
  expect(text).toContain('Difficulty: 3\nQ: Tier 3 question?');
  expect(text).toContain('Difficulty: 1\nQ: Old easy?');
});

test('Category Clash fills a row from the nearest tier when a tier is missing', async ({ page }) => {
  await openBundle(page);
  await page.evaluate(() => {
    const lines = ['Subject: Test', 'Topic: Lopsided', 'Difficulty: 1'];
    for (let i = 1; i <= 4; i++) lines.push(`Q: Easy ${i}?`, `A: E${i}`);
    lines.push('Difficulty: 3', 'Q: Hard one?', 'A: H');
    CGB.bank.addSet('Lopsided set', lines.join('\n'));
  });
  await page.click('[data-play="category-clash"]');
  await page.click('#cc-startBtn');
  const tiles = await page.evaluate(() => CGB.games['category-clash'].board().map(c => c.tiles.map(t => t.q && t.q.q)));
  expect(tiles[0].filter(Boolean).length).toBe(3);
  expect(tiles[0][2]).toBe('Hard one?');
  expect(tiles[0][0]).toMatch(/^Easy/);
});

test('every built-in question has a difficulty, and every topic has questions at all three', async ({ page }) => {
  await openBundle(page);
  const r = await page.evaluate(() => {
    const out = { untagged: 0, short: [], counts: [0, 0, 0] };
    CGB.bank.all().filter(s => s.builtin).forEach(s => {
      const topics = {};
      s.questions.forEach(q => {
        if (![1, 2, 3].includes(q.difficulty)) out.untagged++; else out.counts[q.difficulty - 1]++;
        (topics[q.topic] = topics[q.topic] || [0, 0, 0])[q.difficulty - 1]++;
      });
      Object.entries(topics).forEach(([t, n]) => { if (n.some(x => x < 1)) out.short.push(s.id + ': ' + t + ' ' + n.join('/')); });
    });
    return out;
  });
  expect(r.untagged).toBe(0);
  expect(r.short).toEqual([]);
  // each tier is a real share of the questions, not a token one
  r.counts.forEach(n => expect(n).toBeGreaterThan(200));
});

test('questions without a difficulty get one from their wording', async ({ page }) => {
  await openBundle(page);
  const t = await page.evaluate(() => [
    { q: 'Which gland produces insulin?', a: 'The pancreas' },
    { q: 'What is the test for starch, and what is a positive result?', a: 'Iodine solution turns from orange-brown to blue-black' },
    { q: 'Give the word equation for photosynthesis.', a: 'Carbon dioxide + water → glucose + oxygen' },
    { q: 'Why do veins have valves?', a: 'To stop blood flowing backwards' },
    { q: 'Explain why metals conduct electricity.', a: 'Delocalised electrons' },
    { q: 'A car speeds up from 0 to 20 m/s in 5 s. What is its acceleration?', a: '4 m/s²' }
  ].map(q => CGB.bank.estimateTier(q)));
  expect(t).toEqual([1, 2, 2, 3, 3, 3]);
});

test('Category Clash columns are topics the teacher can choose, remembered for the set', async ({ page }) => {
  const log = await openBundle(page);
  await page.selectOption('#subjectSelect', 'biology');
  await page.click('[data-play="category-clash"]');
  // by default the topics are picked each game
  await expect(page.locator('#cc-topicSum')).toContainText('picked for you each game');
  await page.locator('#cc-topicPick').evaluate(d => { d.open = true; });
  const chips = page.locator('#cc-topicChips [data-topic]');
  await expect(chips).toHaveCount(7);                       // the seven Biology topics
  await expect(page.locator('#cc-topicChips [data-any]')).toHaveAttribute('aria-pressed', 'true');
  for (const n of ['Ecology', 'Bioenergetics']) await page.click(`#cc-topicChips [data-topic="${n}"]`);
  await expect(page.locator('#cc-topicSum')).toContainText('Ecology, Bioenergetics');
  await expect(page.locator('#cc-topicSum')).toContainText('+ 2 picked for you');
  await page.click('#cc-startBtn');
  let cats = await page.evaluate(() => CGB.games['category-clash'].board().map(c => c.topic));
  expect(cats.slice(0, 2)).toEqual(['Ecology', 'Bioenergetics']);
  expect(new Set(cats).size).toBe(4);
  // every column has a question at each tier, on the matching row
  const tiers = await page.evaluate(() => CGB.games['category-clash'].board().map(c => c.tiles.map(t => CGB.bank.difficultyOf(t.q))));
  tiers.forEach(col => expect(col).toEqual([1, 2, 3]));
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  // four chosen: the rest can't be added; remembered next time
  await page.click('[data-play="category-clash"]');
  await page.locator('#cc-topicPick').evaluate(d => { d.open = true; });
  for (const n of ['Cell biology', 'Organisation']) await page.click(`#cc-topicChips [data-topic="${n}"]`);
  await expect(page.locator('#cc-topicChips [data-topic="Ecology"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#cc-topicChips [data-topic="Infection and response"]')).toBeDisabled();
  await page.click('#cc-startBtn');
  cats = await page.evaluate(() => CGB.games['category-clash'].board().map(c => c.topic));
  expect(cats).toEqual(['Ecology', 'Bioenergetics', 'Cell biology', 'Organisation']);
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  // "Pick for me" clears the choice
  await page.click('[data-play="category-clash"]');
  await page.locator('#cc-topicPick').evaluate(d => { d.open = true; });
  await page.click('#cc-topicChips [data-any]');
  await expect(page.locator('#cc-topicSum')).toContainText('picked for you each game');
  // only topics within the chosen subject: a Chemistry board has only Chemistry topics
  await page.keyboard.press('Escape');
  await page.selectOption('#subjectSelect', 'chemistry');
  await page.click('[data-play="category-clash"]');
  await page.click('#cc-startBtn');
  const subs = await page.evaluate(() => CGB.games['category-clash'].board().map(c => c.subject));
  expect(new Set(subs)).toEqual(new Set(['Chemistry']));
  expect(log.errors).toEqual([]);
});
