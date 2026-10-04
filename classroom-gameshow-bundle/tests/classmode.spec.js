// Whole-class play in all four games: full games with 2, 4 and 6 teams, marking, undo,
// the class misconceptions summary, and the time marking and Over the Edge's drop take.
const { test } = require('@playwright/test');
const { openBundle, state, mark, PICK, expect } = require('./helpers');

const note = (name, value) => test.info().annotations.push({ type: name, description: String(value) });

for (const n of [2, 4, 6]) {
  test(`Category Clash, whole class with ${n} teams: marking, undo, full board and misconceptions`, async ({ page }) => {
    const log = await openBundle(page, '#category-clash');
    await page.click(PICK('#cc-segTeams', n));
    await page.click('#cc-startBtn');
    let s = await state(page, 'category-clash');
    expect(s.rows).toBe(5); expect(s.teams).toHaveLength(n);
    // first tile: the choosing team and team 2 correct
    await page.keyboard.press('Enter');
    const chooser = (await state(page, 'category-clash')).turn;
    const ms = await mark(page, 'category-clash', i => i === chooser || i === (chooser + 1) % n);
    note('marking ms', ms);
    s = await state(page, 'category-clash');
    expect(s.teams[chooser].score).toBe(100);
    expect(s.teams[(chooser + 1) % n].score).toBe(50);
    expect(s.teams.reduce((a, t) => a + t.score, 0)).toBe(150);
    expect(s.undoable).toBe(true);
    await page.keyboard.press('u');
    s = await state(page, 'category-clash');
    expect(s.round).toBe('mark');
    expect(s.teams.every(t => t.score === 0)).toBe(true);
    await page.keyboard.press('c'); await page.keyboard.press('Enter');
    s = await state(page, 'category-clash');
    expect(s.teams[chooser].score).toBe(100);
    expect(s.teams.filter(t => t.score === 50)).toHaveLength(n - 1);
    await page.keyboard.press('Enter');
    // the rest of the board: alternate teams right
    let q = 0;
    for (let guard = 0; guard < 200; guard++) {
      s = await state(page, 'category-clash');
      if (s.phase === 'summary') break;
      if (s.phase === 'board') { await page.keyboard.press('Enter'); continue; }
      if (s.round === 'done') { await page.keyboard.press('Enter'); continue; }
      await mark(page, 'category-clash', i => (i + q) % 3 === 0); q++;
    }
    s = await state(page, 'category-clash');
    expect(s.phase).toBe('summary');
    await expect(page.locator('#cc-sumCard .cm-miscon')).toContainText('Reteach these');
    expect(s.misconceptions.length).toBeGreaterThan(0);
    await expect(page.locator('#cc-sumCard .cc-rank')).toHaveCount(n);
    expect(log.errors).toEqual([]);
  });
}

test('Category Clash: a team far behind gets a catch-up pick at the start of the next round', async ({ page }) => {
  await openBundle(page, '#category-clash');
  await page.click(PICK('#cc-segTeams', 2));
  await page.click('#cc-startBtn');
  // team 1 wins a 500 and half of another (at least the top tile ahead); team 2 gets nothing
  for (let k = 0; k < 2; k++) {
    await page.evaluate(k => { const s = CGB.games['category-clash'].state(); }, k);
    await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowDown');
    if (k) await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Enter');
    await mark(page, 'category-clash', i => i === 0);
    await page.keyboard.press('Enter');
  }
  const s = await state(page, 'category-clash');
  expect(s.teams[0].score - s.teams[1].score).toBeGreaterThanOrEqual(500);
  expect(s.turn).toBe(1);
  expect(s.catchUp).toBe(true);
  await expect(page.locator('.cc-catch')).toBeVisible();
});

test('Hex Hunt, whole class: shares per half, tie to the chooser, undo, full round and misconceptions', async ({ page }) => {
  const log = await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  const id = 'hex-hunt';
  // hexagon 1: half 2 has the higher share
  await page.keyboard.press('Enter');
  await page.keyboard.press('Space'); await page.keyboard.press('Space');
  await expect.poll(async () => (await state(page, id)).round).toBe('mark');
  await page.keyboard.press('1'); for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowRight');
  await page.keyboard.press('2'); for (let i = 0; i < 6; i++) await page.keyboard.press('ArrowRight');
  expect((await state(page, id)).shares).toEqual([30, 60]);
  await page.keyboard.press('Enter');
  let s = await state(page, id);
  expect(s.owners.filter(o => o === 1)).toHaveLength(1);
  expect(s.picker).toBe(1);
  await page.keyboard.press('u');
  s = await state(page, id);
  expect(s.owners.filter(o => o >= 0)).toHaveLength(0);
  expect(s.picker).toBe(0);
  // a tie goes to the half that chose (half 1)
  await page.keyboard.press('1'); for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowRight');
  await page.keyboard.press('2'); for (let i = 0; i < 2; i++) await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Enter');
  s = await state(page, id);
  expect(s.owners.filter(o => o === 0)).toHaveLength(1);
  note('marking ms', s.times.confirm - s.times.mark);
  await page.keyboard.press('Enter');
  for (let guard = 0, k = 0; guard < 400; guard++) {
    s = await state(page, id);
    if (s.phase === 'summary') break;
    if (s.phase === 'board' || s.phase === 'won' || s.round === 'done') { await page.keyboard.press('Enter'); continue; }
    if (s.round === 'think' || s.round === 'show') { await page.keyboard.press('Space'); continue; }
    if (s.round === 'mark') { await page.keyboard.press(String(1 + (k++ % 2))); await page.keyboard.press('ArrowRight'); await page.keyboard.press('Enter'); continue; }
    await page.waitForTimeout(50);
  }
  expect((await state(page, id)).phase).toBe('summary');
  await expect(page.locator('#hh-sumCard .cm-miscon')).toContainText('Reteach these');
  expect(log.errors).toEqual([]);
});

for (const n of [2, 4, 6]) {
  test(`Over the Edge, whole class with ${n} teams: marking, undo, fast drop, final and misconceptions`, async ({ page }) => {
    await page.addInitScript(() => { window.__SHOWTIME_MANUAL__ = true; });   // physics moves only when the test says
    const log = await openBundle(page, '#over-the-edge');
    await page.click(PICK('#ote-segTeams', n));
    await page.click('#ote-startBtn');
    const id = 'over-the-edge';
    // question 1: every team correct; undo once, then confirm again
    await mark(page, id, () => true);
    let s = await state(page, id);
    expect(s.step).toBe('lanes'); expect(s.laneQueue).toHaveLength(n); expect(s.undoable).toBe(true);
    await page.keyboard.press('u');
    s = await state(page, id);
    expect(s.step).toBe('ask'); expect(s.correct.every(c => c === 0)).toBe(true);
    await page.keyboard.press('c'); await page.keyboard.press('Enter');
    for (let i = 0; i < n; i++) await page.keyboard.press(String(1 + i % 4));
    s = await state(page, id);
    expect(s.step).toBe('dropping'); expect(s.undoable).toBe(false);
    for (let i = 0; i < 40 && (await state(page, id)).step !== 'next'; i++) await page.evaluate(() => CGB.test.ote.advance(0.5));
    s = await state(page, id);
    const dropSeconds = s.dropDoneAt - s.markAt;
    note('drop phase (game seconds)', dropSeconds.toFixed(2));
    expect(s.step).toBe('next');
    expect(dropSeconds, 'the drop phase finishes within about 10 seconds of marking').toBeLessThanOrEqual(10);
    // the rest of the game
    for (let guard = 0, k = 0; guard < 600; guard++) {
      s = await state(page, id);
      if (s.phase === 'summary') break;
      if (s.step === 'ask') { await mark(page, id, i => (i + k) % 2 === 0); k++; continue; }
      if (s.step === 'lanes') await page.keyboard.press('r');
      else if (s.step === 'dropping') await page.evaluate(() => CGB.test.ote.advance(0.5));
      else if (s.step === 'next') await page.keyboard.press('Space');
      await page.waitForTimeout(30);
    }
    s = await state(page, id);
    expect(s.phase).toBe('summary');
    await expect(page.locator('#ote-sumCard .cm-miscon')).toContainText('Reteach these');
    await expect(page.locator('#ote-sumCard .sum-rank')).toHaveCount(n);
    expect(log.errors).toEqual([]);
  });
}

test('Over the Edge, whole class: six correct teams drop within 10 seconds, and a far-behind team gets a catch-up counter', async ({ page }) => {
  await page.addInitScript(() => { window.__SHOWTIME_MANUAL__ = true; });
  await openBundle(page, '#over-the-edge');
  await page.click(PICK('#ote-segTeams', 6));
  await page.click('#ote-startBtn');
  await page.evaluate(() => { CGB.test.ote.setMoney([600, 400, 300, 0, 200, 500]); CGB.test.ote.toFinal(); });
  const s = await state(page, 'over-the-edge');
  expect(s.phase).toBe('final');
  expect(s.catchUp).toBe(3);
  expect(s.step).toBe('lanes');
  expect(s.laneQueue).toEqual([3]);
  await expect(page.locator('#ote-board .cm-badge')).toContainText('Catch-up counter');
});

for (const n of [2, 4, 6]) {
  test(`Outpace, whole class with ${n} teams: half-the-teams rule, undo, sprint and misconceptions`, async ({ page }) => {
    const log = await openBundle(page, '#outpace');
    await page.click(PICK('#op-segGroups', n));
    await page.click('#op-startBtn');
    const id = 'outpace';
    await page.keyboard.press('2');                       // the class voted Standard
    let s = await state(page, id);
    const r0 = s.runner, h0 = s.hunter;
    // exactly half correct: the runner moves
    await mark(page, id, i => i < n / 2);
    s = await state(page, id);
    expect(s.runner).toBe(r0 - 1); expect(s.hunter).toBe(h0);
    await page.keyboard.press('u');
    s = await state(page, id);
    expect(s.runner).toBe(r0);
    // fewer than half: the Hunter gains
    for (let i = 0; i < n; i++) if (i < n / 2) await page.keyboard.press(String(i + 1));   // untoggle back to none
    await page.keyboard.press('w'); if (n > 2) await page.keyboard.press('1');
    await page.keyboard.press('Enter');
    s = await state(page, id);
    expect(s.hunter).toBe(h0 - 1); expect(s.runner).toBe(r0);
    await page.keyboard.press('Enter');
    let sprintSeen = false, dealsVoted = 1;
    for (let guard = 0, k = 0; guard < 600; guard++) {
      s = await state(page, id);
      if (s.phase === 'summary') break;
      if (s.roundEnd) { await page.keyboard.press('Enter'); continue; }
      if (s.phase === 'deal' && !s.dealReward) { dealsVoted++; await page.keyboard.press('2'); continue; }   // the second Deal Round
      if (s.phase === 'sprint') {
        if (!sprintSeen) { sprintSeen = true; expect(s.target).toBeGreaterThanOrEqual(3); note('sprint target', s.target); }
        if (s.timeLeft > 5 && k > 3) await page.evaluate(() => CGB.test.outpace.setTime(2));
      }
      if (s.round === 'think' || s.round === 'show' || s.round === 'mark') { if (s.phase === 'deal' || s.phase === 'sprint') { await mark(page, id, i => (i + k) % 3 !== 0); k++; continue; } }
      if (s.round === 'done' && s.phase === 'deal') { await page.keyboard.press('Enter'); continue; }
      await page.waitForTimeout(60);
    }
    s = await state(page, id);
    expect(s.phase).toBe('summary');
    expect(sprintSeen).toBe(true);
    expect(dealsVoted).toBe(2);
    await expect(page.locator('#op-summaryPlayers .cm-miscon')).toContainText('Reteach these');
    expect(log.errors).toEqual([]);
  });
}
