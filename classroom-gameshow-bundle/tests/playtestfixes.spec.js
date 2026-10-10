// The fixes from the classroom playtest: the teacher stays in control of the countdown, pause,
// End game and show results, protection from double presses and accidental restarts, Outpace's
// sprint, Hex Hunt's best of three, longer time for calculations, and Over the Edge's big question.
const { test } = require('@playwright/test');
const { openBundle, state, mark, playHexHunt, PICK, expect, afterResults } = require('./helpers');

const count = (page, g) => page.evaluate(id => { const e = document.querySelector(`#game-${id} .cm-count:not([hidden])`); return e ? e.textContent : ''; }, g);
const left = (page) => page.evaluate(() => CGB.pause.now() / 1000);   // the game clock: it stops while paused
async function ccQuestion(page) {
  await page.click('#cc-startBtn');
  await page.locator('#cc-board .cc-tile[data-c]').first().click();
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('think');
}

test('there is no countdown: the question waits until the teacher presses Space for "3, 2, 1, show me!"', async ({ page }) => {
  test.setTimeout(90000);
  const log = await openBundle(page, '#category-clash');
  await ccQuestion(page);
  await page.waitForTimeout(3000);
  expect((await state(page, 'category-clash')).round).toBe('think');      // nothing happens by itself
  await expect(page.locator('#game-category-clash .cm-count:not([hidden])')).toHaveCount(0);
  await expect(page.locator('#cc-qBtns')).not.toContainText("Time's up");
  await expect(page.locator('#cc-qBtns')).not.toContainText('+10 s');
  await page.keyboard.press('t');                                          // T no longer does anything
  expect((await state(page, 'category-clash')).round).toBe('think');
  await page.keyboard.press('Space');
  await expect.poll(async () => (await state(page, 'category-clash')).round).toMatch(/show|mark/);
  expect(log.errors).toEqual([]);
});

test('P pauses everything with a big banner; any prompt pauses the game while it is open', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await ccQuestion(page);
  await page.keyboard.press('p');
  await expect(page.locator('.cgb-pausebar')).toBeVisible();
  await expect(page.locator('.cgb-pausebar')).toContainText('Paused');
  await expect(page.locator('#game-category-clash [data-pause]')).toContainText('Carry on');
  await page.waitForTimeout(200);
  const l0 = await left(page);
  await page.waitForTimeout(1500);
  expect(Math.abs(await left(page) - l0)).toBeLessThan(0.15);                    // the game clock is frozen
  await page.keyboard.press('Space');                                     // and the game's keys wait too
  expect((await state(page, 'category-clash')).round).toBe('think');
  await page.keyboard.press('p');
  await expect(page.locator('.cgb-pausebar')).toBeHidden();
  await page.waitForTimeout(1200);
  expect(await left(page)).toBeGreaterThan(l0 + 0.8);
  // the Leave prompt (Esc) holds the game paused while it is open
  await page.keyboard.press('Escape');
  await expect(page.locator('#leaveModal')).toBeVisible();
  const l1 = await left(page);
  await page.waitForTimeout(1500);
  expect(Math.abs(await left(page) - l1)).toBeLessThan(0.15);
  await page.click('#leaveModal [data-close="leaveModal"]');             // Keep playing
  await page.waitForTimeout(1000);
  expect(await left(page)).toBeGreaterThan(l1 + 0.6);
  // so does the Settings dialog
  await page.click('#cc-settingsBtn');
  const l2 = await left(page);
  await page.waitForTimeout(1000);
  expect(Math.abs(await left(page) - l2)).toBeLessThan(0.15);
  expect(log.errors).toEqual([]);
});

test('pause freezes Outpace\'s sprint clock and Over the Edge\'s physics', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  await page.waitForTimeout(800);
  await page.click('#game-outpace [data-pause]');                          // the top bar button
  await page.waitForTimeout(200);
  const t0 = (await state(page, 'outpace')).timeLeft;
  await page.waitForTimeout(1500);
  expect(Math.abs((await state(page, 'outpace')).timeLeft - t0)).toBeLessThan(0.15);
  await page.click('.cgb-pausebar [data-resume]');
  await page.waitForTimeout(1000);
  expect((await state(page, 'outpace')).timeLeft).toBeLessThan(t0 - 0.5);
  // Over the Edge: the physics clock stops
  await page.click('#op-menuBtn'); await page.click('#leaveConfirm');
  await page.click('[data-play="over-the-edge"]');
  await page.click('#ote-startBtn');
  await page.waitForTimeout(500);
  await page.keyboard.press('p');
  const sim = await page.evaluate(() => CGB.test.ote.simTime());
  await page.waitForTimeout(1200);
  expect(await page.evaluate(() => CGB.test.ote.simTime())).toBe(sim);
  await page.keyboard.press('p');
  await expect.poll(() => page.evaluate(() => CGB.test.ote.simTime())).toBeGreaterThan(sim);
  expect(log.errors).toEqual([]);
});

test('End game and show results, in every game: from the top bar (tap twice) or the Menu prompt', async ({ page }) => {
  const log = await openBundle(page);
  const end = { 'over-the-edge': '#ote-sumCard', outpace: '#op-finalVerdict', 'category-clash': '#cc-sumCard', 'hex-hunt': '#hh-sumCard' };
  let k = 0;
  for (const [g, sel] of Object.entries(end)) {
    await page.click(`[data-play="${g}"]`);
    await page.click(`#game-${g} [id$="-startBtn"]`);
    if (g === 'outpace') await page.keyboard.press('2');
    if (g === 'hex-hunt') await page.locator('#hh-board .hh-hex').first().click();
    if (g === 'category-clash') await page.locator('#cc-board .cc-tile[data-c]').first().click();   // a question is open: End game is still on top
    await expect.poll(async () => (await state(page, g)).round).toBe('think');
    await mark(page, g, i => i === 0);
    if (k++ % 2 === 0) {
      const b = page.locator(`#game-${g} [data-endgame]`);
      await b.click(); await expect(b).toContainText('Tap again'); await b.click();
    } else {
      await page.keyboard.press('Escape');
      await expect(page.locator('#leaveModal')).toBeVisible();
      await page.click('#leaveEnd');
    }
    await expect.poll(async () => (await state(page, g)).phase).toBe('summary');
    await expect(page.locator(sel)).toBeVisible();
    await expect(page.locator(`#game-${g}`)).toContainText('Reteach these');   // the misconceptions summary
    await afterResults(page);
    await page.keyboard.press('Escape');
    await expect(page.locator('#launcher')).toBeVisible();
  }
  expect(log.errors).toEqual([]);
});

test('double presses: Enter after "Back to the board" waits for the captain, and two quick Spaces are one', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await ccQuestion(page);
  await page.keyboard.press('Space'); await page.keyboard.press('Space');   // a quick double press
  expect((await state(page, 'category-clash')).round).toBe('show');        // the "3, 2, 1" still plays
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('mark');
  await page.keyboard.press('1'); await page.keyboard.press('Enter');
  await page.keyboard.press('Enter'); await page.keyboard.press('Enter');   // Back to the board, and a stray second press
  await page.waitForTimeout(200);
  expect((await state(page, 'category-clash')).phase).toBe('board');       // no tile opened before the captain chose
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');                                      // a deliberate press later still works
  expect((await state(page, 'category-clash')).phase).toBe('question');
  // Hex Hunt
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  await page.click('[data-play="hex-hunt"]'); await page.click('#hh-startBtn');
  await page.locator('#hh-board .hh-hex').first().click();
  await mark(page, 'hex-hunt', () => true);
  await page.keyboard.press('Enter'); await page.keyboard.press('Enter'); await page.keyboard.press('Enter');
  await page.waitForTimeout(200);
  expect((await state(page, 'hex-hunt')).phase).toBe('board');
  expect(log.errors).toEqual([]);
});

test('results screens ignore keys for 2 seconds, so Enter can\'t start a new game by accident', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await ccQuestion(page);
  const b = page.locator('#game-category-clash [data-endgame]');
  await b.click(); await b.click();
  await expect.poll(async () => (await state(page, 'category-clash')).phase).toBe('summary');
  await page.keyboard.press('Enter'); await page.keyboard.press('Space');
  await page.waitForTimeout(600);
  expect((await state(page, 'category-clash')).phase).toBe('summary');
  await afterResults(page);
  await page.keyboard.press('Enter');                                      // after 2 seconds Enter plays again
  await expect.poll(async () => (await state(page, 'category-clash')).phase).not.toBe('summary');
  expect(log.errors).toEqual([]);
});

test('closing or refreshing the page mid-game asks first ("Leave this page?")', async ({ page }) => {
  await openBundle(page, '#category-clash');
  await ccQuestion(page);
  let asked = null;
  page.once('dialog', d => { asked = d.type(); d.dismiss(); });
  await page.reload({ timeout: 5000 }).catch(() => {});
  await expect.poll(() => asked).toBe('beforeunload');
  expect(await page.evaluate(() => CGB.games['category-clash'].state().phase)).toBe('question');   // still there
});

test('Outpace: under 10 seconds the question on screen is the final question, and the Deal Round needs more than half', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await page.click(PICK('#op-segGroups', 2));
  await page.click('#op-startBtn');
  await page.keyboard.press('2');
  // with 2 teams, one right is not enough
  let s = await state(page, 'outpace'); const h0 = s.hunter;
  await mark(page, 'outpace', i => i === 0);
  expect((await state(page, 'outpace')).hunter).toBe(h0 - 1);
  await expect(page.locator('#op-dealVerdict')).toContainText('Not both teams');
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  s = await state(page, 'outpace');
  expect(s.target).toBe(4); expect(s.timeLeft).toBe(90);
  await page.evaluate(() => CGB.test.outpace.setTime(9.5));
  await expect(page.locator('#op-sprintFinal')).toContainText('Final question!');
  await mark(page, 'outpace', () => true);                                 // it is answered and counted, then the sprint ends
  expect((await state(page, 'outpace')).net).toBe(2);
  await expect.poll(async () => (await state(page, 'outpace')).phase, { timeout: 20000 }).toBe('summary');
  await expect(page.locator('#op-finalVerdict')).toContainText('Caught');
  expect(log.errors).toEqual([]);
});

test('Hex Hunt: best of three with the round score on screen, and its marks on the edges', async ({ page }) => {
  test.setTimeout(300000);
  const log = await openBundle(page, '#hex-hunt');
  await page.click('#hh-startBtn');
  await expect(page.locator('#hh-board .hh-edgemk')).toHaveCount(4);     // ● on the left and right edges, ■ top and bottom
  await expect(page.locator('#hh-board .hh-edgemk').first()).toHaveText('●');
  let s = await state(page, 'hex-hunt');
  expect(s.toWin).toBe(2);
  await expect(page.locator('#hh-turn')).toContainText('Round 1 · best of three');
  // half 1 wins every question: two rounds, then the results
  const seen = new Set();
  for (let guard = 0; guard < 400; guard++) {
    s = await state(page, 'hex-hunt');
    seen.add(s.roundNo);
    if (s.phase === 'summary') break;
    if (s.phase === 'won') { await expect(page.locator('#hh-win')).toContainText(s.rounds[0] >= 2 ? 'win the match' : 'take round'); await page.keyboard.press('Enter'); await page.waitForTimeout(550); continue; }
    if (s.phase === 'board' || s.round === 'done') { await page.keyboard.press('Enter'); await page.waitForTimeout(80); continue; }
    if (s.round === 'think' || s.round === 'show') { await page.keyboard.press('Space'); await page.waitForTimeout(550); continue; }
    if (s.round === 'mark') { await page.keyboard.press('1'); continue; }
    await page.waitForTimeout(80);
  }
  s = await state(page, 'hex-hunt');
  expect(s.phase).toBe('summary'); expect(s.rounds).toEqual([2, 0]); expect([...seen]).toEqual([1, 2]);
  await expect(page.locator('#hh-sumCard')).toContainText('Rounds won');
  expect(log.errors).toEqual([]);
});

test('Hex Hunt: Maths is a single round; 12 questions with no hexagon won end the round', async ({ page }) => {
  test.setTimeout(240000);
  const log = await openBundle(page);
  await page.selectOption('#subjectSelect', 'maths');
  await page.click('[data-play="hex-hunt"]');
  await page.click('#hh-startBtn');
  expect((await state(page, 'hex-hunt')).toWin).toBe(1);
  await expect(page.locator('#hh-turn')).not.toContainText('best of three');
  // nobody gets 12 questions in a row
  await page.locator('#hh-board .hh-hex').first().click();
  for (let n = 0; n < 12; n++) {
    await expect.poll(async () => (await state(page, 'hex-hunt')).round).toBe('think');
    await page.keyboard.press('Space'); await page.waitForTimeout(550); await page.keyboard.press('Space');
    await expect.poll(async () => (await state(page, 'hex-hunt')).round).toBe('mark');
    await page.keyboard.press('n');                                      // Neither
    const s = await state(page, 'hex-hunt');
    if (n < 11) { expect(s.step).toBe('nobody'); await page.keyboard.press('Enter'); }   // a new question for the same hexagon
  }
  let s = await state(page, 'hex-hunt');
  expect(s.step).toBe('stalled'); expect(s.dry).toBe(12);
  await expect(page.locator('#hh-qMsg')).toContainText('12 questions');
  await page.keyboard.press('Enter');                                    // End the round
  s = await state(page, 'hex-hunt');
  expect(s.phase).toBe('won'); expect(s.rounds).toEqual([0, 0]);
  await expect(page.locator('#hh-win')).toContainText('nobody takes it');
  await expect(page.locator('#hh-winBtn')).toContainText('See the results');   // Maths: one round
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await state(page, 'hex-hunt')).phase).toBe('summary');
  await expect(page.locator('#hh-sumCard')).toContainText("It's a draw!");
  expect(log.errors).toEqual([]);
});

test('Over the Edge: the question is big while the class writes; equal money shares a place', async ({ page }) => {
  const log = await openBundle(page, '#over-the-edge');
  await page.click('#ote-startBtn');
  await expect.poll(async () => (await state(page, 'over-the-edge')).round).toBe('think');
  const q = (await state(page, 'over-the-edge')).q.q;
  await expect(page.locator('#ote-bigq')).toBeVisible();
  await expect(page.locator('#ote-bigText')).toHaveText(q);
  const fs = await page.locator('#ote-bigText').evaluate(e => parseFloat(getComputedStyle(e).fontSize));
  const side = await page.locator('#ote-qText').evaluate(e => parseFloat(getComputedStyle(e).fontSize));
  expect(fs).toBeGreaterThan(side * 1.3);
  await page.keyboard.press('Space');
  await expect(page.locator('#ote-bigq')).toBeHidden();                  // back to the side panel for marking and the drop
  // end at once: every team on £0 shares first place
  await expect.poll(async () => (await state(page, 'over-the-edge')).round).toMatch(/mark|show/);
  await page.keyboard.press('Escape'); await page.click('#leaveEnd');
  await expect(page.locator('#ote-sumCard .sum-rank .pos')).toHaveText(['1', '1', '1', '1']);
  expect(log.errors).toEqual([]);
});

test('A reveals the answer before marking; Category Clash results say "wins"', async ({ page }) => {
  const log = await openBundle(page, '#category-clash');
  await page.click(PICK('#cc-segTeams', 2));
  await ccQuestion(page);
  await page.keyboard.press('Space');
  await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('mark');
  await expect(page.locator('#cc-qAnswer')).not.toHaveClass(/shown/);
  await page.keyboard.press('a');
  await expect(page.locator('#cc-qAnswer')).toHaveClass(/shown/);
  await page.keyboard.press('1'); await page.keyboard.press('Enter');
  const b = page.locator('#game-category-clash [data-endgame]');
  await b.click(); await b.click();
  await expect(page.locator('#cc-sumCard h2')).toContainText('wins!');
  expect(log.errors).toEqual([]);
});
