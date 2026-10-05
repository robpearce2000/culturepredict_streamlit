// The shared question bank: import, both games, backup round trip, shared history.
const fs = require('fs');
const { test } = require('@playwright/test');
const { openBundle, state, mark, playOutpace, setNames, PICK, expect } = require('./helpers');

const PASTE = `Subject: Chemistry
Topic: Test topic alpha
Q: What is the chemical symbol for sodium?
A: Na
Q: What is the chemical symbol for potassium?
A: K
Subject: Physics
Topic: Test topic beta
Q: What is the unit of force?
A: The newton (N)`;

test('a pasted question set appears in every game', async ({ page }) => {
  const log = await openBundle(page);
  await page.click('#openBank');
  await page.fill('#bankName', 'Pasted test set');
  await page.fill('#bankText', PASTE);
  await page.click('#bankSave');
  await expect(page.locator('#bankFormStatus')).toContainText('3 questions');
  await expect(page.locator('#bankList')).toContainText('Pasted test set');
  await page.keyboard.press('Escape');
  // it is now the set in use on the main screen
  await expect(page.locator('#packBtn')).toContainText('Pasted test set');

  // Over the Edge: the setup card names it, and its questions are asked
  await page.click('[data-play="over-the-edge"]');
  await expect(page.locator('#ote-pack')).toContainText('Pasted test set');
  await page.click('#ote-startBtn');
  const q1 = (await state(page, 'over-the-edge')).q;
  expect(['Test topic alpha', 'Test topic beta']).toContain(q1.topic);
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Outpace: same
  await page.click('[data-play="outpace"]');
  await expect(page.locator('#op-pack')).toContainText('Pasted test set');
  await page.click('#op-startBtn');
  await page.keyboard.press('1');
  const q2 = (await state(page, 'outpace')).q;
  expect(['Test topic alpha', 'Test topic beta']).toContain(q2.topic);
  expect(log.errors).toEqual([]);
});

test('JSON backup export and re-import round-trips with no loss', async ({ page }, testInfo) => {
  await openBundle(page);
  // make some data: two sets and some history
  await page.evaluate(text => {
    CGB.bank.addSet('Round trip A', text);
    CGB.bank.addSet('Round trip B', 'Subject: Biology\nTopic: Cells\nQ: Name the control centre of the cell.\nA: The nucleus');
    const q = CGB.bank.all()[0].questions[0];
    CGB.bank.logWrong('Erin', q, 'Over the Edge');
    CGB.bank.logWrong('Erin', q, 'Over the Edge');
    CGB.bank.logWrong('Finn', CGB.bank.all()[1].questions[2], 'Outpace');
  }, PASTE);
  const before = await page.evaluate(() => CGB.bank.exportData());

  // export through the real button
  await page.click('#openBank');
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#bankExport')]);
  const file = testInfo.outputPath('backup.json');
  await download.saveAs(file);
  const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
  expect(saved.format).toBe('classroom-gameshow-bundle-backup');

  // wipe this device and reload
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.waitForFunction(() => window.CGB && CGB.app);
  expect(await page.evaluate(() => CGB.bank.players().length)).toBe(0);

  // import through the real button
  await page.click('#openBank');
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('#bankImport')]);
  await chooser.setFiles(file);
  await expect(page.locator('#bankBackupStatus')).toContainText('Imported');
  const after = await page.evaluate(() => CGB.bank.exportData());
  const strip = d => { const c = JSON.parse(JSON.stringify(d)); delete c.exported; return c; };
  expect(strip(after)).toEqual(strip(before));

  // importing the same file again adds nothing
  const [chooser2] = await Promise.all([page.waitForEvent('filechooser'), page.click('#bankImport')]);
  await chooser2.setFiles(file);
  await expect(page.locator('#bankBackupStatus')).toContainText('0 new sets, 0 updated, 0 history entries');
  expect(strip(await page.evaluate(() => CGB.bank.exportData()))).toEqual(strip(before));
});

test('wrong answers logged in one game are shared with the others for the same team', async ({ page }) => {
  const log = await openBundle(page, '#over-the-edge');
  await page.click(PICK('#ote-segTeams', 2));
  await setNames(page, 'over-the-edge', ['Gina', 'Hal']);
  await page.click('#ote-startBtn');
  const s = await state(page, 'over-the-edge');
  expect(s.step).toBe('ask');
  const missedTopic = s.q.topic;
  await mark(page, 'over-the-edge', i => i === 1);     // Gina is wrong, Hal right
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Outpace remembers the names; play a short game
  await page.click('[data-play="outpace"]');
  await page.click(PICK('#op-segGroups', 2));
  await page.locator('#game-outpace details.tnames summary').click();
  await expect(page.locator('#op-gname0')).toHaveValue('Gina');
  await page.click('#op-startBtn');
  await playOutpace(page, { shortSprint: true });
  // Gina's history holds both games, and the Question bank lists her
  const games = await page.evaluate(() => CGB.bank.wrongLog('gina').map(e => e.game));
  expect(games).toContain('Over the Edge');
  expect(games).toContain('Outpace');
  expect(await page.evaluate(() => CGB.bank.weakTopics('gina', 50).map(t => t[0]))).toContain(missedTopic);
  await page.keyboard.press('Escape');
  await page.click('#openBank');
  await expect(page.locator('#bankPlayers')).toContainText('Gina');
  expect(log.errors).toEqual([]);
});

test('questions from topics the playing teams got wrong before come up more often', async ({ page }) => {
  await openBundle(page);
  const r = await page.evaluate(() => {
    const set = CGB.bank.active(), topic = set.questions[0].topic;
    const inTopic = set.questions.filter(q => q.topic === topic).length / set.questions.length;
    CGB.bank.logWrong('Weak team', set.questions[0], 'Test');
    const p = CGB.bank.createPicker();
    let hits = 0; const N = 30;
    for (let i = 0; i < N; i++) if (p.pick(['Weak team', 'Other team']).topic === topic) hits++;
    return { share: hits / N, base: inTopic };
  });
  expect(r.share).toBeGreaterThan(r.base * 2);
  expect(r.share).toBeLessThan(0.75);                  // still a mix, not only that topic
});

test('the page still works when storage is blocked', async ({ browser }) => {
  const ctx = await browser.newContext();
  await ctx.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', { get() { throw new Error('blocked'); } });
  });
  const page = await ctx.newPage();
  const log = await openBundle(page);
  await expect(page.locator('#storageNote')).toBeVisible();
  await page.click('#openBank');
  await page.fill('#bankName', 'Session only');
  await page.fill('#bankText', PASTE);
  await page.click('#bankSave');
  await expect(page.locator('#bankFormStatus')).toContainText('will be lost');
  await page.keyboard.press('Escape');
  await page.click('[data-play="over-the-edge"]');
  await page.click('#ote-startBtn');
  expect((await state(page, 'over-the-edge')).phase).toBe('r1');
  expect(log.errors.filter(e => !/blocked/.test(e))).toEqual([]);
  await ctx.close();
});
