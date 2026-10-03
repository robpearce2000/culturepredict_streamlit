// The shared question bank: import, both games, backup round trip, shared history.
const fs = require('fs');
const { test } = require('@playwright/test');
const { openBundle, state, playOutpace, setNames, expect } = require('./helpers');

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

test('a pasted question set appears in both games', async ({ page }) => {
  const log = await openBundle(page);
  await page.click('#openBank');
  await page.fill('#bankName', 'Pasted test set');
  await page.fill('#bankText', PASTE);
  await page.click('#bankSave');
  await expect(page.locator('#bankFormStatus')).toContainText('3 questions');
  await expect(page.locator('#bankList')).toContainText('Pasted test set');
  await page.keyboard.press('Escape');
  await expect(page.locator('#activeSetName')).toHaveText('Pasted test set');

  // Over the Edge: the set is listed, selected, and its questions are asked
  await page.click('[data-play="over-the-edge"]');
  const oteSel = page.locator('#ote-setSelect');
  await expect(oteSel.locator('option', { hasText: 'Pasted test set' })).toHaveCount(1);
  const selectedOte = await oteSel.evaluate(s => s.options[s.selectedIndex].text);
  expect(selectedOte).toContain('Pasted test set');
  await page.click('#ote-startBtn');
  const q1 = (await state(page, 'over-the-edge')).q;
  expect(['Test topic alpha', 'Test topic beta']).toContain(q1.topic);
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Outpace: same
  await page.click('[data-play="outpace"]');
  const opSel = page.locator('#op-setSelect');
  expect(await opSel.evaluate(s => s.options[s.selectedIndex].text)).toContain('Pasted test set');
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

test('wrong answers logged in one game show up in the other for the same player', async ({ page }) => {
  const log = await openBundle(page, '#over-the-edge');
  await setNames(page, 'ote', 'Gina', 'Hal');
  await page.click('#ote-startBtn');
  const s = await state(page, 'over-the-edge');
  expect(s.step).toBe('ask');
  const missedTopic = s.q.topic;
  await page.keyboard.press('w');      // Gina is wrong
  await page.keyboard.press('n');      // no steal
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Outpace remembers the names; play a short game and read Gina's history
  await page.click('[data-play="outpace"]');
  await expect(page.locator('#op-name0')).toHaveValue('Gina');
  await page.click('#op-startBtn');
  await playOutpace(page, { shortSprint: true });
  const ginaCard = page.locator('.op-sum-p', { hasText: 'Gina' });
  await expect(ginaCard).toContainText('Weakest topics, all games');
  await expect(ginaCard).toContainText(missedTopic);
  // and the bank agrees the entry came from Over the Edge
  const games = await page.evaluate(() => CGB.bank.wrongLog('gina').map(e => e.game));
  expect(games).toContain('Over the Edge');
  expect(log.errors).toEqual([]);
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
