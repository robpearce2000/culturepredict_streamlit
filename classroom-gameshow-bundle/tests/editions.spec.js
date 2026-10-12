// The editions for sale (tools/editions.js), each built by `node build.js --editions` into
// dist/editions/ (and a test build in test-build/editions/ with the test shortcuts).
const path = require('path');
const fs = require('fs');
const { test } = require('@playwright/test');
const { openBundle, state, playOverTheEdge, playOutpace, playCategoryClash, playHexHunt, afterResults, expect } = require('./helpers');
const { EDITIONS, TASTER, includes } = require('../tools/editions.js');
const { loadAll } = require('../tools/packs.js');
const { rows: AUDIT, MIN } = require('../tools/audit-taster.js');

const ROOT = path.join(__dirname, '..');
const productOf = ed => path.join(ROOT, 'dist', 'editions', ed.file);
const testOf = ed => path.join(ROOT, 'test-build', 'editions', ed.id + '.html');
const { packs } = loadAll();
const LABEL = { biology: 'Biology', chemistry: 'Chemistry', physics: 'Physics', maths: 'Maths', history: 'History', geography: 'Geography' };

test('the edition build makes all nine files and a checksum list', () => {
  const sums = fs.readFileSync(path.join(ROOT, 'dist', 'editions', 'SHA256SUMS.txt'), 'utf8').trim().split('\n');
  expect(sums.map(l => l.split(/\s+/)[1]).sort()).toEqual(EDITIONS.map(e => e.file).sort());
  for (const ed of EDITIONS) {
    const html = fs.readFileSync(productOf(ed));
    const hash = require('crypto').createHash('sha256').update(html).digest('hex');
    expect(sums).toContain(`${hash}  ${ed.file}`);
    expect(String(html)).not.toMatch(/CGB\.test\b|__SHOWTIME_/);
  }
});

for (const ed of EDITIONS) {
  test(`${ed.name}: opens offline with no errors, the right subjects and exactly its own packs`, async ({ page }) => {
    const t0 = Date.now();
    const log = await openBundle(page, '', { file: productOf(ed) });
    const ms = Date.now() - t0;
    expect(ms, 'launcher shown in under 3 seconds').toBeLessThan(3000);
    await expect(page.locator('#launcher')).toBeVisible();
    await expect(page.locator('#wordmark .l-edition')).toHaveText(ed.name);
    // the picker: exactly the edition's subjects, plus Other for the teacher's own sets
    const r = await page.evaluate(() => ({ subjects: CGB.bank.SUBJECTS.map(s => s.id), subject: CGB.bank.subject(), builtin: CGB.bank.all().filter(s => s.builtin && !s.combined).map(s => ({ id: s.id, n: s.questions.length })) }));
    expect(r.subjects).toEqual(ed.subjects.concat('other'));
    expect(r.subject).toBe(ed.subjects[0]);
    if (ed.subjects.length === 1) {
      await expect(page.locator('#subjectSelect')).toBeHidden();
      await expect(page.locator('#subjectFixed')).toHaveText(LABEL[ed.subjects[0]]);
      await page.click('#subjectOwn');                                  // Other: the teacher's own sets
      expect(await page.evaluate(() => CGB.bank.subject())).toBe('other');
      await page.click('#subjectOwn');
      expect(await page.evaluate(() => CGB.bank.subject())).toBe(ed.subjects[0]);
    } else {
      const opts = await page.locator('#subjectSelect option').allTextContents();
      expect(opts).toEqual(ed.subjects.map(s => LABEL[s]).concat('Other'));
    }
    // every built-in pack belongs to the edition, and none is missing
    const want = packs.filter(p => includes(ed, p));
    expect(r.builtin.map(b => b.id).sort()).toEqual(want.map(p => p.id).sort());
    for (const b of r.builtin) expect(b.n, b.id).toBe(want.find(p => p.id === b.id).questions.length);
    if (ed.taster) expect(r.builtin.map(b => b.id).sort()).toEqual(TASTER.slice().sort());
    // the taster says so; the About panel names the edition and, outside the mega bundle, mentions Tes once
    await expect(page.locator('#wordmark .l-taster')).toHaveCount(ed.taster ? 1 : 0);
    await page.click('#openAbout');
    await expect(page.locator('#aboutEdition')).toContainText(ed.name);
    await expect(page.locator('#aboutTaster')).toBeVisible({ visible: !!ed.taster });
    await expect(page.locator('#aboutMore')).toBeVisible({ visible: !ed.all });
    const about = await page.locator('#aboutModal').innerText();
    for (const k of Object.keys(LABEL)) if (!ed.subjects.includes(k)) expect(about, `${ed.name} About mentions ${k}`).not.toContain(LABEL[k]);
    expect(log.errors).toEqual([]);
    expect(log.requests).toEqual([]);
    test.info().annotations.push({ type: 'load', description: `${ms} ms` });
  });

  // Every edition runs the same game code with different packs, so full games are played in the
  // smallest (the free taster) and the largest (the mega bundle); the test above checks every edition's packs
  if (['free-taster', 'mega-bundle'].includes(ed.id)) test(`${ed.name}: one full game of each game plays through with the edition's questions`, async ({ page }) => {
    test.setTimeout(900000);
    const mine = new Set(packs.filter(p => includes(ed, p)).flatMap(p => p.questions.map(q => q.q)));
    const log = await openBundle(page, '', { file: testOf(ed) });
    const seen = new Set();
    const watch = id => page.evaluate(id => { const s = CGB.games[id].state(); return s.q ? s.q.q : null; }, id).then(q => { if (q) seen.add(q); });
    const back = async () => { await afterResults(page); await page.evaluate(() => CGB.app.show('launcher')); };

    // the taster's one calm upgrade line: under the results only (never on the launcher or mid-game), beside the review line
    const UP = ed.taster ? await page.evaluate(() => CGB.upgradeText()) : '';
    const noUpgrade = async () => expect(await page.locator('.cgb-upgrade:visible').count(), 'no upgrade line outside the results').toBe(0);
    const upgrade = async id => {
      if (!ed.taster) return noUpgrade();
      const line = page.locator(`#game-${id} .cgb-upgrade:visible`);
      await expect(line).toHaveCount(1);
      await expect(line).toHaveText(UP);
      await expect(page.locator(`#game-${id} .cgb-review:visible`).first()).toHaveText('Enjoying Showtime? A short review on Tes helps other teachers find it. Thank you!');
      expect((await line.boundingBox()).height, 'one line of text, not a banner').toBeLessThan(60);
    };
    await noUpgrade();
    await page.click('[data-play="over-the-edge"]');
    await page.click('#ote-startBtn');
    await page.evaluate(() => CGB.test.ote.manual(true));
    await noUpgrade();
    expect((await playOverTheEdge(page, { manual: true })).phase).toBe('summary');
    await upgrade('over-the-edge');
    await back();

    await page.click('[data-play="outpace"]');
    await page.click('#op-startBtn');
    await noUpgrade();
    expect((await playOutpace(page, { shortSprint: true })).phase).toBe('summary');
    await upgrade('outpace');
    await back();

    await page.click('[data-play="category-clash"]');
    await page.click('#cc-startBtn');
    await noUpgrade();
    const cc = await playCategoryClash(page);
    expect(cc.phase).toBe('summary');
    await upgrade('category-clash');
    await back();

    await page.click('[data-play="hex-hunt"]');
    await page.click('#hh-startBtn');
    await watch('hex-hunt');
    await noUpgrade();
    expect((await playHexHunt(page)).phase).toBe('summary');
    await upgrade('hex-hunt');

    // every question the games used came from this edition's packs
    const used = await page.evaluate(() => CGB.bank.active().questions.map(q => q.q));
    expect(used.every(q => mine.has(q))).toBe(true);
    for (const q of seen) expect(mine.has(q), q).toBe(true);
    expect(log.errors).toEqual([]);
    expect(log.requests).toEqual([]);
  });
}

// saved data on file:// pages can be disturbed by tests running alongside; one retry (passes on its own)
test.describe(() => { test.describe.configure({ retries: 1 }); test('editions share saved data safely: own sets carry over, and a subject from another edition falls back without being overwritten', async ({ page }) => {
  const by = id => EDITIONS.find(e => e.id === id);
  await openBundle(page, '', { file: productOf(by('mega-bundle')) });
  await page.evaluate(() => {
    CGB.bank.addSet('My Biology quiz', 'Subject: Biology\nTopic: Cells\nQ: What controls the cell?\nA: Nucleus', 'biology');
    CGB.bank.setSubject('history');
  });
  // the Chemistry edition: starts on Chemistry, leaves the saved subject alone, keeps the set (under Other)
  await page.goto('file://' + productOf(by('chemistry')));
  await page.waitForFunction(() => window.CGB && CGB.app);
  const r = await page.evaluate(() => ({ subject: CGB.bank.subject(), saved: CGB.store.get('subject'), own: CGB.bank.ownSets('other').map(s => s.name) }));
  expect(r.subject).toBe('chemistry');
  expect(r.saved).toBe('history');
  expect(r.own).toContain('My Biology quiz');
  // the Biology edition sees it under Biology; the mega bundle still opens on History
  await page.goto('file://' + productOf(by('biology')));
  await page.waitForFunction(() => window.CGB && CGB.app);
  expect(await page.evaluate(() => CGB.bank.ownSets('biology').map(s => s.name))).toContain('My Biology quiz');
  await page.goto('file://' + productOf(by('mega-bundle')));
  await page.waitForFunction(() => window.CGB && CGB.app);
  expect(await page.evaluate(() => CGB.bank.subject())).toBe('history');
});});


// ---- the free taster: its topics, its pools and its one upgrade line ----
const taster = EDITIONS.find(e => e.id === 'free-taster');
test('the free taster has one popular topic per subject on both boards, Biology, Chemistry and Physics included', () => {
  const mine = packs.filter(p => includes(taster, p));
  expect(mine.map(p => p.id).sort()).toEqual(TASTER.slice().sort());
  for (const sj of Object.keys(LABEL)) for (const b of ['AQA', 'Edexcel']) expect(mine.filter(p => p.subject === sj && p.board === b).length, `${b} ${sj}`).toBeGreaterThan(0);
  expect(mine.filter(p => ['biology', 'chemistry', 'physics'].includes(p.subject)).every(p => p.comb), 'every science topic is also in Combined Science').toBe(true);
  expect(mine.map(p => p.topic).sort()).toEqual(['Atomic structure and the periodic table', 'Cell biology', 'Conservation of energy', 'Energy', 'Germany, 1890–1945: Democracy and dictatorship', 'Hazardous Earth', 'Key concepts in biology', 'Key concepts in chemistry', 'Number', 'Number', 'The challenge of natural hazards', 'Weather hazards and climate change', 'Weimar and Nazi Germany, 1918–39']);
});

test('every taster topic is deep enough for a full session of every game, in every way a teacher can use it', async () => {
  expect(AUDIT.length).toBe(TASTER.length);
  for (const r of AUDIT) for (const v of r.views) {
    const where = `${r.label} (${v.name})`;
    expect(v.n, `${where}: questions for Over the Edge, Outpace and Category Clash`).toBeGreaterThanOrEqual(MIN.all);
    expect(v.hex, `${where}: Hex Hunt questions (hexOk)`).toBeGreaterThanOrEqual(MIN.hex);
    v.d.forEach((x, i) => expect(x, `${where}: difficulty ${i + 1} (a Category Clash row each)`).toBeGreaterThanOrEqual(MIN.perDiff));
    expect(r.sub, `${r.label}: subtopics for Category Clash columns`).toBeGreaterThanOrEqual(2);
  }
});

test('the taster file holds the same Hex Hunt pool the audit counts, and nothing else', async ({ page }) => {
  await openBundle(page, '', { file: productOf(taster) });
  const got = await page.evaluate(() => CGB.bank.all().filter(s => s.builtin && !s.combined).map(s => ({ id: s.id, n: s.questions.length, hex: s.questions.filter(q => q.hexOk).length })));
  for (const r of AUDIT) {
    const g = got.find(x => x.id === r.id), all = r.views.find(v => v.name === 'all');
    expect(g, r.id).toBeTruthy();
    expect([g.n, g.hex], r.label).toEqual([all.n, all.hex]);
  }
});

test('the upgrade line is true, and only the taster says it', async ({ page }) => {
  const per = {}; packs.forEach(p => { per[p.subject] = (per[p.subject] || 0) + p.questions.length; });
  const over = Math.floor(Math.min(...Object.values(per)) / 100) * 100;
  await openBundle(page, '', { file: productOf(taster) });
  expect(await page.evaluate(() => CGB.upgradeText())).toBe(`Want more topics? Full subject packs, with over ${over.toLocaleString('en-GB')} questions each, are on Tes.`);
  for (const [sj, n] of Object.entries(per)) expect(n, `the ${sj} pack really has over ${over} questions`).toBeGreaterThan(over);
  const html = fs.readFileSync(productOf(taster), 'utf8');
  expect(html).toContain('Licensed for use in one teacher');
  expect(html).not.toMatch(/https?:\/\/[^"'\s<]*tes\.com|tes\.com/i);
  for (const ed of EDITIONS.filter(e => !e.taster)) {
    await page.goto('file://' + productOf(ed));
    await page.waitForFunction(() => window.CGB && CGB.app);
    expect(await page.evaluate(() => [CGB.upgradeText(), CGB.REVIEW_NOTE.includes('cgb-upgrade')]), ed.name).toEqual(['', false]);
  }
});
