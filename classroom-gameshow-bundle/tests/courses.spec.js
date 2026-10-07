// Combined Science and separate sciences, and the Higher tier switch (version 1.4).
// The course tags come from tools/spec-map through tools/packs.js; these tests check the tags
// themselves, then that the main screen and every game only ever see what the chosen course allows.
const fs = require('fs');
const path = require('path');
const { test } = require('@playwright/test');
const { openBundle, expect } = require('./helpers');
const { loadAll, loadMap } = require('../tools/packs.js');

const SCIENCES = ['biology', 'chemistry', 'physics'];
const { packs } = loadAll();
const COMB_NUMBERING = { AQA: /^[456](\.\d+){1,4}$/, Edexcel: /^\d+\.\d+$/ };

test('every science question has course and tier tags for its board', () => {
  for (const board of ['AQA', 'Edexcel']) for (const sj of SCIENCES) {
    const map = loadMap(board, sj);
    expect(map, `tools/spec-map/${board.toLowerCase()}-${sj}.json`).toBeTruthy();
    expect(map.combined.specCode).toBe(board === 'AQA' ? '8464' : '1SC0');
    expect(map.combined.version).toBeTruthy();
    const mine = packs.filter(p => p.board === board && p.subject === sj);
    expect(mine.length).toBeGreaterThan(0);
    for (const p of mine) {
      // a topic is in Combined Science unless every question in it is separate science only
      expect(!!p.comb, p.id).toBe(p.questions.some(q => !q.sepOnly));
      for (const q of p.questions) {
        expect(['Higher', 'Foundation and Higher'], q.id).toContain(q.tier);
        if (q.sepOnly) { expect(q.combRef, q.id).toBeUndefined(); continue; }
        expect(q.combRef, q.id).toMatch(COMB_NUMBERING[board]);
        if (board === 'Edexcel') expect(q.combRef, q.id).toBe(q.specRef);   // Pearson keeps the numbers
        expect(['Higher', 'Foundation and Higher'], q.id).toContain(q.combTier);
        // a point that is Higher only in either course makes the question Higher there
        const pt = map.points[q.specRef] || Object.entries(map.points).filter(([r]) => q.specRef.startsWith(r + '.')).sort((a, b) => b[0].length - a[0].length)[0][1];
        if (pt.tier === 'Higher') expect(q.tier, q.id).toBe('Higher');
        if (pt.combTier === 'Higher') expect(q.combTier, q.id).toBe('Higher');
      }
    }
  }
  // the map files say which specification versions they were built from
  for (const f of fs.readdirSync(path.join(__dirname, '..', 'tools', 'spec-map'))) {
    const m = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'tools', 'spec-map', f), 'utf8'));
    expect(m.separate.version, f).toBeTruthy();
    expect(m.combined.source, f).toMatch(/^https:\/\//);
  }
});

test('Combined Science never shows a separate-science-only question, and uses its own topic names and order', async ({ page }) => {
  const log = await openBundle(page);
  for (const board of ['aqa', 'edexcel']) for (const sj of SCIENCES) {
    const map = loadMap(board === 'aqa' ? 'AQA' : 'Edexcel', sj);
    const sepOnly = new Set(packs.filter(p => p.board.toLowerCase() === board && p.subject === sj).flatMap(p => p.questions.filter(q => q.sepOnly).map(q => q.id)));
    const r = await page.evaluate(([sj, board]) => {
      const B = CGB.bank; B.setSubject(sj); B.setBoard(board); B.setCourse('combined'); B.setHigher(true);
      const list = B.packs().map(p => { B.setSelection({ ids: [p.id] }); return { ref: p.ref, topic: p.topic, specCode: p.specCode, ids: B.active().questions.map(q => q.id) }; });
      B.setSelection({ mixed: B.courses()[0].specCode });
      const mixed = B.active();
      return { list, mixed: mixed.questions.map(q => q.id), short: mixed.short, courses: B.courses().map(c => c.specCode) };
    }, [sj, board]);
    expect(r.courses).toEqual([map.combined.specCode]);
    expect(r.short).toBe('Mixed: all Combined Science topics');
    expect(r.list.map(p => p.ref + ' ' + p.topic)).toEqual(map.topics.map(t => t.comb + ' ' + t.combTitle));
    for (const p of r.list) expect(p.ids.filter(id => sepOnly.has(id)), `${board} ${sj} ${p.ref}`).toEqual([]);
    expect(r.mixed.filter(id => sepOnly.has(id))).toEqual([]);
    expect(r.mixed.length).toBeGreaterThan(200);
    // Separate science shows everything
    const all = await page.evaluate(() => { const B = CGB.bank; B.setCourse('separate'); B.setSelection({ mixed: B.courses()[0].specCode }); return B.active().questions.length; });
    expect(all).toBe(packs.filter(p => p.board.toLowerCase() === board && p.subject === sj).reduce((n, p) => n + p.questions.length, 0));
  }
  expect(log.errors).toEqual([]);
});

test('with "Include Higher tier questions" unticked, no Higher-only question for the chosen course appears (sciences and Maths)', async ({ page }) => {
  await openBundle(page);
  const combinedTier = new Map(packs.flatMap(p => p.questions.filter(q => q.combTier).map(q => [q.id, q.combTier])));
  for (const board of ['aqa', 'edexcel']) for (const [sj, course] of [...SCIENCES.flatMap(s => [[s, 'separate'], [s, 'combined']]), ['maths', 'separate']]) {
    const r = await page.evaluate(([sj, board, course]) => {
      const B = CGB.bank; B.setSubject(sj); B.setBoard(board); B.setCourse(course);
      const get = on => { B.setHigher(on); B.setSelection({ mixed: B.courses()[0].specCode }); return B.active().questions.map(q => ({ id: q.id, tier: q.tier })); };
      const withH = get(true), withoutH = get(false);
      B.setHigher(true);
      return { withH, withoutH, note: (B.setHigher(false), B.subjectNote()) };
    }, [sj, board, course]);
    const label = `${board} ${sj} ${course}`;
    expect(r.withH.some(q => q.tier === 'Higher'), label).toBe(true);
    expect(r.withoutH.filter(q => q.tier === 'Higher'), label).toEqual([]);
    expect(r.withoutH.length, label).toBe(r.withH.filter(q => q.tier !== 'Higher').length);
    expect(r.note, label).toMatch(/Foundation tier/);
    // in Combined Science the tier is the Combined Science one
    if (course === 'combined') for (const q of r.withH) expect(q.tier, q.id).toBe(combinedTier.get(q.id));
  }
  await page.evaluate(() => CGB.bank.setHigher(true));
});

// saved data on file:// pages can be disturbed by tests running alongside; one retry (passes on its own)
test.describe(() => { test.describe.configure({ retries: 1 }); test('the course choice shows only for the sciences, is remembered, and the Higher tier checkbox is hidden for History and Geography', async ({ page }) => {
  await openBundle(page);
  for (const [sj, course, higher] of [['biology', true, true], ['chemistry', true, true], ['physics', true, true], ['maths', false, true], ['history', false, false], ['geography', false, false]]) {
    await page.selectOption('#subjectSelect', sj);
    await expect(page.locator('#courseRow'), sj).toBeVisible({ visible: course });
    await page.click('#packBtn');
    await expect(page.locator('#pkHigher'), sj).toHaveCount(higher ? 1 : 0);
    await page.click('#packMenu [data-done]');
  }
  // History and Geography are not tiered: an unticked box from another subject leaves nothing out
  await page.evaluate(() => CGB.bank.setHigher(false));
  await page.selectOption('#subjectSelect', 'history');
  expect(await page.evaluate(() => CGB.bank.active().withoutHigher)).toBe(0);
  await page.evaluate(() => CGB.bank.setHigher(true));
  // the course is remembered between sessions
  await page.selectOption('#subjectSelect', 'physics');
  await page.selectOption('#courseSelect', 'combined');
  await expect(page.locator('#packBtn')).toContainText('Combined Science');
  await page.reload();
  await page.waitForFunction(() => window.CGB && CGB.app);
  await expect(page.locator('#courseSelect')).toHaveValue('combined');
  await expect(page.locator('#packBtn')).toContainText('Combined Science');
  await page.selectOption('#courseSelect', 'separate');
});});


/* Enough questions for each game, whatever is chosen: Category Clash fills a 4 × 3 board (12),
   Hex Hunt a 4 × 4 board of letters (16), and Over the Edge and Outpace a game without repeats
   (about 20). Checked for every topic and "Mixed", both boards, both courses, with and without Higher. */
test('every topic has enough questions for each game after filtering', async ({ page }) => {
  await openBundle(page);
  const rows = await page.evaluate(() => {
    const B = CGB.bank, out = [];
    for (const sj of ['biology', 'chemistry', 'physics', 'maths', 'history', 'geography']) for (const bd of ['aqa', 'edexcel'])
      for (const co of (B.hasCourses(sj) ? ['separate', 'combined'] : ['separate'])) for (const hi of [true, false]) {
        B.setSubject(sj); B.setBoard(bd); B.setCourse(co); B.setHigher(hi);
        const sels = B.packs().map(p => ({ ids: [p.id] })).concat(B.courses().map(c => ({ mixed: c.specCode })));
        for (const s of sels) {
          B.setSelection(s);
          const qs = B.active().questions;
          out.push({ at: `${bd} ${sj} ${co} ${hi ? 'with' : 'without'} Higher ${s.ids ? s.ids[0] : 'mixed'}`, n: qs.length, letters: qs.filter(q => q.hexOk !== false && CGB.hexLetter(q.a)).length });
        }
      }
    B.setHigher(true); B.setCourse('separate'); B.setSubject('biology'); B.setBoard('aqa');
    return out;
  });
  expect(rows.length).toBeGreaterThan(300);
  expect(rows.filter(r => r.n < 20).map(r => r.at + ': ' + r.n)).toEqual([]);
  expect(rows.filter(r => r.letters < 16).map(r => r.at + ': ' + r.letters)).toEqual([]);
});

test('Category Clash: every Combined Science topic fills the whole board at Foundation tier', async ({ page }) => {
  await openBundle(page, '#category-clash');
  const all = await page.evaluate(() => {
    const B = CGB.bank, out = [];
    B.setSubject('physics'); B.setHigher(false); B.setCourse('combined');
    for (const sj of ['biology', 'chemistry', 'physics']) for (const bd of ['aqa', 'edexcel']) { B.setSubject(sj); B.setBoard(bd); B.packs().forEach(p => out.push([sj, bd, p.id])); }
    return out;
  });
  for (const [sj, bd, id] of all) {
    await page.evaluate(([sj, bd, id]) => { const B = CGB.bank; B.setSubject(sj); B.setBoard(bd); B.setSelection({ ids: [id] }); }, [sj, bd, id]);
    await page.click('#cc-startBtn');
    await expect(page.locator('#cc-board .cc-tile').first()).toBeVisible();
    expect(await page.locator('#cc-board .cc-tile.empty').count(), id).toBe(0);
    await page.reload();
    await page.waitForFunction(() => window.CGB && CGB.app);
  }
  await page.evaluate(() => { const B = CGB.bank; B.setHigher(true); B.setCourse('separate'); });
});
