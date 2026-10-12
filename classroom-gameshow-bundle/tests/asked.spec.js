// One record of "what has been asked" shared by all four games (bank.js asked / markAsked / preferred / retryTest):
// every game puts first the questions a playing team got wrong last time and the questions no game has asked yet,
// and only when neither is left, the one asked longest ago.
const { test } = require('@playwright/test');
const { openBundle, expect } = require('./helpers');

const PACK = 'aqa-biology-8461-4.1';      // Cell biology: 51 questions
const useTopic = page => page.evaluate(pack => { CGB.bank.setSubject('biology'); CGB.bank.setBoard('aqa'); CGB.bank.setCourse('separate'); CGB.bank.setHigher(true); CGB.bank.setSelection({ ids: [pack] }); return CGB.bank.active().questions.length; }, PACK);

test('the picker asks what no game has asked, then the longest ago, and the record is shared by every picker', async ({ page }) => {
  await openBundle(page);
  const n = await useTopic(page);
  expect(n).toBeGreaterThanOrEqual(40);
  const r = await page.evaluate(n => {
    const B = CGB.bank, key = q => B.qKey(q);
    B.clearAsked();
    const take = (k, picker) => Array.from({ length: k }, () => key(picker.pick([])));
    const a = take(20, B.createPicker());      // game 1
    const b = take(20, B.createPicker());      // game 2: a new picker, as at the start of every game
    const c = take(n - 40, B.createPicker());  // game 3: exactly what is left
    const all = new Set(a.concat(b, c));
    const again = B.createPicker().pick([]);   // everything has been asked: the one asked longest ago
    return { a, distinct: all.size, overlapAB: a.filter(k => b.includes(k)).length, overlapC: c.filter(k => a.includes(k) || b.includes(k)).length, longest: key(again), first: a[0], marked: Object.keys(B.asked()).length };
  }, n);
  expect(new Set(r.a).size).toBe(20);
  expect(r.overlapAB, 'the second game repeats nothing the first asked').toBe(0);
  expect(r.overlapC, 'the third repeats nothing either').toBe(0);
  expect(r.distinct, 'together they covered the whole topic once').toBe(n);
  expect(r.longest, 'with everything asked, the oldest comes back first').toBe(r.first);
  expect(r.marked).toBeGreaterThanOrEqual(n);
});

test('a question a team got wrong comes back first, then settles once it is answered right', async ({ page }) => {
  await openBundle(page);
  await useTopic(page);
  const r = await page.evaluate(() => {
    const B = CGB.bank, qs = B.active().questions, key = q => B.qKey(q);
    B.clearAsked();
    qs.forEach(q => B.markAsked(q));                       // everything has been asked once: the oldest is qs[0]
    const target = qs[7];
    B.logWrong('Owls', target, 'test');                    // Owls got it wrong
    const out = {};
    out.owls = key(B.createPicker().pick(['Owls'])) === key(target);       // Owls are playing: it comes straight back
    out.foxes = key(B.createPicker().pick(['Foxes'])) === key(qs[0]);       // another team is playing: nothing to come back to, so the oldest
    // it was asked again (the pick above marked it) and nobody got it wrong this time: it has settled
    out.settled = key(B.createPicker().pick(['Owls'])) !== key(target);
    B.logWrong('Owls', target, 'test');                    // wrong again: back on the list
    out.again = B.retryTest(['Owls'])(target) === true;
    B.unlogWrong('Owls', target);                          // class-mode undo takes the wrong answer back
    out.undone = B.retryTest(['Owls'])(target) === false;
    return out;
  });
  expect(r).toEqual({ owls: true, foxes: true, settled: true, again: true, undone: true });
});

test('with new questions and ones to come back to both on offer, about half the questions are ones to come back to', async ({ page }) => {
  await openBundle(page);
  await useTopic(page);
  const share = await page.evaluate(() => {
    const B = CGB.bank, qs = B.active().questions, key = q => B.qKey(q);
    let hits = 0; const trials = 80;
    for (let t = 0; t < trials; t++) {
      B.clearAsked();
      qs.slice(0, 20).forEach(q => B.markAsked(q));         // 20 asked, the rest new
      B.logWrong('Owls', qs[3], 'test');                    // one of the asked ones was got wrong
      if (key(B.createPicker().pick(['Owls'])) === key(qs[3])) hits++;
    }
    return hits / trials;
  });
  expect(share, 'the one wrong question comes up about half the time, not never and not always').toBeGreaterThan(0.3);
  expect(share).toBeLessThan(0.7);
});

test('every game starts with the one question no game has asked yet, and counts what it asks', async ({ page }) => {
  test.setTimeout(300000);
  await openBundle(page);
  await useTopic(page);
  // a target every game can use (Hex Hunt needs a question with a letter); everything else counts as already asked
  const prep = () => page.evaluate(() => {
    const B = CGB.bank; B.clearAsked();
    const qs = B.active().questions, target = qs.find(q => q.hexOk && B.difficultyOf(q) === 2);
    qs.forEach(q => { if (q !== target) B.markAsked(q); });
    return { key: B.qKey(target), q: target.q };
  });
  const askedNow = key => page.evaluate(k => CGB.bank.asked()[k], key);

  let t = await prep();
  await page.click('[data-play="over-the-edge"]'); await page.click('#ote-startBtn');
  await expect.poll(() => page.evaluate(() => (CGB.games['over-the-edge'].state().q || {}).q), 'Over the Edge opens with it').toBe(t.q);
  expect(await askedNow(t.key), 'and counts it as the most recent').toBeGreaterThan(50);
  await page.evaluate(() => CGB.app.show('launcher'));

  t = await prep();
  await page.click('[data-play="outpace"]'); await page.click('#op-startBtn'); await page.keyboard.press('2');
  await expect.poll(() => page.evaluate(() => (CGB.games.outpace.state().q || {}).q), 'Outpace opens with it').toBe(t.q);
  await page.evaluate(() => CGB.app.show('launcher'));

  t = await prep();
  await page.click('[data-play="category-clash"]'); await page.click('#cc-startBtn');
  expect(await page.evaluate(() => CGB.games['category-clash'].state().board), 'Category Clash puts it on the board').toContain(t.q);
  await page.evaluate(() => CGB.app.show('launcher'));

  t = await prep();
  await page.click('[data-play="hex-hunt"]'); await page.click('#hh-startBtn');
  await page.locator('#hh-board .hh-hex').first().click();
  await expect.poll(() => page.evaluate(() => { const q = CGB.games['hex-hunt'].state().q; return q && q.q; }), 'Hex Hunt opens with it').toBe(t.q);
  expect(await askedNow(t.key)).toBeGreaterThan(50);
});

test('Category Clash and Hex Hunt bring back a question a team got wrong', async ({ page }) => {
  test.setTimeout(300000);
  await openBundle(page);
  await useTopic(page);
  const prep = () => page.evaluate(() => {
    const B = CGB.bank, qs = B.active().questions; B.clearAsked();
    qs.forEach(q => B.markAsked(q));                                          // all asked once: no new questions left
    const target = qs.find(q => q.hexOk && B.difficultyOf(q) === 2);
    B.logWrong('Team 1', target, 'test'); B.logWrong('Left half', target, 'test');   // the default names of the teams in each game
    return target.q;
  });
  let q = await prep();
  await page.click('[data-play="category-clash"]'); await page.click('#cc-startBtn');
  expect(await page.evaluate(() => CGB.games['category-clash'].state().board), 'Category Clash puts it on the board').toContain(q);
  await page.evaluate(() => CGB.app.show('launcher'));
  q = await prep();
  await page.click('[data-play="hex-hunt"]'); await page.click('#hh-startBtn');
  await page.locator('#hh-board .hh-hex').first().click();
  await expect.poll(() => page.evaluate(() => { const s = CGB.games['hex-hunt'].state().q; return s && s.q; }), 'Hex Hunt asks it first').toBe(q);
});

test('Category Clash puts the questions no game has asked on its board, and counts a tile as asked when it opens', async ({ page }) => {
  await openBundle(page);
  await useTopic(page);
  // every question has been asked except one for each column (topic) and row (difficulty): the board is made of those
  const keep = await page.evaluate(() => {
    const B = CGB.bank; B.clearAsked();
    const qs = B.active().questions, pick = [];
    const col = q => q.subtopic || q.topic;
    [...new Set(qs.map(col))].forEach(t => [1, 2, 3].forEach(d => { const q = qs.find(x => col(x) === t && B.difficultyOf(x) === d); if (q) pick.push(q); }));
    qs.forEach(q => { if (!pick.includes(q)) B.markAsked(q); });
    return pick.map(q => q.q);
  });
  await page.click('[data-play="category-clash"]'); await page.click('#cc-startBtn');
  const tiles = await page.evaluate(() => CGB.games['category-clash'].state().board.filter(Boolean));
  expect(keep.length).toBeGreaterThanOrEqual(8);
  expect(tiles.filter(q => keep.includes(q)).length, 'the board is made of the unasked questions').toBeGreaterThanOrEqual(keep.length - 1);
  await page.locator('#cc-board .cc-tile[data-c]').first().click();
  const q = await page.evaluate(() => CGB.games['category-clash'].state().q);
  expect(q).toBeTruthy();
  expect(await page.evaluate(q => CGB.bank.asked()[CGB.bank.qKey(q)] > 0, q), 'opening a tile counts the question as asked').toBe(true);
});

test('the record survives a reload, carries over from the old Hex Hunt record, and is kept per subject, board and course', async ({ page }) => {
  await openBundle(page);
  await useTopic(page);
  const id = await page.evaluate(() => { CGB.bank.clearAsked(); const q = CGB.bank.active().questions[3]; CGB.bank.markAsked(q); return CGB.bank.qKey(q); });
  await page.reload(); await page.waitForFunction(() => window.CGB && CGB.app);
  await useTopic(page);
  expect(await page.evaluate(k => CGB.bank.asked()[k], id), 'still there after a reload').toBeGreaterThan(0);
  expect(await page.evaluate(() => { CGB.bank.setBoard('edexcel'); return Object.keys(CGB.bank.asked()).length; }), 'another board starts clean').toBe(0);
  // the record Hex Hunt kept on its own before every game shared one is picked up
  await page.evaluate(() => { localStorage.removeItem('cgb.asked'); localStorage.setItem('cgb.hh.recent', JSON.stringify({ 'biology|aqa|separate': { seq: 7, q: { 'aqa-bio-4.1-001': 7 } } })); });
  await page.reload(); await page.waitForFunction(() => window.CGB && CGB.app);
  await useTopic(page);
  expect(await page.evaluate(() => CGB.bank.asked()['aqa-bio-4.1-001'])).toBe(7);
});
