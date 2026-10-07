// The built-in question packs: the checks from the question banks brief, run on every pack.
// tools/packs.js does the checking (it is the same code the build and "node tools/packs.js check"
// use); these tests fail on any error, and check the packs in the built file match the sources.
const { test } = require('@playwright/test');
const { openBundle, expect } = require('./helpers');
const P = require('../tools/packs.js');
const hexLetter = require('../src/shared/hexletter.js');

const all = P.loadAll();

test('every pack passes every check: fields, specification references, Higher tier, duplicates, answer length, counts and spread', () => {
  // the errors themselves are the useful failure message
  expect(all.errors).toEqual([]);
  expect(all.packs.length).toBeGreaterThan(0);
});

test('every question has a stable id, and no id is used twice', () => {
  const ids = all.packs.flatMap(p => p.questions.map(q => q.id));
  expect(ids.filter(id => !id)).toEqual([]);
  expect(new Set(ids).size).toBe(ids.length);
});

test('every specification reference is a real point of that specification, in its own numbering', () => {
  all.packs.forEach(p => {
    const spec = P.loadSpec(p.board, p.specCode);
    const numbering = new RegExp(spec.numbering);
    p.questions.forEach(q => {
      expect(numbering.test(q.specRef), `${q.id} ${q.specRef}`).toBe(true);
      expect(spec.refs[q.specRef], `${q.id} ${q.specRef}`).toBeTruthy();
    });
  });
});

test('Hex Hunt: every hexOk answer starts with its letter after ignoring the/a/an, and at least a third of each topic can go on the board', () => {
  all.packs.forEach(p => {
    p.questions.forEach(q => {
      const L = hexLetter(q.a);
      expect(q.hexOk, q.id).toBe(!!L);
      if (L) {
        const first = q.a.trim().replace(/^((in|into|on|at|from)\s+)?((the|a|an)\s+)?/i, '')[0].toUpperCase();
        expect(first, `${q.id}: ${q.a}`).toBe(L);
      }
    });
    expect(p.stats.hex / p.stats.n, p.id).toBeGreaterThanOrEqual(1 / 3);
  });
});

test('each topic has at least 30 questions (3 per specification point when larger), each difficulty at least a fifth', () => {
  all.packs.forEach(p => {
    expect(p.stats.n, p.id).toBeGreaterThanOrEqual(Math.max(30, p.stats.target));
    p.stats.byD.forEach(c => expect(c / p.stats.n, p.id).toBeGreaterThanOrEqual(0.2));
  });
});

test('answers are short enough for a mini whiteboard, and no question repeats or nearly repeats another', () => {
  const words = s => s.trim().split(/\s+/).length;
  all.packs.forEach(p => p.questions.forEach(q => expect(words(q.a), `${q.id}: ${q.a}`).toBeLessThanOrEqual(P.LIMITS.answerWords)));
  const bySubject = {};
  all.packs.forEach(p => { (bySubject[p.board + p.subject] = bySubject[p.board + p.subject] || []).push(...p.questions); });
  Object.values(bySubject).forEach(qs => {
    const seen = new Set();
    qs.forEach(q => { const k = P.norm(q.q); expect(seen.has(k), q.id).toBe(false); seen.add(k); });
  });
});

test('the built file carries every pack, with the same questions, and the games read them', async ({ page }) => {
  await openBundle(page);
  // the Combined Science packs are views of these (tests/courses.spec.js)
  const built = await page.evaluate(() => CGB.bank.all().filter(s => s.builtin && !s.combined).map(s => ({ id: s.id, n: s.questions.length, ids: s.questions.map(q => q.id), hex: s.questions.map(q => q.hexOk === !!CGB.hexLetter(q.a)) })));
  expect(built.map(b => b.id).sort()).toEqual(all.packs.map(p => p.id).sort());
  built.forEach(b => {
    const src = all.packs.find(p => p.id === b.id);
    expect(b.ids, b.id).toEqual(src.questions.map(q => q.id));
    expect(b.hex.every(Boolean), b.id).toBe(true);
  });
  // a built-in question carries everything the brief asks for
  const q = await page.evaluate(() => CGB.bank.all().find(s => s.builtin).questions[0]);
  ['id', 'board', 'subject', 'specCode', 'specRef', 'topic', 'subtopic', 'tier', 'difficulty', 'q', 'a', 'accept', 'hexOk', 'notes'].forEach(k => expect(q, k).toHaveProperty(k));
});

test('the packs are a reasonable size inside the single file', async () => {
  const fs = require('fs');
  const path = require('path');
  const size = fs.statSync(path.join(__dirname, '..', 'dist', 'showtime-classroom-gameshows.html')).size;
  // the whole product, with every pack, stays small enough to open quickly from a school drive
  expect(size).toBeLessThan(6 * 1024 * 1024);
});
