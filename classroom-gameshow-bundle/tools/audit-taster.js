#!/usr/bin/env node
/*
 * Audits the free taster's topics (tools/editions.js): per pack and for each way a teacher can use it (all questions,
 * Foundation only, Combined Science, Combined and Foundation), how many questions there are, how many can go on a
 * Hex Hunt board (hexOk), the difficulty spread, and the subtopics (Category Clash columns).
 *   node tools/audit-taster.js            print the table
 *   node tools/audit-taster.js --check    exit 1 if any pack is too thin (see MIN)
 * Needs per session: Over the Edge 10 (6 + a 4-question final), Outpace deal round (8) + sprint (about 25),
 * Category Clash 12 tiles (4 categories × 3, one per difficulty row), Hex Hunt 30 usable (4×4 board, best of three).
 */
'use strict';
const P = require('./packs.js');
const E = require('./editions.js');
const ed = E.byId('free-taster');
const MIN = { all: 35, hex: 30, perDiff: 4 };   // Outpace is the hungriest game: a deal round (8) plus a 90-second sprint (about 25)

const all = P.loadAll().packs.filter(p => E.includes(ed, p));
const SCI = ['biology', 'chemistry', 'physics'];
function views(p) {
  const qs = p.questions, sci = SCI.includes(p.subject);
  const f = (q, o) => (!o.found || q.tier !== 'Higher') && (!o.comb || !q.sepOnly);
  const out = { 'all': {}, 'Foundation': { found: 1 } };
  if (sci) { out.Combined = { comb: 1 }; out['Combined, Foundation'] = { comb: 1, found: 1 }; }
  return Object.entries(out).map(([name, o]) => {
    const list = qs.filter(q => f(q, o));
    const d = [1, 2, 3].map(k => list.filter(q => q.difficulty === k).length);
    return { name, n: list.length, hex: list.filter(q => q.hexOk).length, d };
  });
}
const rows = all.map(p => ({ id: p.id, label: `${p.board} ${p.subject}: ${p.topicRef} ${p.topic}`, sub: p.subtopics.length, views: views(p) }));
if (require.main === module) {
  let bad = 0;
  for (const r of rows) {
    console.log(`${r.label} (${r.sub} subtopics)`);
    for (const v of r.views) {
      const thin = v.n < MIN.all || v.hex < MIN.hex || v.d.some(x => x < MIN.perDiff);
      if (thin) bad++;
      console.log(`   ${v.name.padEnd(22)} ${String(v.n).padStart(4)} questions  Hex Hunt ${String(v.hex).padStart(3)}  difficulty 1/2/3: ${v.d.join('/')}${thin ? '   <- THIN' : ''}`);
    }
  }
  if (process.argv.includes('--check') && bad) { console.error(`${bad} thin view(s)`); process.exit(1); }
}
module.exports = { rows, MIN };
