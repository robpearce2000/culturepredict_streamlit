#!/usr/bin/env node
/* Question review helper: prints one pack's questions with their course and tier tags, as JSON.
   node tools/review/export.js packs/aqa/biology/4.1-cell-biology.txt [--notes]   */
'use strict';
const P = require('../packs.js');
const file = process.argv[2], notes = process.argv.includes('--notes');
const built = P.buildPack(P.parseFile(file));
const pack = built.pack || built;
const qs = (pack.questions || []).map(q => {
  const o = { id: q.id, ref: q.specRef, d: q.difficulty, tier: q.tier === 'Higher' ? 'Higher only' : 'Both',
    course: q.sepOnly ? 'Separate only' : 'Separate and Combined', combRef: q.combRef || '', combTier: q.combTier === 'Higher' ? 'Higher only' : (q.sepOnly ? '' : 'Both'),
    calcTag: !!q.calc, hexOk: !!q.hexOk, q: q.q, a: q.a, accept: q.accept };
  if (q.rp) o.rp = q.rp;
  if (notes && q.notes) o.note = q.notes;
  return o;
});
console.log(JSON.stringify({ file, board: pack.board || (qs[0] && qs[0].id.split('-')[0]), topic: pack.topic || '', count: qs.length, questions: qs }, null, 1));
