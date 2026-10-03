#!/usr/bin/env node
/*
 * Searches everything a teacher or buyer can see for names and terms that
 * belong to the TV formats the games were inspired by. Run before release.
 *   node tools/check-banned.js
 * Exit code 1 if anything is found.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
// Words are split so this file itself is the only place the full terms appear.
const TERMS = [
  ['the ', 'chase'], ['chas', 'er'], ['chas', 'ers'], ['tipping ', 'point'], ['beat the ', 'chasers'],
  ['cash ', 'builder'], ['final ', 'chase'], ['offer ', 'round'], ['drop ', 'zone'], ['drop-', 'zone'], ['drop', 'zone'],
  ['mystery ', 'counter'], ['out', 'run'], ['i', 'tv'],
  ['bradley ', 'walsh'], ['ben ', 'shephard'], ['anne ', 'hegerty'], ['mark ', 'labbett'], ['shaun ', 'wallace'],
  ['paul ', 'sinha'], ['jenny ', 'ryan'], ['darragh ', 'ennis'], ['govern', 'ess'], ['dark ', 'destroyer'],
  ['sinner', 'man'], ['the ', 'vixen'], ['the ', 'menace'], ['the ', 'beast'], ['the ', 'bolt'],
  ['mr ', 'pearce']
].map(p => p.join(''));
// "chase" on its own is a normal word, but the bundle avoids it entirely to be safe
const PATTERNS = TERMS.map(t => new RegExp('\\b' + t.replace(/[-]/g, '\\-').replace(/ /g, '\\s+') + '\\b', 'i'))
  .concat([/\bchas(e|ed|es|ing)\b/i]);

function pdfText(file) {
  try { return execFileSync('pdftotext', ['-layout', file, '-'], { encoding: 'utf8' }); }
  catch (e) { return null; }
}
const targets = [
  ['dist/classroom-gameshow-bundle.html', 'text'],
  ['dist/teacher-guide.pdf', 'pdf'],
  ['dist/listing/description.md', 'text'],
  ['docs/teacher-guide.html', 'text'],
  ['README.md', 'text'], ['CHANGELOG.md', 'text'], ['DECISIONS.md', 'text'], ['LICENSES.md', 'text']
];
let bad = 0, checked = 0;
targets.forEach(([rel, kind]) => {
  const file = path.join(ROOT, rel);
  if (!fs.existsSync(file)) { console.log(`skip (missing): ${rel}`); return; }
  const text = kind === 'pdf' ? pdfText(file) : fs.readFileSync(file, 'utf8');
  if (text == null) { console.log(`skip (no pdftotext): ${rel}`); return; }
  checked++;
  // file names in dist and listing count too
  PATTERNS.forEach(re => {
    const m = text.match(new RegExp(re.source, 'gi'));
    if (m) { bad += m.length; console.log(`FOUND in ${rel}: "${m[0]}" ×${m.length}`); }
  });
});
['dist', 'dist/listing'].forEach(d => {
  const dir = path.join(ROOT, d);
  if (!fs.existsSync(dir)) return;
  fs.readdirSync(dir).forEach(f => PATTERNS.forEach(re => { if (re.test(f.replace(/[-_.]/g, ' '))) { bad++; console.log(`FOUND in file name: ${d}/${f}`); } }));
});
console.log(bad ? `\n${bad} banned term(s) found.` : `\nNo banned terms found in ${checked} files (${PATTERNS.length} patterns).`);
process.exit(bad ? 1 : 0);
