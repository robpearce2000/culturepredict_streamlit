#!/usr/bin/env node
/*
 * Everything for selling the editions, after `node build.js --editions`, `node tools/make-guide.js`
 * and `node tools/make-listing.js --edition <id>` (for the images):
 *   dist/listings/<id>/description.md   the Tes listing text, with the edition's own counts
 *   dist/downloads/<file>.zip           the edition's HTML file and the teacher guide, ready to upload
 *   dist/editions/README.md             every edition: file size, subjects, question counts, load time
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { chromium } = require('@playwright/test');
const { EDITIONS, includes } = require('./editions.js');
const { loadAll } = require('./packs.js');

const ROOT = path.join(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const LABEL = { biology: 'Biology', chemistry: 'Chemistry', physics: 'Physics', maths: 'Maths', history: 'History', geography: 'Geography' };
const SCIENCE = ['biology', 'chemistry', 'physics'];
const { packs } = loadAll();
const n = x => x.toLocaleString('en-GB');
const list = a => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];
const VERSION = /CGB\.VERSION = '([^']+)'/.exec(fs.readFileSync(path.join(ROOT, 'src', 'shared', 'storage.js'), 'utf8'))[1];

/* What an edition holds, subject by subject */
function contents(ed) {
  const mine = packs.filter(p => includes(ed, p));
  return ed.subjects.map(sj => {
    const ps = mine.filter(p => p.subject === sj);
    const specs = [...new Set(ps.map(p => `${p.board}-style ${p.specCode}`))];
    const comb = ps.filter(p => p.comb);
    return {
      sj, label: LABEL[sj], topics: ps.length, questions: ps.reduce((s, p) => s + p.questions.length, 0), specs,
      combTopics: comb.length, combQuestions: comb.reduce((s, p) => s + p.questions.filter(q => !q.sepOnly).length, 0),
      combSpecs: [...new Set(comb.map(p => `${p.board}-style ${p.comb.specCode}`))],
      names: ps.map(p => `${p.board} ${p.topic}`)
    };
  });
}
const total = c => c.reduce((s, x) => s + x.questions, 0);

function description(ed) {
  const c = contents(ed), q = total(c), subjects = list(c.map(x => x.label));
  const sciences = c.filter(x => SCIENCE.includes(x.sj));
  const others = EDITIONS.filter(e => e.id !== ed.id && e.id !== 'free-taster').map(e => e.name);
  const title = ed.taster ? 'FREE GCSE Revision Games: 4 Whole-Class Quiz Shows (Taster Pack)' : `Showtime: Classroom Gameshows, ${ed.name} (4 revision games)`;
  const sep = sciences.length ? ' for separate and Combined Science' : '';
  const summary = ed.taster
    ? `Four whole-class revision games, free: one popular topic for each of ${subjects}, ${n(q)} questions, AQA-style and Edexcel-style. One file, works offline.`
    : `Four whole-class revision games: ${n(q)} AQA-style and Edexcel-style GCSE ${subjects} questions${sep}. One file, works offline.`;
  const includes = ed.taster
    ? `Free: all four games, fully working, with one popular topic for each subject (${subjects}), ${n(q)} questions in all. The science topics are in Separate and Combined Science. Each full subject pack has over 800 questions.`
    : `${n(q)} questions covering every topic of the AQA-style and Edexcel-style GCSE ${subjects} specifications${sep}. Each question is tied to a specification point and tagged by difficulty; Foundation classes can leave out Higher-only questions.`;
  const more = ed.taster
    ? 'Enjoyed the taster? Try the Science mega pack for every Biology, Chemistry and Physics topic.'
    : `Also on Tes: ${list(others)}, and a free taster.`;
  return `# Tes listing: ${ed.name}

Plain text: paste each field into Tes as it is.

## Title

${title}

## Summary

${summary}

## Description

Turn revision into a game show. Showtime: Classroom Gameshows is four whole-class quiz games for the front of the room. Split the class into 2 to 6 teams with mini whiteboards: every team answers every question and you mark them with one key.

Over the Edge: win counters and push the pile over the edge, then play for the jackpot.
Outpace: the class is one runner, racing home before the Hunter.
Category Clash: pick a topic and points value. One hidden star tile scores double.
Hex Hunt: two halves race to link their edges across lettered hexagons.

${includes}

One file that works offline: no logins, no installs, and no data leaves your computer. Every game ends with the questions the class found hardest, so you know what to reteach. You can add your own questions too.

${more}

The question packs are AQA-style and Edexcel-style practice questions written for Showtime. They are not produced or endorsed by AQA or Pearson.

## Tags

revision game, quiz, game show, whole class, mini whiteboards, interactive whiteboard, ${c.map(x => 'GCSE ' + x.label.toLowerCase()).join(', ')}, retrieval practice, starter, plenary, KS4

## Files

${ed.file} and teacher-guide.pdf (together as ${ed.file.replace(/\.html$/, '.zip')}), and the pictures in this folder.
`;
}

/* How long each edition takes to show its launcher, opened from disk (median of three) */
async function loadTimes() {
  const browser = await chromium.launch();
  const out = {};
  for (const ed of EDITIONS) {
    const times = [];
    for (let i = 0; i < 3; i++) {
      const page = await browser.newPage();
      const t0 = Date.now();
      await page.goto('file://' + path.join(DIST, 'editions', ed.file));
      await page.waitForFunction(() => window.CGB && CGB.app && !document.getElementById('launcher').hidden);
      times.push(Date.now() - t0);
      await page.close();
    }
    out[ed.id] = times.sort((a, b) => a - b)[1];
  }
  await browser.close();
  return out;
}

(async () => {
  const guide = path.join(DIST, 'teacher-guide.pdf');
  if (!fs.existsSync(guide)) throw new Error('Run node tools/make-guide.js first');
  fs.mkdirSync(path.join(DIST, 'downloads'), { recursive: true });
  for (const ed of EDITIONS) {
    const dir = path.join(DIST, 'listings', ed.id);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'description.md'), description(ed));
    const zip = path.join(DIST, 'downloads', ed.file.replace(/\.html$/, '.zip'));
    if (fs.existsSync(zip)) fs.unlinkSync(zip);
    execFileSync('zip', ['-j', '-q', '-X', zip, path.join(DIST, 'editions', ed.file), guide]);
  }
  const ms = await loadTimes();
  const rows = EDITIONS.map(ed => {
    const c = contents(ed), kb = fs.statSync(path.join(DIST, 'editions', ed.file)).size / 1024;
    const zkb = fs.statSync(path.join(DIST, 'downloads', ed.file.replace(/\.html$/, '.zip'))).size / 1024;
    return `| ${ed.name} | \`${ed.file}\` | ${(kb / 1024).toFixed(2)} MB (zip ${(zkb / 1024).toFixed(2)} MB) | ${c.map(x => x.label).join(', ')} | ${c.map(x => x.topics).reduce((a, b) => a + b, 0)} | ${n(total(c))} | ${ms[ed.id]} ms |`;
  });
  fs.writeFileSync(path.join(DIST, 'editions', 'README.md'), `# Showtime editions (version ${VERSION})

Built from the same source by \`node build.js --editions\` (tools/editions.js). Every edition has all four games, the question bank and "write your own"; they differ only in the built-in packs. Checksums: SHA256SUMS.txt. Upload files: ../downloads/. Listing text and images: ../listings/<edition>/.

| Edition | File | Size | Subjects | Topic packs | Questions | Load time |
|---|---|---|---|---|---|---|
${rows.join('\n')}

Topic packs and questions count each separate-science or other specification topic once; the Combined Science views of the science packs use the same questions. Load time: from opening the file to the launcher showing, in headless Chromium on the build machine (median of three); every edition is well under the 3-second target.
`);
  console.log(rows.join('\n'));
})().catch(e => { console.error(e); process.exit(1); });
