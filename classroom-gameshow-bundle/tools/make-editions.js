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
  const others = EDITIONS.filter(e => e.id !== ed.id && e.id !== 'free-taster').map(e => e.name);
  const what = c.map(x => {
    if (ed.taster) return `- **${x.label}:** ${list(x.names)} (${n(x.questions)} questions${SCIENCE.includes(x.sj) ? ', with the Combined Science version of the same topic' : ''})`;
    const base = `- **${x.label}:** ${x.topics} topic packs, ${n(x.questions)} questions (${x.specs.join(', ')})`;
    return SCIENCE.includes(x.sj) ? `${base}; Combined Science: ${x.combTopics} topics, ${n(x.combQuestions)} questions (${x.combSpecs.join(', ')})` : base;
  }).join('\n');
  const title = ed.taster ? 'Showtime: Classroom Gameshows, Free taster (4 revision games)' : `Showtime: Classroom Gameshows, ${ed.name} (4 revision games)`;
  const summary = ed.taster
    ? `Try all four whole-class revision games free: up to six teams answer every question on mini whiteboards. This free version includes one topic per subject (${subjects}), AQA-style and Edexcel-style, ${n(q)} questions. Full subject packs are available on Tes. One file, works offline.`
    : `Four whole-class revision games for your interactive whiteboard: up to six teams answer every question on mini whiteboards. Built-in AQA-style and Edexcel-style GCSE ${subjects} packs, one for every specification topic${ed.subjects.some(s => SCIENCE.includes(s)) ? ', for separate sciences and Combined Science' : ''} (${n(q)} questions). One file, works offline.`;
  return `# Tes listing: ${ed.name}

## Short title

${title}

## Summary line

${summary}

## Description

Turn any revision lesson into a gameshow. **Showtime: Classroom Gameshows** contains four original quiz games, inspired by classic TV quiz formats, built for the front of the classroom and run from a single file. There is nothing to install, no logins and no internet connection needed.

**What's in the ${ed.name}**
${what}

Every built-in question is tied to a specification point, tagged by difficulty, and marked if it is Higher tier only${c.some(x => ['history', 'geography'].includes(x.sj)) && c.length === 1 ? ' where the subject is tiered' : ''}, so a Foundation class can leave those out.${ed.subjects.some(s => SCIENCE.includes(s)) ? ' Choose Separate science or Combined Science on the menu: a Combined Science class only sees Combined Science content.' : ''}

**In every edition**
- All four games: **Over the Edge** (2 to 6 teams win counters and push them over the edge of a moving shelf), **Outpace** (the whole class is one runner, answering its way home before the Hunter catches it, with a 90-second Final Sprint), **Category Clash** (a board of topics and points values from 100 to 300) and **Hex Hunt** (two halves of the class race across a board of lettered hexagons, best of three rounds)
- Whole-class play: split the class into 2 to 6 teams with mini whiteboards; every team answers every question, and you mark each team right or wrong with one key. No pupil devices are needed
- Pause at any time, end a game early and still show the results, and a results screen listing the questions the class found hardest
- Write your own questions for any subject by pasting simple text, and a full question bank to manage, copy and back up your sets
- Wrong answers are saved for each team, so weak topics come round again more often
- Works offline from one file in Chrome, Edge, Firefox or Safari; keyboard shortcuts, mouse and touch for interactive whiteboards; large text and reduced motion options
- No data leaves your computer: no accounts, no tracking, no analytics
- Short teacher guide (PDF)

**Also available**
${ed.taster ? `The full editions on Tes: ${list(others)}.` : `Other editions on Tes: ${list(others)}, and a free taster with one topic per subject.`}

**What's included**
- ${ed.file} (the games)
- teacher-guide.pdf

The question packs are AQA-style and Edexcel-style practice questions written for Showtime to match the AQA and Pearson Edexcel GCSE specifications. They are not produced or endorsed by AQA or Pearson. The games are original works and are not connected with any television programme or broadcaster.

## Suggested tags

revision game, quiz, gameshow, interactive whiteboard, ${c.map(x => 'GCSE ' + x.label.toLowerCase()).join(', ')}, retrieval practice, team game, mini whiteboards, whole class, starter, plenary, KS4, offline

## Images

cover.png, over-the-edge.png, outpace.png, category-clash.png, hex-hunt.png, question-bank.png (in this folder)
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
