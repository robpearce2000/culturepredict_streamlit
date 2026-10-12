#!/usr/bin/env node
/*
 * Makes the Tes listing images from the built bundle (run `node build.js` first):
 *   dist/listing/cover.png                1600×1200  (4:3, the shape Tes crops to)  wordmark, host and all four games
 *   dist/listing/over-the-edge.png        1600×1200  whole-class game, six teams
 *   dist/listing/outpace.png              1600×1200  whole-class game, six teams
 *   dist/listing/category-clash.png       1600×1200  whole-class game, six teams
 *   dist/listing/hex-hunt.png             1600×1200  whole-class game, two halves
 *   dist/listing/question-bank.png        1600×1200  the shared question bank
 *
 *   node tools/make-listing.js --edition <id>   the same images for one edition (tools/editions.js),
 *     from dist/editions/<file> into dist/listings/<id>/, showing that edition's subject in every
 *     game and its name on the cover (run `node build.js --editions` first)
 *   ... --edition <id> --reuse <other id>       only the cover is new; the game and question bank
 *     pictures are copied from another edition's listing (e.g. the science pack uses Chemistry's)
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');

const ROOT = path.join(__dirname, '..');
const ED = (() => { const i = process.argv.indexOf('--edition'); return i > 0 ? require('./editions.js').byId(process.argv[i + 1]) : null; })();
if (process.argv.includes('--edition') && !ED) { console.error('Unknown edition'); process.exit(1); }
const REUSE = (() => { const i = process.argv.indexOf('--reuse'); return i > 0 ? process.argv[i + 1] : null; })();
const URL = 'file://' + (ED ? path.join(ROOT, 'dist', 'editions', ED.file) : path.join(ROOT, 'dist', 'showtime-classroom-gameshows.html'));
const OUT = ED ? path.join(ROOT, 'dist', 'listings', ED.id) : path.join(ROOT, 'dist', 'listing');
// the subject the screenshots show: the edition's own (Chemistry for the science pack)
const SHOW = ED ? (ED.id === 'science' ? 'chemistry' : ED.subjects[0]) : 'chemistry';
// a teacher's own set for the question bank picture, in that subject
const OWN = {
  biology: ['Year 10 cells quiz', 'Subject: Biology\nTopic: Cells\nQ: Which organelle controls the cell?\nA: Nucleus\nQ: Where does aerobic respiration happen?\nA: Mitochondria'],
  chemistry: ['Year 10 bonding quiz', 'Subject: Chemistry\nTopic: Bonding\nQ: What type of bonding is in sodium chloride?\nA: Ionic\nQ: What type of bonding is in methane?\nA: Covalent'],
  physics: ['Year 10 energy quiz', 'Subject: Physics\nTopic: Energy\nQ: What is the unit of energy?\nA: Joule\nQ: What is the unit of power?\nA: Watt'],
  maths: ['Year 10 number quiz', 'Subject: Maths\nTopic: Number\nQ: What is 15% of 60?\nA: 9\nQ: What is the square root of 144?\nA: 12'],
  history: ['Year 10 Germany quiz', 'Subject: History\nTopic: Germany\nQ: In which year did Hitler become Chancellor?\nA: 1933\nQ: What was the Nazi secret police called?\nA: Gestapo'],
  geography: ['Year 10 hazards quiz', 'Subject: Geography\nTopic: Hazards\nQ: Which scale measures earthquake magnitude?\nA: Moment magnitude scale\nQ: What is the centre of a tropical storm called?\nA: The eye']
};
const ARGS = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

async function until(page, fn, ms, arg) {
  const end = Date.now() + (ms || 60000);
  while (Date.now() < end) { if (await page.evaluate(fn, arg)) return; await page.waitForTimeout(150); }
  const st = await page.evaluate(() => { const g = CGB.app.current, s = g && CGB.games[g] && CGB.games[g].state(); return JSON.stringify(s && { phase: s.phase, step: s.step, round: s.round }); });
  throw new Error('timed out waiting: ' + fn.toString().slice(0, 80) + ' state ' + st);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ARGS });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(URL);
  await page.waitForFunction(() => window.CGB && CGB.app);
  await page.waitForTimeout(1500);

  // Mascot on its own (transparent) for the cover
  await page.evaluate(() => {
    document.getElementById('mascotHello').style.display = 'none';
    // transparent page so only the host is captured
    ['launcher'].forEach(id => { document.getElementById(id).style.background = 'transparent'; });
    document.querySelector('#launcher .rays').style.display = 'none';
    document.documentElement.style.background = document.body.style.background = 'transparent';
  });
  await page.waitForTimeout(2500);
  const mascot = await page.locator('#mascot').screenshot({ omitBackground: true });
  if (!ED) fs.writeFileSync(path.join(ROOT, 'docs', 'mascot.png'), mascot);   // used by the teacher guide
  await page.evaluate(() => {
    document.getElementById('mascotHello').style.display = '';
    document.getElementById('launcher').style.background = '';
    document.querySelector('#launcher .rays').style.display = '';
    document.documentElement.style.background = document.body.style.background = '';
  });

  const SHOTS = ['question-bank', 'over-the-edge', 'outpace', 'category-clash', 'hex-hunt'];
  let oteShot, opShot, ccShot, hhShot;
  if (REUSE) {
    const from = path.join(ROOT, 'dist', 'listings', REUSE);
    SHOTS.forEach(n => fs.copyFileSync(path.join(from, n + '.png'), path.join(OUT, n + '.png')));
    [oteShot, opShot, ccShot, hhShot] = SHOTS.slice(1).map(n => fs.readFileSync(path.join(OUT, n + '.png')));
  } else {
  // Question bank, with one of the teacher's own sets added
  await page.evaluate(([sj, own]) => {
    CGB.bank.addSet(own[0], own[1], sj);
    CGB.bank.setSubject(sj); CGB.bank.setBoard('aqa'); CGB.bank.setSelection({ mixed: CGB.bank.courses()[0].specCode });
  }, [SHOW, OWN[SHOW]]);
  await page.click('#openBank');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'question-bank.png') });
  await page.keyboard.press('Escape');

  // Every game with six teams (two halves in Hex Hunt): what schools buy it for
  await page.evaluate(() => CGB.store.setJSON('teams', ['Owls', 'Foxes', 'Hawks', 'Otters', 'Badgers', 'Wolves']));
  const markRound = async (id, keys) => {
    // Space skips the countdown and the "3, 2, 1"; pressed again (spaced like a teacher) if a slow machine missed it
    for (let k = 0; k < 6 && await page.evaluate(id => CGB.games[id].state().round !== 'mark', id); k++) { await page.keyboard.press('Space'); await page.waitForTimeout(700); }
    await until(page, id => CGB.games[id].state().round === 'mark', 15000, id);
    for (const k of keys) await page.keyboard.press(k);
    await page.keyboard.press('Enter');
  };

  // Over the Edge: five teams right, their captains' counters dropping in team colours
  await page.click('[data-play="over-the-edge"]');
  await page.waitForTimeout(1500);
  await page.click('#ote-segTeams button[data-v="6"]');
  await page.click('#ote-startBtn');
  await page.waitForTimeout(3500);
  await markRound('over-the-edge', ['c', '4']);
  await page.waitForTimeout(600);
  for (const k of ['2', '3', '1', '4', '2']) { await page.keyboard.press(k); await page.waitForTimeout(150); }
  await page.waitForTimeout(1600);
  await page.screenshot({ path: path.join(OUT, 'over-the-edge.png') });
  oteShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.evaluate(() => CGB.app.show('launcher'));

  // Outpace: the whole class as one runner, four of six groups right
  await page.click('[data-play="outpace"]');
  await page.waitForTimeout(1000);
  await page.click('#op-segGroups button[data-v="6"]');
  await page.click('#op-startBtn');
  await page.keyboard.press('2');
  await page.waitForTimeout(1200);
  await markRound('outpace', ['1', '2', '4', '5']);
  await page.waitForTimeout(1800);
  await page.screenshot({ path: path.join(OUT, 'outpace.png') });
  opShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.evaluate(() => CGB.app.show('launcher'));

  // Category Clash: six teams part-way through a board, a question just marked
  await page.click('[data-play="category-clash"]');
  await page.waitForTimeout(600);
  await page.click('#cc-segTeams button[data-v="6"]');
  await page.click('#cc-startBtn');
  const ccMarks = [[0, 0, ['1', '2', '3']], [1, 0, ['c']], [2, 1, ['2', '5']], [3, 0, ['c', '6']], [0, 1, ['1', '3', '4', '5']], [3, 2, ['w', '2']]];
  for (const [c, r, keys] of ccMarks) {
    await page.locator(`#cc-board .cc-tile[data-c="${c}"][data-r="${r}"]`).click();
    await markRound('category-clash', keys);
    // back to the board: Enter again if a reveal (the star tile) was still showing
    for (let k = 0; k < 10 && await page.evaluate(() => CGB.games['category-clash'].state().phase !== 'board'); k++) { await page.keyboard.press('Enter'); await page.waitForTimeout(1500); }
    await until(page, () => CGB.games['category-clash'].state().phase === 'board', 5000);
  }
  await page.locator('#cc-board .cc-tile[data-c="1"][data-r="2"]').click();
  await markRound('category-clash', ['1', '2', '4', '6']);
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, 'category-clash.png') });
  ccShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.evaluate(() => CGB.app.show('launcher'));

  // Hex Hunt: two halves of the class part-way across the board
  await page.evaluate(() => CGB.store.setJSON('teams', ['Left half', 'Right half']));
  await page.click('[data-play="hex-hunt"]');
  await page.waitForTimeout(600);
  await page.click('#hh-startBtn');
  const hexMoves = [[[0, 1], '1'], [[1, 0], '2'], [[1, 2], '1'], [[2, 1], '2']];
  for (const [[c, r], side] of hexMoves) {
    await page.locator(`#hh-board .hh-hex[data-c="${c}"][data-r="${r}"]`).click();
    await page.keyboard.press('Space');
    await until(page, () => CGB.games['hex-hunt'].state().round === 'mark', 15000);
    await page.keyboard.press(side); await page.keyboard.press('Enter');
  }
  await page.locator('#hh-board .hh-hex[data-c="3"][data-r="0"]').click();
  await page.keyboard.press('Space');
  await until(page, () => CGB.games['hex-hunt'].state().round === 'mark', 15000);
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'hex-hunt.png') });
  hhShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.evaluate(() => CGB.app.show('launcher'));

  }

  // Cover (1600×1200, the 4:3 shape Tes crops to): built in the same page so it uses the bundle's own fonts and artwork.
  // The backdrop is a frame of the demo video (docs/cover-backdrop.jpg): the gold jackpot counter at the edge.
  await page.setViewportSize({ width: 1600, height: 1200 });
  const b64 = buf => 'data:image/' + (buf[0] === 0x89 ? 'png' : 'jpeg') + ';base64,' + buf.toString('base64');
  // (the counters carry atoms, so Maths, History and Geography get the plain Showtime stage instead)
  const plain = !!ED && ['maths', 'history', 'geography'].includes(ED.id);
  const backdrop = plain ? '' : 'data:image/jpeg;base64,' + fs.readFileSync(path.join(ROOT, 'docs', 'cover-backdrop.jpg')).toString('base64');
  const NOTE = { 'free-taster': 'Free: one topic per subject', biology: 'Separate & Combined Science', chemistry: 'Separate & Combined Science', physics: 'Separate & Combined Science',
    science: 'Biology · Chemistry · Physics', maths: 'Foundation & Higher', history: 'Every topic and option', geography: 'Every topic and option', 'mega-bundle': 'All six subjects' };
  const pills = ['Every team answers', 'AQA & Edexcel', NOTE[ED ? ED.id : 'mega-bundle'], 'Works offline'];
  await page.evaluate(({ ote, op, cc, hh, backdrop, edName, pills }) => {
    document.body.innerHTML = `
      <div id="cover">
        <div class="bg"></div><div class="shade"></div>
        <div class="wm">${CGB.brand.wordmark()}</div>
        <div class="hl">Whole-class <b>GCSE revision</b></div>
        ${edName ? `<div class="bdg">${edName}</div>` : ''}
        <div id="pip"></div>
        <div class="row">
          ${[[ote, 'oteLogo'], [op, 'opLogo'], [cc, 'ccLogo'], [hh, 'hhLogo']].map(([im, f]) => `<div class="g"><img class="im" src="${im}" alt=""><div class="lg">${CGB.brand[f]()}</div></div>`).join('')}
        </div>
        <div class="pills">${pills.map(t => `<span class="pill">${t.replace(/&/g, '&amp;')}</span>`).join('')}</div>
      </div>`;
    const st = document.createElement('style');
    st.textContent = `
      html, body { overflow: hidden; background: #141834; }
      #cover { position: fixed; inset: 0; overflow: hidden; background: #141834; font-family: var(--body); }
      #cover .bg { position: absolute; inset: 0; background: ${backdrop ? `url('${backdrop}') 30% 60% / cover` : 'radial-gradient(ellipse at 50% -10%, #3A3F86 0%, #1F2550 40%, #141834 80%)'}; }
      #cover .shade { position: absolute; inset: 0; ${backdrop ? '' : 'display: none;'} background: linear-gradient(180deg, rgba(14,16,44,0.86) 0%, rgba(14,16,44,0.55) 38%, rgba(14,16,44,0.86) 100%); }
      #cover .wm { position: absolute; left: 60px; top: 28px; width: 780px; }
      #cover .wm svg, #cover .lg svg { display: block; width: 100%; height: auto; filter: drop-shadow(0 10px 0 rgba(0,0,0,0.4)); }
      #cover .hl { position: absolute; left: 66px; top: 400px; font-family: var(--display); font-size: 88px; line-height: 1; color: #fff; text-shadow: 0 6px 0 rgba(0,0,0,0.45); }
      #cover .hl b { font-weight: 400; color: #FFC93C; }
      #cover .bdg { position: absolute; left: 66px; top: 520px; font-family: var(--display); font-size: 48px; color: #1B1F3B; background: #FFC93C; border: 5px solid #1B1F3B; border-radius: 16px; padding: 4px 26px 8px; box-shadow: 0 8px 0 rgba(0,0,0,0.35); transform: rotate(-2deg); }
      #cover #pip { position: absolute; right: 60px; top: 30px; width: 420px; height: 620px; }
      #cover .row { position: absolute; left: 40px; right: 40px; top: 660px; display: flex; gap: 20px; }
      #cover .g { flex: 1; text-align: center; }
      #cover .g .im { width: 100%; height: 270px; object-fit: cover; object-position: top left; display: block; border-radius: 18px; border: 6px solid #FFC93C; box-shadow: 0 10px 0 rgba(0,0,0,0.4); box-sizing: border-box; }
      #cover .g .lg { width: 210px; margin: 18px auto 0; }
      #cover .pills { position: absolute; left: 0; right: 0; bottom: 44px; display: flex; justify-content: center; gap: 14px; }
      #cover .pill { font-weight: 900; font-size: 25px; color: #1B1F3B; background: #FFF9F0; border: 5px solid #1B1F3B; border-radius: 999px; padding: 6px 24px 9px; box-shadow: 0 6px 0 rgba(0,0,0,0.35); white-space: nowrap; }`;
    document.head.appendChild(st);
    CGB.createHost2D(document.getElementById('pip'), { className: 'host-launcher' }).gesture('wave', 600000);
  }, { ote: b64(oteShot), op: b64(opShot), cc: b64(ccShot), hh: b64(hhShot), backdrop, edName: ED ? ED.name : '', pills });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'cover.png') });
  await browser.close();
  console.log(`Listing images written to ${path.relative(ROOT, OUT)}/`);
})().catch(e => { console.error(e); process.exit(1); });
