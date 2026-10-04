#!/usr/bin/env node
/*
 * Makes the Tes listing images from the built bundle (run `node build.js` first):
 *   dist/listing/cover.png                1600×1000  wordmark, mascot and all four games
 *   dist/listing/over-the-edge.png        1920×1080  whole-class game, six teams
 *   dist/listing/outpace.png              1920×1080  whole-class game, six teams
 *   dist/listing/category-clash.png       1920×1080  whole-class game, six teams
 *   dist/listing/hex-hunt.png             1920×1080  whole-class game, two halves
 *   dist/listing/question-bank.png        1920×1080  the shared question bank
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');

const ROOT = path.join(__dirname, '..');
const URL = 'file://' + path.join(ROOT, 'dist', 'showtime-classroom-gameshows.html');
const OUT = path.join(ROOT, 'dist', 'listing');
const ARGS = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

async function until(page, fn, ms, arg) {
  const end = Date.now() + (ms || 60000);
  while (Date.now() < end) { if (await page.evaluate(fn, arg)) return; await page.waitForTimeout(150); }
  throw new Error('timed out waiting');
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ARGS });
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
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
  fs.writeFileSync(path.join(ROOT, 'docs', 'mascot.png'), mascot);   // used by the teacher guide
  await page.evaluate(() => {
    document.getElementById('mascotHello').style.display = '';
    document.getElementById('launcher').style.background = '';
    document.querySelector('#launcher .rays').style.display = '';
    document.documentElement.style.background = document.body.style.background = '';
  });

  // Question bank, with one of the teacher's own sets added
  await page.evaluate(() => {
    CGB.bank.addSet('Year 10 bonding quiz', 'Subject: Chemistry\nTopic: Bonding\nQ: What type of bonding is in sodium chloride?\nA: Ionic\nQ: What type of bonding is in methane?\nA: Covalent\nQ: Why can metals conduct electricity?\nA: They have delocalised electrons');
    CGB.bank.setActive('gcse-combined-mix');
  });
  await page.click('#openBank');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'question-bank.png') });
  await page.keyboard.press('Escape');

  // Every game with six teams (two halves in Hex Hunt): what schools buy it for
  await page.evaluate(() => CGB.store.setJSON('teams', ['Owls', 'Foxes', 'Hawks', 'Otters', 'Badgers', 'Wolves']));
  const markRound = async (id, keys) => {
    await page.keyboard.press('Space');
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
  const oteShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

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
  const opShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Category Clash: six teams part-way through a board, a question just marked
  await page.click('[data-play="category-clash"]');
  await page.waitForTimeout(600);
  await page.click('#cc-segTeams button[data-v="6"]');
  await page.click('#cc-startBtn');
  const ccMarks = [[0, 0, ['1', '2', '3']], [1, 0, ['c']], [2, 1, ['2', '5']], [3, 0, ['c', '6']], [0, 1, ['1', '3', '4', '5']], [4, 2, ['w', '2']]];
  for (const [c, r, keys] of ccMarks) {
    await page.locator(`#cc-board .cc-tile[data-c="${c}"][data-r="${r}"]`).click();
    await markRound('category-clash', keys);
    await page.keyboard.press('Enter');
    await until(page, () => CGB.games['category-clash'].state().phase === 'board', 5000);
  }
  await page.locator('#cc-board .cc-tile[data-c="1"][data-r="2"]').click();
  await markRound('category-clash', ['1', '2', '4', '6']);
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, 'category-clash.png') });
  const ccShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Hex Hunt: two halves of the class part-way across the board
  await page.evaluate(() => CGB.store.setJSON('teams', ['Left half', 'Right half']));
  await page.click('[data-play="hex-hunt"]');
  await page.waitForTimeout(600);
  await page.click('#hh-startBtn');
  const hexMoves = [[[0, 2], [7, 3]], [[2, 0], [2, 8]], [[1, 2], [9, 4]], [[2, 1], [3, 7]], [[3, 2], [8, 6]], [[1, 4], [4, 9]]];
  for (const [[c, r], [a, b]] of hexMoves) {
    await page.locator(`#hh-board .hh-hex[data-c="${c}"][data-r="${r}"]`).click();
    await page.keyboard.press('Space');
    await until(page, () => CGB.games['hex-hunt'].state().round === 'mark', 15000);
    await page.keyboard.press('1'); for (let i = 0; i < a; i++) await page.keyboard.press('ArrowRight');
    await page.keyboard.press('2'); for (let i = 0; i < b; i++) await page.keyboard.press('ArrowRight');
    await page.keyboard.press('Enter'); await page.keyboard.press('Enter');
  }
  await page.locator('#hh-board .hh-hex[data-c="4"][data-r="1"]').click();
  await page.keyboard.press('Space');
  await until(page, () => CGB.games['hex-hunt'].state().round === 'mark', 15000);
  await page.keyboard.press('1'); for (let i = 0; i < 8; i++) await page.keyboard.press('ArrowRight');
  await page.keyboard.press('2'); for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, 'hex-hunt.png') });
  const hhShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Cover: built in the same page so it uses the bundle's own fonts and artwork
  await page.setViewportSize({ width: 1600, height: 1000 });
  const b64 = buf => 'data:image/' + (buf[0] === 0x89 ? 'png' : 'jpeg') + ';base64,' + buf.toString('base64');
  await page.evaluate(({ ote, op, cc, hh, mascot }) => {
    document.body.innerHTML = `
      <div id="cover">
        <svg class="rays" viewBox="0 0 100 100" preserveAspectRatio="none"><g fill="rgba(255,255,255,0.05)"><path d="M50 -10 L30 110 L40 110 Z"/><path d="M50 -10 L55 110 L68 110 Z"/><path d="M50 -10 L80 110 L95 110 Z"/><path d="M50 -10 L5 110 L15 110 Z"/></g></svg>
        <div class="wm">${CGB.brand.wordmark()}</div>
        <img class="mascot" src="${mascot}" alt="">
        <div class="shots">
          <figure class="s1"><img src="${ote}" alt=""><figcaption>${CGB.brand.oteLogo()}</figcaption></figure>
          <figure class="s2"><img src="${op}" alt=""><figcaption>${CGB.brand.opLogo()}</figcaption></figure>
          <figure class="s3"><img src="${cc}" alt=""><figcaption>${CGB.brand.ccLogo()}</figcaption></figure>
          <figure class="s4"><img src="${hh}" alt=""><figcaption>${CGB.brand.hhLogo()}</figcaption></figure>
        </div>
        <div class="chips"><span>Whole class: every team answers</span><span>4 games</span><span>190+ AQA-style GCSE science questions</span><span>Works offline</span></div>
      </div>`;
    const st = document.createElement('style');
    st.textContent = `
      html, body { overflow: hidden; background: #141834; }
      #cover { position: fixed; inset: 0; background: radial-gradient(ellipse at 50% -10%, #3A3F86 0%, #1F2550 40%, #141834 80%); font-family: var(--body); }
      #cover .rays { position: absolute; inset: 0; width: 100%; height: 100%; }
      #cover .wm { position: absolute; left: 50%; top: 18px; transform: translateX(-50%); width: 820px; }
      #cover .wm svg { width: 100%; height: auto; display: block; }
      #cover .mascot { position: absolute; right: 30px; top: 40px; height: 330px; }
      #cover .shots { position: absolute; left: 0; right: 0; top: 452px; display: grid; grid-template-columns: repeat(4, 368px); justify-content: center; gap: 20px; }
      #cover figure { margin: 0; width: 368px; position: relative; }
      #cover figure img { width: 100%; display: block; border: 6px solid #1B1F3B; border-radius: 20px; box-shadow: 0 12px 0 rgba(0,0,0,0.35); }
      #cover .s1, #cover .s3 { transform: rotate(-2.5deg); } #cover .s2, #cover .s4 { transform: rotate(2.5deg); }
      #cover figcaption { position: absolute; left: 50%; bottom: -96px; transform: translateX(-50%); width: 250px; }
      #cover figcaption svg { width: 100%; height: auto; display: block; filter: drop-shadow(0 6px 0 rgba(0,0,0,0.35)); }
      #cover .chips { position: absolute; left: 0; right: 0; bottom: 26px; display: flex; justify-content: center; gap: 14px; }
      #cover .chips span { background: #FFF9F0; color: #1B1F3B; border: 4px solid #1B1F3B; border-radius: 999px; padding: 8px 18px; font-weight: 900; font-size: 22px; white-space: nowrap; }`;
    document.head.appendChild(st);
  }, { ote: b64(oteShot), op: b64(opShot), cc: b64(ccShot), hh: b64(hhShot), mascot: b64(mascot) });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'cover.png') });
  await browser.close();
  console.log('Listing images written to dist/listing/');
})().catch(e => { console.error(e); process.exit(1); });
