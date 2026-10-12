/*
 * Demo shot "results": Category Clash's class results screen with "Reteach these", the questions most teams got
 * wrong. Five questions are played quickly with nothing drawn (AQA Physics, 4 teams, a mix of right and wrong),
 * then End game; the results screen is recorded as it appears.
 *   node tools/video/demo/shot-results.js [seed] [--still file]
 */
'use strict';
const L = require('./lib.js');
const SEED = +(process.argv[2] || 2);
const MARKS = [['1', '2', '3', '4'], ['1'], ['2', '4'], [], ['1', '2', '3']];   // the teams right on each question

(async () => {
  const { browser, page } = await L.open({ seed: SEED });
  const T = L.timer(page);
  const st = () => page.evaluate(() => CGB.games['category-clash'].state());
  await page.evaluate(() => { window.__reelSkip = true; CGB.bank.setBoard('aqa'); CGB.bank.setSubject('physics'); CGB.app.show('category-clash'); });
  await T.wait(300);
  await page.evaluate(() => { document.querySelector('#cc-segTeams button[data-v="4"]').click(); document.getElementById('cc-startBtn').click(); });
  await T.until(() => CGB.games['category-clash'].state().phase === 'board');
  for (let q = 0; q < MARKS.length; q++) {
    await T.until(() => CGB.games['category-clash'].state().phase === 'board');
    await T.wait(1500);
    await page.evaluate(() => { const b = [...document.querySelectorAll('#cc-board button[data-c]')].find(b => !b.disabled); b.click(); });
    await T.wait(3500);
    for (let k = 0; k < 6 && (await st()).round !== 'mark'; k++) { await page.keyboard.press('Space'); await T.wait(560); }
    for (const k of MARKS[q]) await page.keyboard.press(k);
    await page.keyboard.press('Enter');
    await T.until(() => CGB.games['category-clash'].state().round === 'done');
    await T.wait(800);
    await page.keyboard.press('Enter');
  }
  await T.wait(1500);
  await page.evaluate(() => CGB.app.endGame());
  const i = process.argv.indexOf('--still');
  await page.evaluate(() => { window.__reelSkip = false; });
  if (i > 0) { await T.wait(3000); await L.still(page, process.argv[i + 1]); await browser.close(); return; }
  const s = new L.Shot(page, 'results'); await s.start();
  s.cue('results'); await s.run(5.5);
  s.cue('end');
  await s.done({ seed: SEED });
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
