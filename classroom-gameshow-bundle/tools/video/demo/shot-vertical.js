/*
 * Demo shot "v-ote" for the vertical reel: Over the Edge in portrait (1080×1920, the game's own tall layout), the
 * same question as the 16:9 shot (seed 4, "Why do veins have valves?"); the six teams marked ✓ ✓ ✗ ✓ ✓ ✗ and
 * confirmed. The tall layout keeps the panels between y 960 and 1530, inside the platforms' safe zone.
 *   node tools/video/demo/shot-vertical.js
 */
'use strict';
const L = require('./lib.js');
const SEED = 4;
(async () => {
  const { browser, page } = await L.open({ seed: SEED, width: 1080, height: 1920 });
  const T = L.timer(page);
  await page.evaluate(() => { window.__reelSkip = true; CGB.bank.setBoard('aqa'); CGB.bank.setSubject('biology'); CGB.app.show('over-the-edge'); });
  await T.wait(300);
  await page.evaluate(() => { document.querySelector('#ote-segTeams button[data-v="6"]').click(); document.getElementById('ote-startBtn').click(); });
  await T.until(() => CGB.games['over-the-edge'].state().step === 'ask', 60000);
  await T.wait(2600);
  await page.keyboard.press('Space');                         // "3, 2, 1, show me!"
  await T.until(() => CGB.games['over-the-edge'].state().round === 'mark', 10000);
  await page.evaluate(() => { window.__reelSkip = false; }); await T.wait(100);
  const s = new L.Shot(page, 'v-ote'); await s.start();
  s.cue('mark'); await s.run(0.4);
  for (const k of ['1', '2', '3', '4', '5', '6']) {
    await page.keyboard.press(k);
    if (k === '3' || k === '6') { await s.run(1 / 60); await page.keyboard.press(k); }
    await s.run(0.3);
  }
  s.cue('confirm'); await page.keyboard.press('Enter');
  await s.run(1.2);
  s.cue('end');
  await s.done({ seed: SEED });
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
