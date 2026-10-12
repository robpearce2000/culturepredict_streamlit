/*
 * Demo shot "ote": an Over the Edge class question as a teacher runs it, with the game's own panels on screen.
 * The question (seed 4: "Why do veins have valves?", AQA Biology), "3, 2, 1, show me!", six teams marked
 * ✓ ✓ ✗ ✓ ✓ ✗ in seconds, the answer, then random lanes: the counters bounce down the pegs and push counters
 * off the edge while the scores update.
 *   node tools/video/demo/shot-ote.js [seed]
 */
'use strict';
const L = require('./lib.js');
const SEED = +(process.argv[2] || 4);

(async () => {
  const { browser, page } = await L.open({ seed: SEED });
  const T = L.timer(page);
  await page.evaluate(() => { window.__reelSkip = true; CGB.bank.setBoard('aqa'); CGB.bank.setSubject('biology'); CGB.app.show('over-the-edge'); });
  await T.wait(300);
  await page.evaluate(() => { document.querySelector('#ote-segTeams button[data-v="6"]').click(); document.getElementById('ote-startBtn').click(); });
  await T.until(() => CGB.games['over-the-edge'].state().step === 'ask', 60000);
  await T.wait(2600);                                        // the "Round 1" banner has gone
  await page.evaluate(() => { window.__reelSkip = false; });
  console.log('question:', await page.evaluate(() => CGB.games['over-the-edge'].state().q.q));
  const s = new L.Shot(page, 'ote'); await s.start();
  s.cue('ask'); await s.run(2.4);
  s.cue('showme'); await page.keyboard.press('Space');       // "3, 2, 1, show me!"
  for (let i = 0; i < 6 * 60 && (await page.evaluate(() => CGB.games['over-the-edge'].state().round)) !== 'mark'; i++) await s.run(1 / 60);
  s.cue('mark'); await s.run(0.4);
  // left to right, ✓ ✓ ✗ ✓ ✓ ✗ (a team's key toggles ✓/✗; a wrong team is two presses a frame apart)
  for (const k of ['1', '2', '3', '4', '5', '6']) {
    await page.keyboard.press(k);
    if (k === '3' || k === '6') { await s.run(1 / 60); await page.keyboard.press(k); }
    await s.run(0.3);
  }
  await s.run(0.3);
  s.cue('confirm'); await page.keyboard.press('Enter');
  await s.run(1.6);
  s.cue('lanes'); await page.keyboard.press('r');            // random lanes: the drops begin
  await s.run(5.5);
  s.cue('end');
  await s.done({ seed: SEED });
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
