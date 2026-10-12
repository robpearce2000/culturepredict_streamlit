/*
 * Demo shot "outpace": Outpace with its own panels, AQA Physics, six teams (seed 3, the reel 4 deal).
 *   Part 1 (deal round, Standard deal): a right answer (the class surges a step), a wrong one (the Hunter lunges),
 *   then five right answers to HOME and the burst through the arch ("Escaped!"). Only the moves are recorded,
 *   each from the moment the teacher confirms the marking; cues name each move for the edit.
 *   Part 2 (a fresh game, straight to the Final Sprint): the clock running and a step of the sprint.
 *   node tools/video/demo/shot-outpace.js
 */
'use strict';
const L = require('./lib.js');
const SEED = 3;
const PLAN = ['right', 'wrong', 'right', 'right', 'right', 'right', 'right'];
const KEEP = [1.4, 1.6, 0.9, 0.9, 0.9, 0.9, 1.0];            // seconds recorded of each move

async function boot() {
  const { browser, page } = await L.open({ seed: SEED });
  const T = L.timer(page);
  await page.evaluate(() => { window.__reelSkip = true; CGB.bank.setBoard('aqa'); CGB.bank.setSubject('physics'); CGB.app.show('outpace'); });
  await T.wait(300);
  await page.evaluate(() => { document.querySelector('#op-segGroups button[data-v="6"]').click(); document.getElementById('op-startBtn').click(); });
  await T.until(() => CGB.games.outpace.state().phase === 'deal', 30000);
  await T.wait(800);
  return { browser, page, T, st: () => page.evaluate(() => { const s = CGB.games.outpace.state(); return { phase: s.phase, round: s.round }; }) };
}

(async () => {
  {
    const { browser, page, T, st } = await boot();
    await page.keyboard.press('2');                            // the Standard deal: the Hunter starts 3 steps behind
    const s = new L.Shot(page, 'outpace');
    for (let q = 0; q < PLAN.length; q++) {
      await T.until(() => /think|show|mark/.test(CGB.games.outpace.state().round));
      for (let k = 0; k < 6 && (await st()).round !== 'mark'; k++) { await page.keyboard.press('Space'); await T.wait(560); }
      if (PLAN[q] === 'right') await page.keyboard.press('c');
      if (!q) { await page.evaluate(() => { window.__reelSkip = false; }); await T.wait(100); await s.start(); }
      s.cue('move' + q);
      await page.keyboard.press('Enter');                      // confirm: the step happens now
      await s.run(KEEP[q]);
      await T.until(() => CGB.games.outpace.state().round === 'done');
      await page.keyboard.press('Enter');                      // next question (after the last: HOME)
      await T.wait(50);
    }
    s.cue('escape');
    await s.run(2.8);
    s.cue('end');
    await s.done({ seed: SEED, plan: PLAN });
    await browser.close();
  }
  {
    const { browser, page, T, st } = await boot();
    await page.evaluate(() => CGB.test.outpace.toSprint(600));
    await T.wait(600);
    await page.keyboard.press('Enter');                        // Start the Final Sprint
    await T.wait(400);
    await page.evaluate(() => { window.__reelSkip = false; }); await T.wait(100);
    const s = new L.Shot(page, 'outpace-sprint'); await s.start();
    s.cue('clock'); await s.run(2.2);
    for (let k = 0; k < 6 && (await st()).round !== 'mark'; k++) { await page.keyboard.press('Space'); await s.run(0.56); }
    await page.keyboard.press('c'); await s.run(0.3);
    s.cue('step'); await page.keyboard.press('Space');
    await s.run(1.6);
    s.cue('end');
    await s.done({ seed: SEED });
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
