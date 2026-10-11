/*
 * The two board-game shots for reel 2 (recording only), in a tall stacked layout: caption space on top, the board
 * large in the middle, the team panels, host and key hints hidden.
 *   node tools/video/reels/render-boards.js cc-star [seed]   Category Clash: the hidden star tile is picked and spins
 *                                                            round to show it is worth double (found by a first pass)
 *   node tools/video/reels/render-boards.js hh-path [seed]   Hex Hunt: team 1 claims a row of hexagons; the last claim
 *                                                            completes the winning path from edge to edge
 */
'use strict';
const { open, Reel, W, H, FPS } = require('./reel.js');
const which = process.argv[2], SEED = +(process.argv[3] || 5);

const LAYOUT = `
#game-category-clash .cc-teams, #game-category-clash .host-foot, #game-category-clash .keys, #game-category-clash .cc-head,
#game-hex-hunt .hh-teams, #game-hex-hunt .host-foot, #game-hex-hunt .keys, #game-hex-hunt .hh-head, #game-hex-hunt #hh-curHint { display: none !important; }
#game-category-clash #cc-play { padding: 400px 40px 470px !important; box-sizing: border-box; height: 100vh; display: flex; flex-direction: column; }
#game-category-clash #cc-board { flex: 1 1 auto; height: auto !important; min-height: 1000px; }
#game-hex-hunt #hh-play { padding-top: 380px !important; padding-bottom: 470px !important; }
#game-hex-hunt #hh-board { scale: 1.16; }
`;

async function boot(game) {
  const { browser, page } = await open({ seed: SEED, quality: 'high', css: LAYOUT });
  await page.setViewportSize({ width: W, height: H });
  const wait = ms => page.clock.runFor(ms);
  const until = async (fn, ms) => { for (let t = 0; t < (ms || 20000); t += 50) { if (await page.evaluate(fn)) return true; await wait(50); } return false; };
  await page.evaluate(g => { CGB.bank.setBoard('aqa'); CGB.bank.setSubject(g === 'category-clash' ? 'chemistry' : 'biology'); CGB.app.show(g); }, game);
  await wait(300);
  return { browser, page, wait, until, st: () => page.evaluate(g => CGB.games[g].state(), game) };
}
async function markRound(page, wait, st, keys) {
  for (let k = 0; k < 6; k++) { const s = await st(); if (s.round === 'mark') break; await page.keyboard.press('Space'); await wait(560); }
  for (const k of keys) await page.keyboard.press(k);
  await page.keyboard.press('Enter');
}

async function ccStar() {
  // pass 1: open tiles in turn until the star turns up
  let star = null;
  {
    const { browser, page, wait, until, st } = await boot('category-clash');
    await page.evaluate(() => { document.querySelector('#cc-segTeams button[data-v="4"]').click(); document.getElementById('cc-startBtn').click(); });
    await until(() => CGB.games['category-clash'].state().phase === 'board');
    for (let r = 0; r < 3 && !star; r++) for (let c = 0; c < 4 && !star; c++) {
      await until(() => CGB.games['category-clash'].state().phase === 'board');
      await wait(1500);
      const ok = await page.evaluate(([c, r]) => { const b = document.querySelector(`#cc-board button[data-c="${c}"][data-r="${r}"]`); if (!b || b.disabled) return false; b.click(); return true; }, [c, r]);
      if (process.env.DEBUG) console.log('tile', c, r, ok, await page.evaluate(() => { const s = CGB.games['category-clash'].state(); return s.phase + '/' + s.step + '/' + s.round; }));
      if (!ok) continue;
      await wait(200);
      if (await page.evaluate(() => CGB.games['category-clash'].state().step === 'reveal' || !document.getElementById('cc-qStar').hidden)) { star = { c, r }; break; }
      await wait(1500);
      await markRound(page, wait, st, ['1']);
      await until(() => CGB.games['category-clash'].state().round === 'done');
      await page.keyboard.press('Enter');
    }
    await browser.close();
  }
  if (!star) throw new Error('no star tile found');
  console.log('star tile', star);
  // pass 2: the same seed, straight to the star tile, recorded
  const { browser, page, wait, until } = await boot('category-clash');
  await page.evaluate(() => { document.querySelector('#cc-segTeams button[data-v="4"]').click(); document.getElementById('cc-startBtn').click(); });
  await until(() => CGB.games['category-clash'].state().phase === 'board');
  await page.evaluate(([c, r]) => { /* put the keyboard cursor on the star tile */ }, [star.c, star.r]);
  for (let i = 0; i < star.c; i++) await page.keyboard.press('ArrowRight');
  for (let i = 0; i < star.r; i++) await page.keyboard.press('ArrowDown');
  await wait(1200);
  const reel = new Reel(page, 'board-cc-star'); await reel.start();
  for (let i = 0; i < 0.5 * FPS; i++) { await wait(1000 / FPS); await reel.frame(true, true); }
  await page.keyboard.press('Enter');
  for (let i = 0; i < 2.6 * FPS; i++) { await wait(1000 / FPS); await reel.frame(true, true); }
  console.log('board-cc-star frames', await reel.finish({ seed: SEED, star }));
  await browser.close();
}

async function hhPath() {
  const { browser, page, wait, until, st } = await boot('hex-hunt');
  await page.evaluate(() => document.getElementById('hh-startBtn').click());
  await until(() => CGB.games['hex-hunt'].state().phase === 'board');
  const n = (await st()).size;
  const row = Math.floor(n / 2);
  const reel = new Reel(page, 'board-hh-path');
  for (let c = 0; c < n; c++) {
    await until(() => CGB.games['hex-hunt'].state().phase === 'board');
    await wait(700);
    const last = c === n - 1;
    if (last) { await reel.start(); for (let i = 0; i < 0.4 * FPS; i++) { await wait(1000 / FPS); await reel.frame(true, true); } }
    await page.evaluate(([c, r]) => document.querySelector(`#hh-board [data-c="${c}"][data-r="${r}"]`).dispatchEvent(new MouseEvent('click', { bubbles: true })), [c, row]);
    await wait(300);
    // one press marks it (the half with more right answers); then Enter claims it, or starts the celebration
    for (let k = 0; k < 6; k++) { const s = await st(); if (s.round === 'mark') break; await page.keyboard.press('Space'); await wait(560); }
    await page.keyboard.press('1');
    await until(() => { const s = CGB.games['hex-hunt'].state(); return s.round === 'done' || s.step === 'winning'; });
    await wait(400);
    if (!last) { await page.keyboard.press('Enter'); continue; }
    await page.keyboard.press('Enter');
    for (let i = 0; i < 3.3 * FPS; i++) { await wait(1000 / FPS); await reel.frame(true, true); }
  }
  console.log('board-hh-path frames', await reel.finish({ seed: SEED, row }));
  await browser.close();
}

(which === 'cc-star' ? ccStar() : hhPath()).catch(e => { console.error(e); process.exit(1); });
