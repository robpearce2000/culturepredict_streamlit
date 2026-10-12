/*
 * Demo shots for the two board games, with their own panels on screen (recording only).
 *   node tools/video/demo/shot-boards.js cc [seed]   Category Clash, AQA Chemistry, 4 teams: a first pass (nothing drawn)
 *                                                   finds the hidden ★ tile; the take then picks it: it spins round
 *                                                   with gold light ("Double points!"), the question, "3, 2, 1, show
 *                                                   me!", three of four teams right, and the scores jump.
 *   node tools/video/demo/shot-boards.js hh [seed]   Hex Hunt, AQA Biology: Team 1 claims the middle row; recorded are
 *                                                   one hexagon turning its colour and the last claim, when the
 *                                                   winning path lights up from edge to edge.
 */
'use strict';
const L = require('./lib.js');
const which = process.argv[2], SEED = +(process.argv[3] || 5);

async function boot(game, subject) {
  const { browser, page } = await L.open({ seed: SEED });
  const T = L.timer(page);
  await page.evaluate(([g, s]) => { window.__reelSkip = true; CGB.bank.setBoard('aqa'); CGB.bank.setSubject(s); CGB.app.show(g); }, [game, subject]);
  await T.wait(300);
  return { browser, page, T, st: () => page.evaluate(g => CGB.games[g].state(), game) };
}
async function toMark(page, T, st, s) {
  for (let k = 0; k < 6; k++) { if ((await st()).round === 'mark') return; await page.keyboard.press('Space'); if (s) await s.run(0.56); else await T.wait(560); }
}

async function cc() {
  const start = async (page, T) => {
    await page.evaluate(() => { document.querySelector('#cc-segTeams button[data-v="4"]').click(); document.getElementById('cc-startBtn').click(); });
    await T.until(() => CGB.games['category-clash'].state().phase === 'board');
  };
  // pass 1: open tiles in turn until the star turns up
  let star = null;
  {
    const { browser, page, T, st } = await boot('category-clash', 'chemistry');
    await start(page, T);
    for (let r = 0; r < 3 && !star; r++) for (let c = 0; c < 4 && !star; c++) {
      await T.until(() => CGB.games['category-clash'].state().phase === 'board');
      await T.wait(1500);
      const ok = await page.evaluate(([c, r]) => { const b = document.querySelector(`#cc-board button[data-c="${c}"][data-r="${r}"]`); if (!b || b.disabled) return false; b.click(); return true; }, [c, r]);
      if (!ok) continue;
      await T.wait(200);
      if (await page.evaluate(() => CGB.games['category-clash'].state().step === 'reveal' || !document.getElementById('cc-qStar').hidden)) { star = { c, r }; break; }
      await T.wait(1500);
      await toMark(page, T, st); await page.keyboard.press('1'); await page.keyboard.press('Enter');
      await T.until(() => CGB.games['category-clash'].state().round === 'done');
      await page.keyboard.press('Enter');
    }
    await browser.close();
  }
  if (!star) throw new Error('no star tile found');
  console.log('star tile', star);
  // pass 2: the same seed (the star is placed when the board is made), a fresh board, the first pick is the star
  const { browser, page, T, st } = await boot('category-clash', 'chemistry');
  await start(page, T);
  await T.until(() => CGB.games['category-clash'].state().phase === 'board');
  await T.wait(1500);
  const cur = (await st()).cursor;                       // walk the keyboard cursor to the star tile, on camera
  await page.evaluate(() => { window.__reelSkip = false; }); await T.wait(100);
  const s = new L.Shot(page, 'cc'); await s.start();
  s.cue('board'); await s.run(0.6);
  const steps = [];
  for (let c = cur.c; c < star.c; c++) steps.push('ArrowRight'); for (let c = cur.c; c > star.c; c--) steps.push('ArrowLeft');
  for (let r = cur.r; r < star.r; r++) steps.push('ArrowDown'); for (let r = cur.r; r > star.r; r--) steps.push('ArrowUp');
  for (const k of steps) { await page.keyboard.press(k); await s.run(0.25); }
  await s.run(0.3);
  s.cue('pick'); await page.keyboard.press('Enter');
  await s.run(3.2);                                      // the star tile spins: double points
  s.cue('question');
  await s.run(1.2);
  await toMark(page, T, st, s);
  s.cue('mark');
  for (const k of ['1', '2', '4']) { await page.keyboard.press(k); await s.run(0.25); }
  await page.keyboard.press('3'); await s.run(1 / 60); await page.keyboard.press('3'); await s.run(0.4);
  s.cue('scores'); await page.keyboard.press('Enter');
  await s.run(2.6);
  s.cue('end');
  await s.done({ seed: SEED, star });
  await browser.close();
}

async function hh() {
  const { browser, page, T, st } = await boot('hex-hunt', 'biology');
  await page.evaluate(() => document.getElementById('hh-startBtn').click());
  await T.until(() => CGB.games['hex-hunt'].state().phase === 'board');
  const n = (await st()).size, row = Math.floor(n / 2);
  const s = new L.Shot(page, 'hh');
  for (let c = 0; c < n; c++) {
    await T.until(() => CGB.games['hex-hunt'].state().phase === 'board');
    await T.wait(700);
    const last = c === n - 1, shown = c === n - 2;
    if (shown) { await page.evaluate(() => { window.__reelSkip = false; }); await T.wait(100); await s.start(); s.cue('pick'); await s.run(0.5); }
    if (last) { s.cue('lastpick'); await s.run(0.4); }
    await page.evaluate(([c, r]) => document.querySelector(`#hh-board [data-c="${c}"][data-r="${r}"]`).dispatchEvent(new MouseEvent('click', { bubbles: true })), [c, row]);
    if (shown || last) await s.run(1.2); else await T.wait(300);
    await toMark(page, T, st, shown || last ? s : null);
    await page.keyboard.press('1');                      // one press marks it for Team 1; Enter claims it
    if (shown || last) await s.run(0.5);
    await T.until(() => { const x = CGB.games['hex-hunt'].state(); return x.round === 'done' || x.step === 'winning'; });
    if (!shown && !last) { await T.wait(400); await page.keyboard.press('Enter'); continue; }
    s.cue(last ? 'win' : 'claim');
    await page.keyboard.press('Enter');
    await s.run(last ? 4.0 : 1.8);
  }
  s.cue('end');
  await s.done({ seed: SEED, row });
  await browser.close();
}

(which === 'cc' ? cc() : hh()).catch(e => { console.error(e); process.exit(1); });
