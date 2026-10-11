/*
 * Reel 4, "The Hunter is coming": Outpace's Deal Round through the reel camera, low behind the Hunter looking down
 * the track to HOME. Standard deal (Hunter 3 steps behind), two wrong answers (the Hunter closes to one step: the
 * near-catch), then six right answers carry the class home and it bursts through the arch with confetti.
 * Only the moves are recorded (each answer's step, the escape), joined as quick cuts.
 *   node tools/video/reels/render-outpace.js [seed]
 */
'use strict';
const { open, Reel, W, H, FPS } = require('./reel.js');

const SEED = +(process.argv[2] || process.env.SEED || 3);
const PLAN = ['wrong', 'wrong', 'right', 'right', 'right', 'right', 'right', 'right'];
const MOVE_S = [1.25, 1.35, 0.95, 0.9, 0.85, 0.85, 0.9, 1.0];   // seconds kept of each move (longer for the Hunter's)
const ESCAPE_S = 2.9;

(async () => {
  const { browser, page } = await open({ seed: SEED, quality: 'high' });
  const wait = ms => page.clock.runFor(ms);
  const until = async (fn, ms) => { for (let t = 0; t < (ms || 20000); t += 50) { if (await page.evaluate(fn)) return true; await wait(50); } return false; };
  const st = () => page.evaluate(() => CGB.games.outpace.state());
  await page.evaluate(() => { CGB.bank.setBoard('aqa'); CGB.bank.setSubject('physics'); CGB.app.show('outpace'); });
  await wait(300);
  await page.evaluate(() => { document.querySelector('#op-segGroups button[data-v="6"]').click(); document.getElementById('op-startBtn').click(); });
  await until(() => CGB.games.outpace.state().phase === 'deal');
  await wait(600);
  await page.keyboard.press('2');                                   // the Standard deal: Hunter 3 steps behind
  // only the 3D stage: every panel, label and card hidden
  await page.addStyleTag({ content: `#game-outpace > :not(#op-stage) { display: none !important; } #op-labels, .op-hostbar { display: none !important; }
    #op-stage { position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important; }` });
  // the reel camera: low behind the Hunter, looking past it to the class and HOME; it follows them both
  await page.evaluate(() => {
    let cx = null;
    window.__reelCam = (scene, camera, r) => {
      const gs = scene.children.filter(o => o.type === 'Group' && o.children.length === 1);
      const m = CGB.test.outpace.motion();
      const runner = gs.reduce((a, g) => Math.abs(g.position.x - m.runner[0]) < Math.abs(a.position.x - m.runner[0]) ? g : a, gs[0]);
      const hunter = gs.find(g => g !== runner) || runner;
      const rx = runner.position.x, hx = Math.max(hunter.position.x, rx + 1.2);
      // behind and beside the Hunter, high enough to see the gap close; in the escape it glides after the class
      const esc = !!window.__reelEsc;
      // straight down the track from behind the Hunter: the class ahead of it, HOME beyond
      const want = esc ? rx + 3.4 : hx + 2.7;
      cx = cx === null ? want : cx + (want - cx) * (esc ? 0.06 : 0.12);
      const size = r.getSize(new THREE.Vector2());
      camera.aspect = size.x / size.y; camera.fov = 52; camera.clearViewOffset();
      camera.position.set(cx, esc ? 1.25 : 1.45, 0.75);
      camera.lookAt(rx - 2.4, esc ? 0.85 : 0.55, -0.05);
      camera.updateProjectionMatrix(); camera.updateMatrixWorld();
    };
  });
  const reel = new Reel(page, 'reel-4-hunter');
  let vf = 0, started = false;
  const rec = async s => {
    if (!started) { started = true; await page.evaluate(() => { window.__reelSkip = false; }); await reel.start(); await reel.caption("One wrong answer and it's over…", { size: 92, top: 300 }); }
    for (let i = 0; i < Math.round(s * FPS); i++) { await page.clock.runFor(1000 / FPS); await reel.frame(true, true); vf++; }
  };
  for (let q = 0; q < PLAN.length; q++) {
    await until(() => { const s = CGB.games.outpace.state(); return s.round === 'think' || s.round === 'show' || s.round === 'mark'; });
    for (let k = 0; k < 6 && (await st()).round !== 'mark'; k++) { await page.keyboard.press('Space'); await wait(560); }
    if (PLAN[q] === 'right') for (let i = 1; i <= 6; i++) await page.keyboard.press(String(i));
    await page.keyboard.press('Enter');                              // confirm: the step happens now
    await rec(MOVE_S[q]);
    await until(() => CGB.games.outpace.state().round === 'done');
    await page.keyboard.press('Enter');                              // next question (or HOME)
    await wait(80);
  }
  const escape = vf;
  await page.evaluate(() => { window.__reelEsc = true; });
  await rec(0.35);                                                   // the game waits 300 ms, then the burst
  await reel.caption('');
  await rec(ESCAPE_S - 0.35);
  await reel.mark(true);
  await rec(0.9);
  console.log('reel-4-hunter frames', await reel.finish({ seed: SEED, plan: PLAN, cues: { escape } }));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
