/*
 * Renders an Over the Edge reel shot: replays a take (seed + the same actions) found by search-ote.js and records
 * the frames from `from` to `to` (physics frames) through the reel camera.
 *   node tools/video/reels/render-ote.js <shot.json>
 * shot.json: { name, seed, questions, from, to, slow (1 = real time, 2 = half speed), cam: [{at, pos, look, fov}],
 *              captions: [{at, until, text, size, top, small}], freeze (seconds held on the last frame), mark (seconds of
 *              "Showtime" mark at the end), stage: 'high'|'low', width/height (default 1080×1920), work (output folder
 *              under the repo, default marketing/reels/work), pop: {text, confetti, hold} at the first fall }
 * Camera keys are in world units (the tray's front edge is z = 5.5, its surface y = 0, x from -6 to 6); `at` is
 * seconds from the start of the shot; positions ease between keys.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { open, Reel, W, H, FPS } = require('./reel.js');
const ote = require('./ote-lib.js');

const { STAGE_CSS } = ote;

const ease = t => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
function camAt(keys, t) {
  let a = keys[0], b = keys[keys.length - 1];
  for (let i = 0; i < keys.length - 1; i++) if (t >= keys[i].at && t <= keys[i + 1].at) { a = keys[i]; b = keys[i + 1]; break; }
  if (t <= keys[0].at) a = b = keys[0];
  if (t >= keys[keys.length - 1].at) a = b = keys[keys.length - 1];
  const f = a === b ? 0 : (b.ease === 'linear' ? (t - a.at) / (b.at - a.at) : ease((t - a.at) / (b.at - a.at)));
  const L = (u, v) => u.map((x, i) => x + (v[i] - x) * f);
  return { pos: L(a.pos, b.pos), look: L(a.look, b.look), fov: (a.fov || 40) + ((b.fov || 40) - (a.fov || 40)) * f };
}

(async () => {
  const shot = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const slow = shot.slow || 1;
  const { browser, page } = await open({ seed: shot.seed, quality: shot.stage || 'high', css: STAGE_CSS, width: shot.width, height: shot.height });
  // slow motion: the game's frame clock (the time requestAnimationFrame hands it) can run slower than real time;
  // physics, falling counters and camera all slow together, and every video frame is a real render
  const T = ote.timer(page, true);
  await ote.setup(page, T);
  const reel = new Reel(page, shot.name, shot.work ? path.join(__dirname, '..', '..', '..', shot.work) : undefined);
  let recording = false, vf = 0;
  await page.evaluate(() => {
    window.__reelCamKey = null;
    window.__reelCam = (scene, camera, r) => {
      const k = window.__reelCamKey; if (!k) return;
      const size = r.getSize(new THREE.Vector2());
      camera.aspect = size.x / size.y; camera.fov = k.fov; camera.clearViewOffset();
      camera.position.set(k.pos[0], k.pos[1], k.pos[2]); camera.up.set(0, 1, 0); camera.lookAt(k.look[0], k.look[1], k.look[2]);
      camera.updateProjectionMatrix(); camera.updateMatrixWorld();
    };
  });
  const capAt = async t => {
    const c = (shot.captions || []).find(c => t >= c.at && t < c.until);
    const key = c ? JSON.stringify(c) : '';
    if (key !== reel._cap) { reel._cap = key; await reel.caption(c ? c.text : '', c || {}); }
    const m = !!shot.mark && t >= reel._len - shot.mark;
    if (m !== reel._mark) { reel._mark = m; await reel.mark(m); }
  };
  reel._len = (shot.to - shot.from) * slow / FPS + (shot.freeze || 0);
  // physics frames of the take; inside the window each one becomes `slow` video frames
  const runner = shot.final ? (step => ote.playFinal(page, T, shot.near, shot.questions || 3, step)) : (step => ote.play(page, T, shot.questions || 6, step));
  await runner(async (n, frame) => {
    if (frame >= shot.to) return null;
    if (!recording && frame + n <= shot.from) return ote.advance(page, n);
    const out = [];
    for (let i = 0; i < n; i++) {
      const fr = frame + i;
      if (fr >= shot.to) return null;
      if (fr < shot.from) { out.push(...await ote.advance(page, 1)); continue; }
      if (!recording) {
        recording = true;
        await page.evaluate(() => { window.__reelSkip = false; });
        await page.evaluate(k => window.__reelSlow(1 / k), slow);
        await page.evaluate(k => { window.__reelCamKey = k; }, camAt(shot.cam, 0));
        await reel.start();
      }
      // one physics step (exactly as in the search), then `slow` video frames of the page clock
      const m = await page.evaluate(() => { CGB.test.ote.advance(1 / 60); const t = CGB.test.ote.info().tray; return t ? t.split(';').length : 0; });
      if (shot.final && !reel._jf && await page.evaluate(() => CGB.test.ote.info().jackpotFell)) {
        reel._jf = true; (reel._cues = reel._cues || {}).fall = vf; console.log('jackpot fell at physics frame', fr, '(video', (vf / FPS).toFixed(2) + 's)');
        await page.evaluate(() => { window.__reelConfetti(340); window.__reelPop('+£5,000'); });
      }
      if (reel._n != null && m < reel._n && !(reel._cues = reel._cues || {}).fall) {
        reel._cues.fall = vf;
        if (shot.pop) await page.evaluate(p => { window.__reelPop(p.text, p.hold); if (p.confetti) window.__reelConfetti(p.confetti); }, shot.pop);
      }
      if (reel._n != null && m < reel._n) console.log('fall at physics frame', fr, '(video', (vf / FPS).toFixed(2) + 's)', reel._n - m);
      reel._n = m;
      for (let s = 0; s < slow; s++) {
        const t = vf / FPS;
        await page.evaluate(k => { window.__reelCamKey = k; }, camAt(shot.cam, t));
        await capAt(t);
        await page.clock.runFor(1000 / FPS);
        await reel.frame(true, true);
        vf++;
      }
      out.push(null);
    }
    return out;
  });
  // hold the last frame
  await page.evaluate(() => { window.__reelSlow(0); });   // frozen: nothing moves, the picture holds
  for (let i = 0; i < Math.round((shot.freeze || 0) * FPS); i++) { await capAt(vf / FPS); await page.clock.runFor(1000 / FPS); await reel.frame(true, true); vf++; }
  console.log(shot.name, 'frames', await reel.finish({ slow, shot, cues: reel._cues || {} }));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
