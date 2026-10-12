/*
 * Demo video recording core (recording only; never shipped). The 16:9 counterpart of ../reels/reel.js: it opens the
 * Science edition's test build at 1920×1080 on High graphics, on the same controlled clock, with the same seeded
 * randomness, renderer trap (the shot camera), slow-motion hook and recorded sounds. The game's own panels stay on
 * screen; only the top bar's buttons are hidden, as in a clean full-screen recording.
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { open: openReel, Reel, CSS, ROOT, FPS } = require('../reels/reel.js');

const W = 1920, H = 1080;
const FILE = path.join(ROOT, 'test-build', 'editions', 'science.html');
const WORK = path.join(ROOT, 'marketing', 'video', 'work');
const TEAMS = ['Electrons', 'Neutrons', 'Protons', 'Photons', 'Ions', 'Atoms'];   // short enough to show in full on every team panel
// the reel layer's styles without its "hide every panel" rule: here only the top bar's buttons and notes go
const BASE = CSS.replace(/^\s*\.topbar[^\n]*\n/m, '\n.topbar, .gpu-note, #storageNote, .toast { display: none !important; }\n');

async function open(opts) {
  opts = Object.assign({ width: W, height: H, file: FILE, baseCss: BASE }, opts || {});
  const r = await openReel(opts);
  await r.page.evaluate(t => CGB.saveTeamNames(t), TEAMS);
  return r;
}

/* time on the page's fake clock, and a wait-until on it */
function timer(page) {
  const wait = ms => page.clock.runFor(ms);
  const until = async (fn, ms, arg) => { for (let t = 0; t < (ms || 20000); t += 50) { if (await page.evaluate(fn, arg)) return true; await wait(50); } return false; };
  return { wait, until };
}

/* a recorded shot: frames into work/<name>/, with named cue frames saved for the edit */
class Shot extends Reel {
  constructor(page, name) { super(page, name, WORK); this.cues = {}; this.times = []; }
  cue(k) { this.cues[k] = this.n; }
  /* s seconds of frames; each frame's page time is kept, so the edit can place the game's sounds even where the
     script skipped time between recorded moves */
  async run(s, each) {
    const k = Math.round(s * FPS);
    for (let i = 0; i < k; i++) {
      if (each) await each(i);
      await this.page.clock.runFor(1000 / FPS);
      // CSS animations and transitions run on the browser's own clock: step them by the same 1/60 s by hand
      this.times.push(await this.page.evaluate(dt => {
        document.getAnimations().forEach(a => { try { if (a.playState === 'running') a.pause(); if (a.playState !== 'finished') a.currentTime = (a.currentTime || 0) + dt; } catch (e) { /* finished */ } });
        return performance.now() / 1000;
      }, 1000 / FPS));
      await this.frame(true, true);
    }
  }
  async done(meta) { const n = await this.finish(Object.assign({ cues: this.cues, times: this.times }, meta || {})); console.log(this.name, 'frames', n, (n / FPS).toFixed(2) + 's', JSON.stringify(this.cues)); return n; }
}

/* the question round as a teacher runs it: Space for "3, 2, 1, show me!" until marking opens, then keys */
async function toMarking(page, T, game) {
  for (let k = 0; k < 6; k++) {
    if ((await page.evaluate(g => CGB.games[g].state().round, game)) === 'mark') return true;
    await page.keyboard.press('Space'); await T.wait(560);
  }
  return false;
}

/* a still: straight from DevTools (the paused clock never gives Playwright's screenshot its animation frame) */
async function still(page, file) {
  const cdp = await page.context().newCDPSession(page);
  const s = await cdp.send('Page.captureScreenshot', { format: 'png' });
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.from(s.data, 'base64'));
  await cdp.detach();
}

module.exports = { open, timer, Shot, toMarking, still, W, H, FPS, WORK, ROOT, TEAMS, FILE };
