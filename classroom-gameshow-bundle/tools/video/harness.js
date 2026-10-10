/*
 * Recording harness for the demo video (never shipped). Opens the SCIENCE edition's test build at 1920×1080 on
 * High graphics with a controlled clock: every frame the page's time advances by exactly 1/60 s (Playwright's
 * page.clock), then one screenshot is taken, so the video is smooth however slowly this machine draws.
 *   - random seed fixed (window.__SHOWTIME_SEED__), no "Start game" gate, sound on
 *   - a fake AudioContext records every sound the game plays (the games' own synthesis calls) with its time
 *   - the software-renderer check is told it has a graphics chip, so Over the Edge draws in full quality
 *   - CSS animations and transitions are stepped by hand with the same clock
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');

const ROOT = path.join(__dirname, '..', '..');
const FILE = 'file://' + path.join(ROOT, 'test-build', 'editions', 'science.html');
const OUT = path.join(ROOT, 'marketing', 'video', 'work');
const FPS = 60;

const INIT = ([seed, q]) => {
  window.__Q = q;
  window.__SHOWTIME_SEED__ = seed;
  window.__SHOWTIME_NOGATE__ = true;
  try { localStorage.setItem('cgb.settings', JSON.stringify({ quality: (window.__Q || 'low'), sound: true })); } catch (e) { /* ignore */ }
  // the games ask the renderer's name to decide on light mode: say it is a real graphics chip
  [window.WebGLRenderingContext, window.WebGL2RenderingContext].forEach(C => {
    if (!C) return;
    const gp = C.prototype.getParameter;
    C.prototype.getParameter = function (p) { return p === 0x9246 ? 'ANGLE (NVIDIA GeForce RTX 3060)' : p === 0x9245 ? 'NVIDIA' : gp.call(this, p); };
  });
  // a fake AudioContext that only records (time comes from the controlled clock)
  window.__sounds = [];
  class Param {
    constructor(rec, key) { this.rec = rec; this.key = key; this.value = 0; this.rec[key] = []; }
    setValueAtTime(v, t) { this.rec[this.key].push(['set', t, v]); return this; }
    exponentialRampToValueAtTime(v, t) { this.rec[this.key].push(['exp', t, v]); return this; }
    linearRampToValueAtTime(v, t) { this.rec[this.key].push(['lin', t, v]); return this; }
    setTargetAtTime() { return this; } cancelScheduledValues() { return this; }
  }
  class FakeCtx {
    constructor() { this.state = 'running'; this.destination = {}; this.sampleRate = 44100; }
    get currentTime() { return performance.now() / 1000; }
    resume() { return Promise.resolve(); } suspend() { return Promise.resolve(); } close() { return Promise.resolve(); }
    createOscillator() {
      const rec = { type: 'sine', start: 0, stop: 0 }; window.__sounds.push(rec);
      const o = { connect() { return o; }, disconnect() {}, start(t) { rec.start = t; }, stop(t) { rec.stop = t; }, frequency: new Param(rec, 'freq'), detune: new Param({}, 'd') };
      Object.defineProperty(o, 'type', { set(v) { rec.type = v; }, get() { return rec.type; } });
      o._rec = rec; return o;
    }
    createGain() { const g = { connect(n) { if (n && n._rec) n._rec.gainNode = g._rec; return n; }, disconnect() {}, _rec: {} }; g.gain = new Param(g._rec, 'gain'); return g; }
    createBiquadFilter() { return { connect(n) { return n; }, disconnect() {}, frequency: new Param({}, 'f'), Q: new Param({}, 'q'), type: 'lowpass' }; }
    createBuffer() { return { getChannelData() { return new Float32Array(1); } }; }
    createBufferSource() { return { connect(n) { return n; }, start() {}, stop() {}, buffer: null, playbackRate: new Param({}, 'r') }; }
    decodeAudioData() { return Promise.resolve({}); }
  }
  window.AudioContext = window.webkitAudioContext = FakeCtx;
  // wire gain nodes to the oscillators they follow
  const cg = FakeCtx.prototype.createGain;
  FakeCtx.prototype.createGain = function () {
    const g = cg.call(this);
    const c = g.connect; g.connect = n => { return c.call(g, n); };
    return g;
  };
};

async function open(opts) {
  opts = opts || {};
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--hide-scrollbars'] });
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  page.on('pageerror', e => console.log('pageerror:', e.message));
  await page.addInitScript(INIT, [opts.seed || 20261004, opts.quality || 'low']);
  await page.goto(FILE + (opts.hash || ''));
  await page.waitForFunction(() => window.CGB && CGB.app);
  return { browser, page };
}

/* A shot: frames go to work/<name>/NNNNN.jpg at 30 per second (the page runs at 60 Hz underneath, so the physics
   is fed a steady 60 fps); the sounds are saved with their times, which are the page clock's */
const CAP_FPS = 30;
class Shot {
  constructor(page, name) {
    this.page = page; this.name = name; this.dir = path.join(OUT, name); this.n = 0; this.simMs = 0; this.ticks = 0; this.t0 = null; this.count = 0; this.marks = {};
    fs.rmSync(this.dir, { recursive: true, force: true }); fs.mkdirSync(this.dir, { recursive: true });
  }
  async start() {
    await this.page.clock.install({ time: 0 });
    await this.page.evaluate(() => { window.__sounds.length = 0; });
    await this.page.addStyleTag({ content: CSS });
    await this.page.evaluate(() => { const c = document.createElement('div'); c.id = 'vid-cap'; c.innerHTML = '<div class="vc-main"></div><div class="vc-sub"></div>'; document.body.appendChild(c); });
  }
  /* one 1/60 s tick of the page's clock; the CSS animations and transitions are stepped by hand with it */
  async tick() {
    const target = Math.round(++this.ticks * 1000 / FPS), dt = target - this.simMs; this.simMs = target;
    await this.page.clock.runFor(dt);
    await this.page.evaluate(dt => { document.getAnimations().forEach(a => { try { if (a.playState === 'running') a.pause(); a.currentTime = (a.currentTime || 0) + dt; } catch (e) { /* finished */ } }); }, dt);
  }
  /* one video frame (two ticks), saved */
  async frame(save) {
    await this.tick(); await this.tick();
    // PREVIEW=n keeps one frame in n (to look at a take quickly, without rendering the whole of it)
    if (save === false || (++this.count % (+process.env.PREVIEW || 1))) return;
    if (this.t0 === null) this.t0 = this.simMs;
    await this.page.screenshot({ path: path.join(this.dir, String(this.n++).padStart(5, '0') + '.jpg'), type: 'jpeg', quality: 93 });
  }
  /* run for s seconds: saving frames (default) or just letting time pass */
  async run(s, save) { const k = Math.round(s * CAP_FPS); for (let i = 0; i < k; i++) await this.frame(save); }
  /* the big on-screen caption (large, bold, in the Showtime fonts); '' hides it */
  async caption(main, sub, pos) { await this.page.evaluate(([m, b, p]) => { const c = document.getElementById('vid-cap'); c.querySelector('.vc-main').textContent = m || ''; c.querySelector('.vc-sub').textContent = b || ''; c.classList.toggle('on', !!m); c.classList.toggle('top', p === 'top'); }, [main, sub, pos]); }
  /* run until the page says so (checked every few frames), at most `max` seconds */
  async until(fn, max, arg) { for (let i = 0; i < Math.round((max || 10) * CAP_FPS); i++) { if (i % 3 === 0 && await this.page.evaluate(fn, arg)) return true; await this.frame(); } return false; }
  async untilQuiet(fn, max, arg) { for (let i = 0; i < Math.round((max || 10) * CAP_FPS / 3); i++) { if (await this.page.evaluate(fn, arg)) return true; await this.frame(false); } return false; }
  async key(k) { await this.page.keyboard.press(k); }
  mark(name) { this.marks[name] = this.n; }
  async finish() {
    const sounds = await this.page.evaluate(() => window.__sounds.filter(s => s.start || s.stop));
    fs.writeFileSync(path.join(this.dir, 'sounds.json'), JSON.stringify({ t0: (this.t0 || 0) / 1000, frames: this.n, fps: CAP_FPS, marks: this.marks, sounds }));
    return this.n;
  }
}
const CSS = `
#vid-cap { position: fixed; left: 0; right: 0; bottom: 64px; z-index: 99999; text-align: center; pointer-events: none; opacity: 0; transition: opacity .25s, transform .25s; transform: translateY(12px); }
#vid-cap.on { opacity: 1; transform: none; }
#vid-cap.top { top: 34px; bottom: auto; }
.topbar { display: none !important; }
#vid-cap .vc-main { display: inline-block; font-family: var(--display, 'Lilita One'), sans-serif; font-size: 76px; line-height: 1.1; color: #fff; padding: 8px 40px 12px; background: rgba(20,24,52,0.88); border: 6px solid #FFC93C; border-radius: 28px; box-shadow: 0 10px 0 rgba(0,0,0,0.35); text-shadow: 0 4px 0 rgba(0,0,0,0.35); }
#vid-cap .vc-main:empty { display: none; }
#vid-cap .vc-sub { margin-top: 12px; display: inline-block; font-family: var(--body, 'Nunito'), sans-serif; font-weight: 900; font-size: 40px; color: #1B1F3B; background: #FFF9F0; padding: 6px 28px; border-radius: 999px; border: 5px solid #1B1F3B; }
#vid-cap .vc-sub:empty { display: none; }
`;
module.exports = { open, Shot, OUT, ROOT, FPS, CAP_FPS };
