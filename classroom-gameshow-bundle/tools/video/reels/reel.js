/*
 * Reel recording core (recording only; never shipped). Opens the test build at 1080×1920 on a controlled clock and
 * adds a "reel camera": every WebGL renderer the page makes is wrapped, so just before each render the reel script
 * can place the 3D camera anywhere (window.__reelCam). The game code is untouched; its own panels are hidden by CSS
 * and the 3D stage fills the tall frame. Frames are taken at 60 per second, one per 1/60 s tick of the page clock.
 * Sounds the games make are recorded (fake AudioContext from ../harness.js) and rebuilt later by mix.py.
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');

const ROOT = path.join(__dirname, '..', '..', '..');
const FILE = 'file://' + path.join(ROOT, 'test-build', 'showtime-test.html');
const WORK = path.join(ROOT, 'marketing', 'reels', 'work');
const W = 1080, H = 1920, FPS = 60;
const SAFE = { top: 250, bottom: 450 };

// runs in the page before any of its scripts: the renderer trap, recorded sounds and fixed settings
const INIT = ([seed, quality]) => {
  window.__SHOWTIME_SEED__ = seed;
  // decoration also uses Math.random (some of it lands on the tray), so it is seeded too: a take replays exactly
  let a = (seed * 2654435761) >>> 0;
  Math.random = () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  try { localStorage.setItem('cgb.settings', JSON.stringify({ quality, sound: true, music: false, reducedMotion: false })); } catch (e) { /* ignore */ }
  [window.WebGLRenderingContext, window.WebGL2RenderingContext].forEach(C => {
    if (!C) return;
    const gp = C.prototype.getParameter;
    C.prototype.getParameter = function (p) { return p === 0x9246 ? 'ANGLE (NVIDIA GeForce RTX 3060)' : p === 0x9245 ? 'NVIDIA' : gp.call(this, p); };
  });
  // slow motion: the time requestAnimationFrame hands the page can run slower than the clock (always installed, so
  // a searched take and a rendered one run the same code)
  {
    let k = 1, base = null, scaled = 0;
    window.__reelSlow = v => { k = v; };
    const wrap = raf => cb => raf(t => { if (base === null) { base = t; scaled = t; } scaled += (t - base) * k; base = t; cb(scaled); });
    let cur = wrap(window.requestAnimationFrame.bind(window));
    Object.defineProperty(window, 'requestAnimationFrame', { configurable: true, get() { return cur; }, set(v) { cur = wrap(v); } });
  }
  // the reel camera: wrap THREE.WebGLRenderer as soon as the library defines it
  let NS = null;
  Object.defineProperty(window, 'THREE', {
    configurable: true,
    get() { return NS; },
    set(v) {
      NS = v; let Orig = null;
      Object.defineProperty(v, 'WebGLRenderer', {
        configurable: true, enumerable: true,
        get() { return Orig && Orig.__reelWrapped; },
        set(C) {
          Orig = C;
          function Wrapped(...a) {
            const r = new C(...a), render = r.render;
            r.render = function (scene, camera) {
              if (window.__reelSkip) return;          // replaying a take quickly: no drawing at all
              if (camera && camera.isPerspectiveCamera) window.__reelLast = { scene, camera, renderer: r };
              if (window.__reelCam && camera && camera.isPerspectiveCamera) window.__reelCam(scene, camera, r);
              return render.call(this, scene, camera);
            };
            (window.__reelRenderers = window.__reelRenderers || []).push(r);
            return r;
          }
          Wrapped.prototype = C.prototype;
          C.__reelWrapped = Wrapped;
        }
      });
    }
  });
  // recorded sounds: the games synthesise every effect with oscillators; this fake context keeps a list
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
      const o = { connect(n) { if (n && n._rec) rec.gain = n._rec; return n; }, disconnect() {}, start(t) { rec.start = t; }, stop(t) { rec.stop = t; }, frequency: new Param(rec, 'freq'), detune: new Param({}, 'd') };
      Object.defineProperty(o, 'type', { set(v) { rec.type = v; }, get() { return rec.type; } });
      return o;
    }
    createGain() { const g = { connect(n) { return n; }, disconnect() {}, _rec: {} }; g.gain = new Param(g._rec, 'gain'); return g; }
    createBiquadFilter() { return { connect(n) { return n; }, disconnect() {}, frequency: new Param({}, 'f'), Q: new Param({}, 'q'), type: 'lowpass' }; }
    createBuffer() { return { getChannelData() { return new Float32Array(1); } }; }
    createBufferSource() { return { connect(n) { return n; }, start() {}, stop() {}, buffer: null, playbackRate: new Param({}, 'r') }; }
    decodeAudioData() { return Promise.resolve({ duration: 1 }); }
  }
  window.AudioContext = window.webkitAudioContext = FakeCtx;
};

/* the tall stage: hide every panel, let the game's 3D stage (or a 2D board) fill the frame, and add the
   caption layer and the optional safe-zone guide */
const CSS = `
.topbar, .l-tools, #mascot, .host-bar, .pause-chip, .toast, .gpu-note, #storageNote { display: none !important; }
#reel-cap { position: fixed; left: 60px; right: 60px; z-index: 99999; text-align: center; pointer-events: none; opacity: 0; }
#reel-cap.on { opacity: 1; }
#reel-cap .rc { display: inline-block; white-space: pre-line; font-family: var(--display, 'Lilita One'), sans-serif; color: #fff; line-height: 1.05; padding: 14px 40px 20px; background: rgba(20,24,52,0.9); border: 7px solid #FFC93C; border-radius: 34px; box-shadow: 0 12px 0 rgba(0,0,0,0.35); text-shadow: 0 5px 0 rgba(0,0,0,0.35); }
#reel-cap.small .rc { font-family: var(--body, 'Nunito'), sans-serif; font-weight: 800; background: rgba(0,0,0,0.35); border: 0; box-shadow: none; text-shadow: 0 2px 6px rgba(0,0,0,0.6); letter-spacing: 0.04em; }
#reel-mark { position: fixed; right: 40px; z-index: 99999; font-family: var(--display, 'Lilita One'), sans-serif; font-size: 44px; color: #FFC93C; text-shadow: 0 4px 0 rgba(0,0,0,0.45); opacity: 0; pointer-events: none; }
#reel-mark.on { opacity: 0.95; }
#reel-safe { position: fixed; inset: 0; z-index: 99998; pointer-events: none; display: none; }
#reel-safe.on { display: block; }
#reel-safe .z { position: absolute; left: 0; right: 0; background: rgba(255,0,80,0.35); border: 3px dashed #ff0050; }
`;

async function open(opts) {
  opts = opts || {};
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--hide-scrollbars'] });
  const ctx = await browser.newContext({ viewport: { width: opts.width || W, height: opts.height || H }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  page.on('pageerror', e => console.log('pageerror:', e.message));
  await page.addInitScript(INIT, [opts.seed || 20261004, opts.quality || 'high']);
  // the fake clock from the very start, so nothing in the take depends on how fast this machine is
  // (paused: time moves only when the script says so)
  await page.clock.install({ time: 0 });
  await page.clock.pauseAt(1000);
  await page.goto((opts.file ? 'file://' + opts.file : FILE) + (opts.hash || ''));
  await page.waitForFunction(() => window.CGB && CGB.app);
  await page.addStyleTag({ content: (opts.baseCss != null ? opts.baseCss : CSS) + (opts.css || '') });
  await page.evaluate(([safe]) => {
    const c = document.createElement('div'); c.id = 'reel-cap'; c.innerHTML = '<div class="rc"></div>'; document.body.appendChild(c);
    const m = document.createElement('div'); m.id = 'reel-mark'; m.textContent = 'Showtime'; document.body.appendChild(m);
    // confetti for the big moments (recording only): pieces fall from the top, driven by the page's clock
    const cv = document.createElement('canvas'); cv.id = 'reel-confetti'; cv.width = innerWidth; cv.height = innerHeight;
    cv.style.cssText = 'position:fixed;inset:0;z-index:99997;pointer-events:none'; document.body.appendChild(cv);
    const g = cv.getContext('2d'); let bits = [], last = null;
    const cols = ['#FFC93C', '#FF5D8F', '#3EC1F3', '#7CE577', '#FFFFFF', '#B57BFF'];
    window.__reelConfetti = (n, ox, oy) => { for (let i = 0; i < (n || 260); i++) { const a = Math.random() * Math.PI * 2, v = 500 + Math.random() * 1100;
      bits.push({ x: ox == null ? Math.random() * innerWidth : ox, y: oy == null ? -20 : oy, vx: Math.cos(a) * v * 0.6, vy: (oy == null ? Math.random() * 300 : Math.sin(a) * v - 600), r: Math.random() * 6.28, vr: (Math.random() - 0.5) * 14, w: 14 + Math.random() * 14, h: 8 + Math.random() * 8, c: cols[i % cols.length] }); } };
    // a big gold label that pops in (the jackpot's "+£5,000"), animated on the same clock
    const pop = document.createElement('div'); pop.id = 'reel-pop';
    pop.style.cssText = "position:fixed;left:0;right:0;top:42%;text-align:center;z-index:99998;pointer-events:none;font-family:var(--display,'Lilita One'),sans-serif;font-size:170px;color:#FFC93C;-webkit-text-stroke:8px #1B1F3B;paint-order:stroke fill;text-shadow:0 10px 0 rgba(0,0,0,0.4);opacity:0";
    document.body.appendChild(pop); let popT = -1, popHold = 0;
    // hold: seconds on screen before it fades away (0 = it stays)
    window.__reelPop = (text, hold) => { pop.textContent = text; popT = 0; popHold = hold || 0; };
    const tick = t => { const dt = last === null ? 0 : Math.min(0.05, (t - last) / 1000); last = t; g.clearRect(0, 0, cv.width, cv.height);
      if (popT >= 0) { popT += dt; const k = Math.min(1, popT / 0.35), sc = k < 1 ? 0.3 + 0.9 * Math.sin(k * Math.PI / 2) * (1 + 0.25 * Math.sin(k * Math.PI)) : 1 + 0.03 * Math.sin(popT * 6); pop.style.opacity = Math.max(0, Math.min(1, popT / 0.12, popHold ? (popHold + 0.3 - popT) / 0.3 : 1)); pop.style.transform = `scale(${sc.toFixed(3)}) rotate(${(-4 * (1 - k)).toFixed(2)}deg)`; }
      bits = bits.filter(b => b.y < innerHeight + 40); for (const b of bits) { b.vy += 1400 * dt; b.vx *= 0.99; b.x += b.vx * dt; b.y += b.vy * dt; b.r += b.vr * dt;
        g.save(); g.translate(b.x, b.y); g.rotate(b.r); g.fillStyle = b.c; g.fillRect(-b.w / 2, -b.h / 2, b.w, b.h); g.restore(); }
      requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    const s = document.createElement('div'); s.id = 'reel-safe'; s.innerHTML = `<div class="z" style="top:0;height:${safe.top}px"></div><div class="z" style="bottom:0;height:${safe.bottom}px"></div>`; document.body.appendChild(s);
  }, [SAFE]);
  if (process.env.SAFE_DEBUG) await page.evaluate(() => document.getElementById('reel-safe').classList.add('on'));
  return { browser, page };
}

class Reel {
  constructor(page, name, work) {
    this.page = page; this.name = name; this.dir = path.join(work || WORK, name); this.n = 0; this.ticks = 0; this.simMs = 0; this.t0 = null;
    fs.rmSync(this.dir, { recursive: true, force: true }); fs.mkdirSync(this.dir, { recursive: true });
    this.keepEvery = +(process.env.PREVIEW || 1);
  }
  /* start recording; the page clock must already be installed (fake time) */
  async start() {
    this.t0Page = await this.page.evaluate(() => performance.now());
    await this.page.evaluate(() => { window.__sounds.length = 0; });
  }
  async tick() {
    const target = Math.round(++this.ticks * 1000 / FPS), dt = target - this.simMs; this.simMs = target;
    await this.page.clock.runFor(dt);
    await this.page.evaluate(dt => { document.getAnimations().forEach(a => { try { if (a.playState === 'running') a.pause(); a.currentTime = (a.currentTime || 0) + dt; } catch (e) { /* finished */ } }); }, dt);
  }
  /* one frame: one tick, then a screenshot (PREVIEW=n keeps one frame in n, for quick looks) */
  async frame(save, noTick) {
    if (!noTick) await this.tick();
    if (save === false) return;
    const k = this.n++;
    if (k % this.keepEvery) return;
    // straight from DevTools: Playwright's own screenshot waits for an animation frame, which the paused clock never gives
    if (!this.cdp) this.cdp = await this.page.context().newCDPSession(this.page);
    const shot = await this.cdp.send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
    fs.writeFileSync(path.join(this.dir, String(k).padStart(5, '0') + '.jpg'), Buffer.from(shot.data, 'base64'));
  }
  async run(s, save, each) { const k = Math.round(s * FPS); for (let i = 0; i < k; i++) { if (each) await each(i / k, i); await this.frame(save); } }
  async caption(text, opts) {
    opts = opts || {};
    await this.page.evaluate(([t, o]) => {
      const c = document.getElementById('reel-cap'), r = c.querySelector('.rc');
      r.textContent = t || ''; c.classList.toggle('on', !!t); c.classList.toggle('small', !!o.small);
      r.style.fontSize = (o.size || 96) + 'px'; c.style.top = (o.top == null ? 320 : o.top) + 'px';
    }, [text, opts]);
  }
  async mark(on) { await this.page.evaluate(([on, b]) => { const m = document.getElementById('reel-mark'); m.classList.toggle('on', !!on); m.style.bottom = (b + 24) + 'px'; }, [on, SAFE.bottom]); }
  async finish(meta) {
    const sounds = await this.page.evaluate(() => window.__sounds.filter(s => s.start || s.stop));
    fs.writeFileSync(path.join(this.dir, 'sounds.json'), JSON.stringify(Object.assign({ t0: (this.t0Page || 0) / 1000, frames: this.n, fps: FPS, sounds }, meta || {})));
    return this.n;
  }
}

module.exports = { open, Reel, WORK, ROOT, W, H, FPS, SAFE, CSS };
