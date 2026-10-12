/*
 * The Tes listing cover (1600×1200, 4:3: the ratio Tes crops listing images to), drawn by the app in its own fonts and artwork over frames of the demo shots:
 * the gold jackpot counter teetering as the backdrop, the wordmark, "Science mega pack", a still and logo for each
 * of the four games, the host, and the trust points. Render the shots first (tools/video/demo/build.sh).
 *   node tools/video/demo/cover.js   ->  marketing/video/tes-cover.png
 */
'use strict';
const path = require('path');
const fs = require('fs');
const L = require('./lib.js');

const W = L.WORK;
const cue = (shot, k, off) => { const r = JSON.parse(fs.readFileSync(path.join(W, shot, 'sounds.json'))); return Math.max(0, Math.min(r.frames - 1, r.cues[k] + (off || 0))); };
const data = (shot, n) => 'data:image/jpeg;base64,' + fs.readFileSync(path.join(W, shot, String(n).padStart(5, '0') + '.jpg')).toString('base64');
const TILES = [
  ['oteLogo', data('ote', cue('ote', 'lanes', 40))],
  ['opLogo', data('outpace', cue('outpace', 'move1', 60))],
  ['ccLogo', data('cc', cue('cc', 'pick', -10))],
  ['hhLogo', data('hh', cue('hh', 'win', 120))]
];
const BG = data('jackpot', cue('jackpot', 'fall', -40));

(async () => {
  const { browser, page } = await L.open({ seed: 1, width: 1600, height: 1200 });
  await page.addStyleTag({ content: `
    body > :not(#cv) { display: none !important; }
    #cv { position: fixed; inset: 0; overflow: hidden; background: #141834; }
    #cv .bg { position: absolute; inset: 0; background-size: cover; background-position: 30% 60%; filter: saturate(1.1); }
    #cv .shade { position: absolute; inset: 0; background: linear-gradient(180deg, rgba(14,16,44,0.86) 0%, rgba(14,16,44,0.55) 40%, rgba(14,16,44,0.82) 100%); }
    #cv .wm { position: absolute; left: 60px; top: 28px; width: 780px; }
    #cv .wm svg, #cv .lg svg { display: block; width: 100%; height: auto; filter: drop-shadow(0 10px 0 rgba(0,0,0,0.4)); }
    #cv .hl { position: absolute; left: 66px; top: 430px; font-family: var(--display); font-size: 88px; line-height: 1; color: #fff; text-shadow: 0 6px 0 rgba(0,0,0,0.45); }
    #cv .hl b { font-weight: 400; color: #FFC93C; }
    #cv .bdg { position: absolute; left: 66px; top: 560px; font-family: var(--display); font-size: 48px; color: #1B1F3B; background: #FFC93C; border: 5px solid #1B1F3B; border-radius: 16px; padding: 4px 24px 8px; box-shadow: 0 8px 0 rgba(0,0,0,0.35); transform: rotate(-2deg); }
    #cv #pip { position: absolute; right: 60px; top: 30px; width: 420px; height: 660px; }
    #cv .row { position: absolute; left: 40px; right: 40px; top: 740px; display: flex; gap: 20px; }
    #cv .g { flex: 1; text-align: center; }
    #cv .g .im { height: 207px; border-radius: 18px; border: 6px solid #FFC93C; background-size: cover; background-position: center; box-shadow: 0 10px 0 rgba(0,0,0,0.4); }
    #cv .g .lg { width: 220px; margin: 22px auto 0; }
    #cv .pills { position: absolute; left: 0; right: 0; bottom: 44px; display: flex; justify-content: center; gap: 14px; }
    #cv .pill { font-family: var(--body); font-weight: 900; font-size: 25px; color: #1B1F3B; background: #FFF9F0; border: 5px solid #1B1F3B; border-radius: 999px; padding: 6px 24px 9px; box-shadow: 0 6px 0 rgba(0,0,0,0.35); }
  ` });
  await page.evaluate(([tiles, bg]) => {
    const e = document.createElement('div'); e.id = 'cv';
    e.innerHTML = `<div class="bg" style="background-image:url('${bg}')"></div><div class="shade"></div>
      <div class="wm">${CGB.brand.wordmark()}</div>
      <div class="hl">Whole-class <b>GCSE revision</b></div>
      <div class="bdg">Science mega pack</div>
      <div id="pip"></div>
      <div class="row">${tiles.map(([f, d]) => `<div class="g"><div class="im" style="background-image:url('${d}')"></div><div class="lg">${CGB.brand[f]()}</div></div>`).join('')}</div>
      <div class="pills"><span class="pill">Every team answers</span><span class="pill">AQA &amp; Edexcel</span><span class="pill">Separate &amp; Combined</span><span class="pill">Works offline</span></div>`;
    document.body.appendChild(e);
    CGB.createHost2D(document.getElementById('pip'), { className: 'host-launcher' }).gesture('wave', 600000);
  }, [TILES, BG]);
  await page.evaluate(() => document.fonts.ready);
  await L.timer(page).wait(1200);
  await L.still(page, path.join(L.ROOT, 'marketing', 'video', 'tes-cover.png'));
  console.log('cover');
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
