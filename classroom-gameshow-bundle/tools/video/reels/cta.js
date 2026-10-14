/*
 * The call to action at the end of every reel (recording only): transparent 1080×1920 layers drawn by the app in its
 * own fonts and artwork, kept inside the platforms' safe zone (y 250 to 1470): the wordmark, a headline written for
 * that reel, a line under it, the button-style "Play free now on Tes" and "Link in bio"; and the host waving, peeking
 * in from the bottom right (frames of his wave, pip-<n>.png). cta.py animates them over the last frame.
 *   node tools/video/reels/cta.js   ->  marketing/reels/work/cta/<n>-<layer>.png
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');
const ROOT = path.join(__dirname, '..', '..', '..');
const OUT = path.join(ROOT, 'marketing', 'reels', 'work', 'cta');

const ALL = 'Turn your classroom into a game show';
// one hook per reel (the idea of that reel), the same button, and the shared line under it
const REELS = {
  1: { head: 'Can your class make it fall?', sub: ALL },
  2: { head: ALL, sub: 'No logins · Works offline' },
  3: { head: 'Revision this satisfying?', sub: ALL },
  4: { head: 'Can your class outrun the Hunter?', sub: ALL },
  5: { head: 'Win the jackpot with your class', sub: ALL },
  6: { head: 'Try this with your class', sub: ALL }
};
const BUTTON = 'Play free now on Tes';

const STYLE = `
html, body { margin: 0; background: transparent !important; overflow: hidden; }
body > :not(#cta):not(#pipbox) { display: none !important; }
#cta { position: fixed; inset: 0; }
.wm { position: absolute; left: 240px; top: 262px; width: 600px; }
.wm svg { display: block; width: 100%; height: auto; filter: drop-shadow(0 10px 0 rgba(0,0,0,0.4)); }
.head { position: absolute; left: 70px; right: 70px; top: 540px; text-align: center; }
.head .box { display: inline-block; font-family: var(--display, 'Lilita One'), sans-serif; color: #fff; font-size: 100px; line-height: 1.04; padding: 18px 42px 28px;
  background: rgba(20,24,52,0.94); border: 9px solid #FFC93C; border-radius: 40px; box-shadow: 0 14px 0 rgba(0,0,0,0.4); text-shadow: 0 6px 0 rgba(0,0,0,0.4); }
.sub { position: absolute; left: 60px; right: 60px; top: 965px; text-align: center; }
.sub span { display: inline-block; font-family: var(--body, 'Nunito'), sans-serif; font-weight: 900; font-size: 44px; line-height: 1.15; color: #1B1F3B; background: #FFF9F0; border: 6px solid #1B1F3B; border-radius: 40px; padding: 8px 34px 12px; box-shadow: 0 8px 0 rgba(0,0,0,0.3); }
.btn { position: absolute; left: 0; right: 0; top: 1075px; text-align: center; }
.btn span { display: inline-block; font-family: var(--display, 'Lilita One'), sans-serif; font-size: 80px; line-height: 1; color: #1B1F3B; padding: 22px 58px 30px;
  background: linear-gradient(180deg, #FFE27A 0%, #FFC93C 55%, #F5A91A 100%); border: 9px solid #1B1F3B; border-radius: 999px; box-shadow: 0 14px 0 #9A5B05, 0 22px 30px rgba(0,0,0,0.45); }
.bio { position: absolute; left: 110px; top: 1305px; }
.bio span { display: inline-block; font-family: var(--display, 'Lilita One'), sans-serif; font-size: 62px; line-height: 1; color: #fff; padding: 14px 40px 22px; background: rgba(20,24,52,0.94); border: 7px solid #FFC93C; border-radius: 999px; box-shadow: 0 10px 0 rgba(0,0,0,0.4); }
.bio b { color: #FFC93C; font-weight: 400; }
`;

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(ROOT, 'dist', 'editions', 'showtime-mega-bundle.html'));
  await page.waitForFunction(() => window.CGB && CGB.brand);
  await page.addStyleTag({ content: STYLE });
  await page.evaluate(() => document.fonts.ready);
  for (const [n, t] of Object.entries(REELS)) {
    const layers = {
      wm: '<div class="wm">${W}</div>',
      head: `<div class="head"><span class="box">${t.head}</span></div>`,
      btn: `<div class="btn"><span>${BUTTON}</span></div>`,
      bio: `<div class="bio"><span>Link in <b>bio</b></span></div>`,
      sub: `<div class="sub"><span>${t.sub.replace(/&/g, '&amp;')}</span></div>`
    };
    for (const [name, html] of Object.entries(layers)) {
      await page.evaluate(([h]) => { const old = document.getElementById('cta'); if (old) old.remove(); const d = document.createElement('div'); d.id = 'cta'; d.innerHTML = h.replace('${W}', CGB.brand.wordmark()); document.body.appendChild(d); }, [html]);
      await page.waitForTimeout(250);
      await page.screenshot({ path: path.join(OUT, `${n}-${name}.png`), omitBackground: true });
    }
    console.log('cta layers', n, JSON.stringify(t));
  }
  // the host, waving from the bottom right corner: frames of his wave (the wave swings, so a short run of frames loops)
  await page.evaluate(() => { const old = document.getElementById('cta'); if (old) old.remove(); });
  await page.addStyleTag({ content: '#pipbox { position: fixed; z-index: 9; right: -30px; bottom: -170px; width: 520px; height: 820px; } #pipbox > * { width: 100% !important; height: 100% !important; }' });
  await page.evaluate(() => { const d = document.createElement('div'); d.id = 'pipbox'; document.body.appendChild(d); window.__h = CGB.createHost2D(d, { className: 'host-launcher' }); window.__h.gesture('wave', 600000); });
  await page.waitForTimeout(1200);
  const CLIP = { x: 540, y: 1250, width: 540, height: 670 };
  for (let i = 0; i < 48; i++) { await page.waitForTimeout(35); await page.screenshot({ path: path.join(OUT, `pip-${String(i).padStart(2, '0')}.png`), omitBackground: true, clip: CLIP }); }
  fs.writeFileSync(path.join(OUT, 'pip.json'), JSON.stringify({ x: CLIP.x, y: CLIP.y, frames: 48 }));
  console.log('host frames');
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
