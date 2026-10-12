/*
 * Stills for the reels (recording only), drawn in the app's own fonts:
 *   - the caption layer for reel 2 (transparent PNG, 1080×1920)
 *   - a cover for each reel: its most eye-catching frame with large text kept inside the centre square
 *     (y 420–1500), so the cover still reads when a grid crops it to a square
 *   node tools/video/reels/stills.js
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');
const ROOT = path.join(__dirname, '..', '..', '..');
const WORK = path.join(ROOT, 'marketing', 'reels', 'work');
const OUT = path.join(ROOT, 'marketing', 'reels');

const COVERS = [
  { name: 'reel-1-will-it-fall', cue: 'fall', off: -40, title: 'Will it fall?', sub: '' },
  { name: 'reel-2-pov-game-show', frame: 70, title: 'POV:', sub: 'you turned GCSE revision into a game show' },
  { name: 'reel-3-oddly-satisfying', cue: 'fall', off: 60, title: 'oddly satisfying', sub: '', small: true },
  { name: 'reel-4-hunter', frame: 150, title: "One wrong answer…", sub: "and it's over" },
  { name: 'reel-5-jackpot', cue: 'fall', off: -30, title: '£5,000', sub: 'jackpot · last question' }
];

const STYLE = `html,body{margin:0;width:1080px;height:1920px;overflow:hidden;background:transparent}
  #st{position:fixed;inset:0;z-index:2147483647;background-size:cover;background-position:center}
  .t{position:absolute;left:60px;right:60px;text-align:center;font-family:var(--display,'Lilita One'),sans-serif;color:#fff;line-height:1.02}
  .box{display:inline-block;padding:16px 44px 24px;background:rgba(20,24,52,0.9);border:8px solid #FFC93C;border-radius:38px;box-shadow:0 14px 0 rgba(0,0,0,0.35);text-shadow:0 6px 0 rgba(0,0,0,0.35)}
  .big{font-size:150px}.sub{display:inline-block;margin-top:22px;font-family:var(--body,'Nunito'),sans-serif;font-weight:900;font-size:58px;color:#1B1F3B;background:#FFF9F0;padding:10px 34px;border-radius:999px;border:6px solid #1B1F3B}
  .small{font-family:var(--body,'Nunito'),sans-serif;font-weight:800;font-size:96px;letter-spacing:.04em;text-shadow:0 4px 14px rgba(0,0,0,.7)}`;

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(ROOT, 'test-build', 'showtime-test.html'));
  await page.waitForFunction(() => window.CGB && CGB.app);
  await page.addStyleTag({ content: STYLE });
  const draw = (bg, html) => page.evaluate(([bg, html]) => {
    let st = document.getElementById('st'); if (!st) { st = document.createElement('div'); st.id = 'st'; document.body.appendChild(st); }
    st.style.backgroundImage = bg ? `url("${bg}")` : 'none'; st.innerHTML = html;
  }, [bg, html]);
  await page.evaluate(() => document.fonts.ready);
  // reel 2's caption layer: on screen all the way through, inside the safe zone
  await page.evaluate(() => { document.body.style.background = 'transparent'; document.querySelectorAll('body > :not(#st)').forEach(e => { e.style.visibility = 'hidden'; }); });
  await draw('', `<div class="t" style="top:290px"><div class="box" style="font-size:70px">POV: you turned GCSE<br>revision into a game show</div></div>`);
  await page.screenshot({ path: path.join(WORK, 'caption-reel-2.png'), omitBackground: true });
  for (const c of COVERS) {
    const d = path.join(WORK, c.name);
    let frame = c.frame || 0;
    if (c.cue) { const rec = JSON.parse(fs.readFileSync(path.join(d, 'sounds.json'))); frame = Math.max(0, ((rec.cues || {})[c.cue] || 0) + c.off); }
    const src = path.join(d, String(frame).padStart(5, '0') + '.jpg');
    if (!fs.existsSync(src)) { console.log('missing', src); continue; }
    const text = c.small ? `<div class="t small" style="top:860px">${c.title}</div>`
      : `<div class="t" style="top:${c.sub ? 760 : 840}px"><div class="box big">${c.title}</div>${c.sub ? `<br><span class="sub">${c.sub}</span>` : ''}</div>`;
    await draw('file://' + src, text);
    await page.screenshot({ path: path.join(OUT, c.name + '-cover.jpg'), type: 'jpeg', quality: 92 });
    console.log('cover', c.name, 'from frame', frame);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
