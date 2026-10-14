/*
 * Reel 6, "Answer before it falls": the text layers (recording only), drawn by the app in its own fonts and colours as
 * transparent PNGs in marketing/reels/work/reel6/: hook, question, timer numbers 7..1, the answer, the comment line
 * and the tag line. reel6.py lays them over the jackpot footage on the beat of its fall.
 *   node tools/video/reels/reel6.js
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');
const ROOT = path.join(__dirname, '..', '..', '..');
const OUT = path.join(ROOT, 'marketing', 'reels', 'work', 'reel6');

const STYLE = `
html, body { margin: 0; background: transparent !important; overflow: hidden; }
body > :not(#l) { display: none !important; }
#l { position: fixed; inset: 0; font-family: var(--display, 'Lilita One'), sans-serif; text-align: center; }
.abs { position: absolute; left: 0; right: 0; }
.dark { display: inline-block; color: #fff; background: rgba(20,24,52,0.94); border: 8px solid #FFC93C; border-radius: 40px; box-shadow: 0 12px 0 rgba(0,0,0,0.4); padding: 12px 40px 22px; text-shadow: 0 5px 0 rgba(0,0,0,0.4); line-height: 1.02; }
.gold { color: #FFC93C; }
.card { display: inline-block; color: #1B1F3B; background: #FFF9F0; border: 10px solid #1B1F3B; border-radius: 44px; box-shadow: 0 14px 0 rgba(0,0,0,0.4); padding: 14px 44px 24px; line-height: 1.08; font-size: 68px; }
.blank { display: inline-block; min-width: 400px; margin: 6px 0 2px; padding: 0 30px 4px; color: #B83A06; background: #FFE9B8; border: 8px dashed #1B1F3B; border-radius: 26px; font-size: 80px; line-height: 1.1; }
.pill { display: inline-block; font-family: var(--body, 'Nunito'), sans-serif; font-weight: 900; font-size: 50px; color: #1B1F3B; background: #FFF9F0; border: 6px solid #1B1F3B; border-radius: 999px; padding: 8px 38px 12px; box-shadow: 0 8px 0 rgba(0,0,0,0.35); }
.disc { position: absolute; left: 440px; top: 910px; width: 200px; height: 200px; border-radius: 50%; background: rgba(20,24,52,0.94); color: #fff; font-size: 150px; line-height: 200px; text-shadow: 0 6px 0 rgba(0,0,0,0.4); }
.ok { display: inline-block; color: #fff; background: #1E9E5A; border: 10px solid #0B5E33; border-radius: 44px; box-shadow: 0 14px 0 rgba(0,0,0,0.4); padding: 14px 52px 28px; font-size: 118px; line-height: 1; text-shadow: 0 6px 0 rgba(0,0,0,0.35); }
`;
const LAYERS = {
  hook: `<div class="abs" style="top:262px"><span class="dark" style="font-size:94px"><span class="gold" style="font-size:58px">YEAR 11s</span><br>Answer before it falls</span></div>`,
  question: `<div class="abs" style="top:486px"><span class="card">The mitochondria is the<br><span class="blank">?</span><br>of the cell</span></div>`,
  comment: `<div class="abs" style="top:1300px"><span class="pill">Type your answer in the comments</span></div>`,
  answer: `<div class="abs" style="top:965px"><span class="ok">&#10003; Powerhouse</span></div>
    <div class="abs" style="top:1165px"><span class="pill" style="font-size:44px">Site of aerobic respiration</span></div>`,
  tag: `<div class="abs" style="top:1300px"><span class="pill">Be honest: did you get it?</span></div>`,
  tag2: `<div class="abs" style="top:1300px"><span class="dark" style="font-size:62px">Tag the friend who forgot</span></div>`
};
for (let n = 1; n <= 7; n++) LAYERS['num' + n] = `<div class="disc">${n}</div>`;

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(ROOT, 'dist', 'editions', 'showtime-mega-bundle.html'));
  await page.waitForFunction(() => window.CGB && CGB.brand);
  await page.addStyleTag({ content: STYLE });
  await page.evaluate(() => document.fonts.ready);
  for (const [name, html] of Object.entries(LAYERS)) {
    await page.evaluate(h => { const o = document.getElementById('l'); if (o) o.remove(); const d = document.createElement('div'); d.id = 'l'; d.innerHTML = h; document.body.appendChild(d); }, html);
    await page.waitForTimeout(200);
    await page.screenshot({ path: path.join(OUT, name + '.png'), omitBackground: true });
    console.log('layer', name);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
