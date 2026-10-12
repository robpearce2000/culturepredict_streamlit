/*
 * Captions and title cards for the demo videos (recording only), drawn by the app itself in its own fonts,
 * colours and artwork (the Showtime wordmark, the game logos and the host), as transparent PNGs that the edit
 * lays over the footage:  node tools/video/demo/cards.js   ->  marketing/video/work/cards/<name>.png
 *   16:9 (1920×1080): one PNG per caption, already in its place on the frame
 *   9:16 (1080×1920): the reel's captions, inside the platform safe zone (top 250 px and bottom 450 px kept clear)
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');
const ROOT = path.join(__dirname, '..', '..', '..');
const OUT = path.join(ROOT, 'marketing', 'video', 'work', 'cards');

const STYLE = `
html, body { margin: 0; background: transparent !important; overflow: hidden; }
body > :not(#cards) { display: none !important; }
#cards { position: fixed; inset: 0; z-index: 2147483647; }
.box { position: absolute; display: inline-block; box-sizing: border-box; text-align: center; font-family: var(--display, 'Lilita One'), sans-serif; color: #fff; line-height: 1.04;
  padding: 14px 38px 20px; background: rgba(20, 24, 52, 0.92); border: 7px solid #FFC93C; border-radius: 30px; box-shadow: 0 12px 0 rgba(0,0,0,0.35); text-shadow: 0 5px 0 rgba(0,0,0,0.35); }
.pill { display: inline-block; font-family: var(--body, 'Nunito'), sans-serif; font-weight: 900; color: #1B1F3B; background: #FFF9F0; border: 6px solid #1B1F3B; border-radius: 999px; padding: 8px 30px 10px; box-shadow: 0 8px 0 rgba(0,0,0,0.3); }
.logo svg { display: block; width: 100%; height: auto; filter: drop-shadow(0 10px 0 rgba(0,0,0,0.35)); }
.vig { position: absolute; inset: 0; background: radial-gradient(ellipse at 50% 42%, rgba(10,12,40,0.55) 0%, rgba(10,12,40,0.25) 55%, rgba(10,12,40,0) 80%); }
`;

/* each card: size and the HTML laid out on it (positions in px of the final frame) */
const CARDS = {
  // the title card, over the machine as the camera pulls back
  'title': [1920, 1080, B => `<div class="vig"></div>
    <div class="logo" style="position:absolute;left:470px;top:110px;width:980px">${B.wordmark()}</div>
    <div style="position:absolute;left:0;right:0;top:640px;text-align:center"><span class="pill" style="font-size:60px">Whole-class GCSE revision</span></div>
    <div id="pip" style="position:absolute;left:1430px;top:330px;width:420px;height:680px"></div>`],
  'answers': [1920, 1080, () => `<div class="box" style="left:60px;bottom:40px;width:1400px;font-size:68px;white-space:nowrap">Every student answers every question</div>`],
  'mark': [1920, 1080, () => `<div class="box" style="left:60px;bottom:40px;width:1400px;font-size:68px;white-space:nowrap">Mark the whole class in seconds</div>`],
  'ote': [1920, 1080, B => `<div class="logo" style="position:absolute;left:60px;bottom:40px;width:560px">${B.oteLogo()}</div>`],
  'outpace': [1920, 1080, B => `<div style="position:absolute;right:40px;top:30px;width:700px;text-align:center"><div class="logo">${B.opLogo()}</div>
    <span class="pill" style="font-size:32px;margin-top:10px;white-space:nowrap">Race home before the Hunter catches you</span></div>`],
  'cc': [1920, 1080, B => `<div class="logo" style="position:absolute;left:50px;top:8px;width:440px">${B.ccLogo()}</div>`],
  'hh': [1920, 1080, B => `<div class="logo" style="position:absolute;left:50px;top:12px;width:440px">${B.hhLogo()}</div>`],
  'launcher': [1920, 1080, () => `<div class="box" style="left:260px;right:260px;bottom:40px;font-size:66px">AQA &amp; Edexcel · Separate &amp; Combined<br><span style="color:#FFC93C">Biology · Chemistry · Physics</span></div>`],
  'results': [1920, 1080, () => `<div class="box" style="left:36px;top:520px;width:420px;font-size:76px">See exactly what to reteach</div>`],
  // the vertical reel (1080×1920): text between y 250 and 1470
  'v-answers': [1080, 1920, () => `<div class="box" style="left:70px;right:70px;top:300px;font-size:84px">Every student answers every question</div>`],
  'v-ote': [1080, 1920, B => `<div class="logo" style="position:absolute;left:240px;top:290px;width:600px">${B.oteLogo()}</div>`],
  'v-outpace': [1080, 1920, B => `<div class="logo" style="position:absolute;left:240px;top:290px;width:600px">${B.opLogo()}</div>`],
  'v-cc': [1080, 1920, B => `<div class="logo" style="position:absolute;left:260px;top:280px;width:560px">${B.ccLogo()}</div>`],
  'v-hh': [1080, 1920, B => `<div class="logo" style="position:absolute;left:260px;top:280px;width:560px">${B.hhLogo()}</div>`],
  'v-free': [1080, 1920, B => `<div class="vig"></div><div class="logo" style="position:absolute;left:120px;top:420px;width:840px">${B.wordmark()}</div>
    <div style="position:absolute;left:0;right:0;top:960px;text-align:center"><div class="box" style="position:relative;font-size:110px">Free on Tes</div><br>
    <span class="pill" style="font-size:46px;margin-top:34px">No logins · Works offline</span></div>`],
  // the end of each 15-second game clip
  'clip-end': [1920, 1080, B => `<div style="position:absolute;inset:0;background:rgba(14,16,44,0.78)"></div>
    <div class="logo" style="position:absolute;left:510px;top:170px;width:900px">${B.wordmark()}</div>
    <div style="position:absolute;left:0;right:0;top:700px;text-align:center"><span class="pill" style="font-size:56px">Whole-class GCSE revision · Free on Tes</span></div>`]
};

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const only = process.argv.slice(2);
  const browser = await chromium.launch();
  for (const [name, [w, h, html]] of Object.entries(CARDS)) {
    if (only.length && !only.includes(name)) continue;
    const page = await browser.newPage({ viewport: { width: w, height: h } });
    await page.goto('file://' + path.join(ROOT, 'test-build', 'editions', 'science.html'));
    await page.waitForFunction(() => window.CGB && CGB.brand && CGB.createHost2D);
    await page.addStyleTag({ content: STYLE });
    await page.evaluate(src => { const d = document.createElement('div'); d.id = 'cards'; d.innerHTML = (0, eval)('(' + src + ')')(CGB.brand); document.body.appendChild(d); }, html.toString());
    // the host, waving, on the title card
    await page.evaluate(() => { const p = document.getElementById('pip'); if (p) { const h = CGB.createHost2D(p, { className: 'host-launcher' }); h.gesture('wave', 60000); } });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(900);
    await page.screenshot({ path: path.join(OUT, name + '.png'), omitBackground: true });
    console.log('card', name);
    await page.close();
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
