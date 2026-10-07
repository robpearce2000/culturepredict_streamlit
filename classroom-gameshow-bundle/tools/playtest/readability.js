/* Screens at 1920×1080 and 1366×768, normal and full screen, each also shrunk to a quarter of its
   size (roughly the view from the back of a classroom). usage: node tools/playtest/readability.js */
const { chromium } = require('@playwright/test');
const path = require('path'), fs = require('fs');
const DIST = 'file://' + path.join(__dirname, '..', '..', 'dist', 'showtime-classroom-gameshows.html');
const OUT = path.join(__dirname, '..', '..', 'playtest-screenshots', 'readability');
fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch();
  for (const [w, h] of [[1920, 1080], [1366, 768]]) for (const fsMode of [false, true]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h } });
    const page = await ctx.newPage();
    await page.addInitScript(() => { try { localStorage.setItem('cgb.settings', JSON.stringify({ quality: 'low' })); } catch (e) {} });
    await page.goto(DIST); await page.waitForFunction(() => window.CGB && CGB.app); await sleep(1500);
    await page.selectOption('#subjectSelect', 'biology');
    const tag = `${w}x${h}${fsMode ? '-fullscreen' : ''}`;
    const snap = async name => { const f = path.join(OUT, `${tag}-${name}.png`); await page.screenshot({ path: f }); return f; };
    const shots = [];
    if (fsMode) { await page.keyboard.press('f'); await sleep(800); }
    for (const [g, p] of [['over-the-edge', 'ote'], ['outpace', 'op'], ['category-clash', 'cc'], ['hex-hunt', 'hh']]) {
      await page.click(`[data-play="${g}"]`); await sleep(1500);
      await page.click(`#${p}-startBtn`); await sleep(1200);
      await page.keyboard.press('Enter'); await sleep(2500);                    // Start game card
      if (g === 'outpace') { await page.keyboard.press('2'); await sleep(3500); }
      if (g === 'category-clash') { await page.locator('#cc-board .cc-tile[data-c]').first().click(); await sleep(2500); }
      if (g === 'hex-hunt') { shots.push(await snap('hh-board')); await page.locator('#hh-board .hh-hex').nth(5).click(); await sleep(2500); }
      if (g === 'category-clash') shots.push(await snap('cc-question')); else shots.push(await snap(p + '-question'));
      // mark: show me, then teams 1 and 3 right
      await page.keyboard.press('Space'); await sleep(4500);
      if (g === 'hex-hunt') await page.keyboard.press('1'); else { await page.keyboard.press('1'); await page.keyboard.press('3'); await page.keyboard.press('Enter'); }
      await sleep(3500);
      shots.push(await snap(p + '-marked'));
      await page.click(`#${p}-menuBtn`); await sleep(800);   // (in full screen, Esc first leaves full screen)
      if (await page.locator('#leaveModal:not([hidden])').count()) await page.click('#leaveConfirm');
      await sleep(1000);
    }
    // the quarter-size versions
    for (const f of shots) {
      const img = 'data:image/png;base64,' + fs.readFileSync(f).toString('base64');
      const p2 = await ctx.newPage();
      await p2.setViewportSize({ width: Math.round(w / 2), height: Math.round(h / 2) });
      await p2.setContent(`<body style="margin:0;background:#000"><img src="${img}" style="width:${Math.round(w / 2)}px;height:${Math.round(h / 2)}px;image-rendering:auto"></body>`);
      await p2.screenshot({ path: f.replace('.png', '-quarter.png') });
      await p2.close();
    }
    await ctx.close();
  }
  await browser.close();
  console.log('done');
})();
