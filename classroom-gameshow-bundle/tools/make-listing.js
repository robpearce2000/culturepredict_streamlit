#!/usr/bin/env node
/*
 * Makes the Tes listing images from the built bundle (run `node build.js` first):
 *   dist/listing/cover.png                1600×1000  wordmark, mascot and both games
 *   dist/listing/over-the-edge.png        1920×1080  gameplay screenshot
 *   dist/listing/outpace.png              1920×1080  gameplay screenshot
 *   dist/listing/question-bank.png        1920×1080  the shared question bank
 */
'use strict';
const path = require('path');
const fs = require('fs');
const { chromium } = require('@playwright/test');

const ROOT = path.join(__dirname, '..');
const URL = 'file://' + path.join(ROOT, 'dist', 'classroom-gameshow-bundle.html');
const OUT = path.join(ROOT, 'dist', 'listing');
const ARGS = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];

async function until(page, fn, ms) {
  const end = Date.now() + (ms || 60000);
  while (Date.now() < end) { if (await page.evaluate(fn)) return; await page.waitForTimeout(150); }
  throw new Error('timed out waiting');
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ARGS });
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(URL);
  await page.waitForFunction(() => window.CGB && CGB.app);
  await page.evaluate(() => { CGB.store.setJSON('names', ['Amira', 'Josh']); });
  await page.reload();
  await page.waitForFunction(() => window.CGB && CGB.app);
  await page.waitForTimeout(1500);

  // Mascot on its own (transparent) for the cover
  await page.evaluate(() => { document.getElementById('mascotHello').style.display = 'none'; });
  await page.waitForTimeout(2500);
  const mascot = await page.locator('#mascot').screenshot({ omitBackground: true });
  fs.writeFileSync(path.join(ROOT, 'docs', 'mascot.png'), mascot);   // used by the teacher guide
  await page.evaluate(() => { document.getElementById('mascotHello').style.display = ''; });

  // Question bank, with one of the teacher's own sets added
  await page.evaluate(() => {
    CGB.bank.addSet('Year 10 bonding quiz', 'Subject: Chemistry\nTopic: Bonding\nQ: What type of bonding is in sodium chloride?\nA: Ionic\nQ: What type of bonding is in methane?\nA: Covalent\nQ: Why can metals conduct electricity?\nA: They have delocalised electrons');
    CGB.bank.setActive('gcse-combined-mix');
  });
  await page.click('#openBank');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, 'question-bank.png') });
  await page.keyboard.press('Escape');

  // Over the Edge: a counter has just been won and the lanes are lit
  await page.click('[data-play="over-the-edge"]');
  await page.waitForTimeout(1500);
  await page.click('#ote-startBtn');
  await page.waitForTimeout(3500);
  await page.keyboard.press('c');
  await page.waitForTimeout(1800);
  await page.screenshot({ path: path.join(OUT, 'over-the-edge.png') });
  const oteShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.keyboard.press('2');
  await page.waitForTimeout(400);
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Outpace: Deal Round question with the answer showing
  await page.click('[data-play="outpace"]');
  await page.waitForTimeout(1000);
  await page.click('#op-startBtn');
  await page.keyboard.press('2');
  await page.waitForTimeout(1200);
  await page.keyboard.press('a'); await page.keyboard.press('c');
  await until(page, () => { const s = CGB.games.outpace.state(); return !s.awaitingNext && !s.answerShown; });
  await page.keyboard.press('a');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, 'outpace.png') });
  const opShot = await page.screenshot({ type: 'jpeg', quality: 88 });
  await page.keyboard.press('Escape'); await page.click('#leaveConfirm');

  // Cover: built in the same page so it uses the bundle's own fonts and artwork
  await page.setViewportSize({ width: 1600, height: 1000 });
  const b64 = buf => 'data:image/' + (buf[0] === 0x89 ? 'png' : 'jpeg') + ';base64,' + buf.toString('base64');
  await page.evaluate(({ ote, op, mascot }) => {
    document.body.innerHTML = `
      <div id="cover">
        <svg class="rays" viewBox="0 0 100 100" preserveAspectRatio="none"><g fill="rgba(255,255,255,0.05)"><path d="M50 -10 L30 110 L40 110 Z"/><path d="M50 -10 L55 110 L68 110 Z"/><path d="M50 -10 L80 110 L95 110 Z"/><path d="M50 -10 L5 110 L15 110 Z"/></g></svg>
        <div class="wm">${CGB.brand.wordmark()}</div>
        <img class="mascot" src="${mascot}" alt="">
        <div class="shots">
          <figure class="s1"><img src="${ote}" alt=""><figcaption>${CGB.brand.oteLogo()}</figcaption></figure>
          <figure class="s2"><img src="${op}" alt=""><figcaption>${CGB.brand.opLogo()}</figcaption></figure>
        </div>
        <div class="chips"><span>2 classroom games</span><span>1 shared question bank</span><span>190+ AQA-style GCSE Combined Science questions</span><span>Works offline</span></div>
      </div>`;
    const st = document.createElement('style');
    st.textContent = `
      html, body { overflow: hidden; background: #141834; }
      #cover { position: fixed; inset: 0; background: radial-gradient(ellipse at 50% -10%, #3A3F86 0%, #1F2550 40%, #141834 80%); font-family: var(--body); }
      #cover .rays { position: absolute; inset: 0; width: 100%; height: 100%; }
      #cover .wm { position: absolute; left: 50%; top: 18px; transform: translateX(-50%); width: 820px; }
      #cover .wm svg { width: 100%; height: auto; display: block; }
      #cover .mascot { position: absolute; right: 30px; top: 40px; height: 330px; }
      #cover .shots { position: absolute; left: 0; right: 0; top: 400px; display: flex; justify-content: center; gap: 46px; }
      #cover figure { margin: 0; width: 700px; position: relative; }
      #cover figure img { width: 100%; display: block; border: 6px solid #1B1F3B; border-radius: 20px; box-shadow: 0 12px 0 rgba(0,0,0,0.35); }
      #cover .s1 { transform: rotate(-2.5deg); } #cover .s2 { transform: rotate(2.5deg); }
      #cover figcaption { position: absolute; left: 50%; bottom: -70px; transform: translateX(-50%); width: 330px; }
      #cover figcaption svg { width: 100%; height: auto; display: block; filter: drop-shadow(0 6px 0 rgba(0,0,0,0.35)); }
      #cover .chips { position: absolute; left: 0; right: 0; bottom: 26px; display: flex; justify-content: center; gap: 14px; }
      #cover .chips span { background: #FFF9F0; color: #1B1F3B; border: 4px solid #1B1F3B; border-radius: 999px; padding: 8px 18px; font-weight: 900; font-size: 22px; }`;
    document.head.appendChild(st);
  }, { ote: b64(oteShot), op: b64(opShot), mascot: b64(mascot) });
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(OUT, 'cover.png') });
  await browser.close();
  console.log('Listing images written to dist/listing/');
})().catch(e => { console.error(e); process.exit(1); });
