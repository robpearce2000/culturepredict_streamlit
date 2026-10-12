/*
 * Demo shot "launcher": the Science mega pack's launcher, set up the way a teacher does it. An on-screen pointer
 * (recording only) chooses Chemistry, Edexcel and Combined Science, ticks four topics, then plays Category Clash
 * and starts it from its setup card. Dropdown choices are made with real select changes (a native dropdown list
 * never appears in a capture), so the pointer clicks and the box changes.
 *   node tools/video/demo/shot-launcher.js
 */
'use strict';
const L = require('./lib.js');

(async () => {
  const { browser, page } = await L.open({ seed: 1 });
  const T = L.timer(page);
  await page.evaluate(() => { CGB.bank.setBoard('aqa'); CGB.bank.setSubject('biology'); });
  await T.wait(1500);
  // the pointer: an arrow and a ring that pulses on each click
  await page.evaluate(() => {
    const p = document.createElement('div'); p.id = 'demo-ptr';
    p.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;width:0;height:0';
    p.innerHTML = '<div class="ring" style="position:absolute;left:-26px;top:-26px;width:52px;height:52px;border-radius:50%;border:5px solid #FFC93C;opacity:0"></div>' +
      '<svg width="46" height="56" viewBox="0 0 23 28" style="position:absolute;left:-3px;top:-2px;filter:drop-shadow(0 3px 3px rgba(0,0,0,.45))"><path d="M2 2 L2 23 L7.5 18 L11.5 26.5 L15 25 L11 16.8 L18.5 16.8 Z" fill="#fff" stroke="#1B1F3B" stroke-width="2" stroke-linejoin="round"/></svg>';
    document.body.appendChild(p);
    window.__ptr = (x, y, ring) => { p.style.transform = `translate(${x}px, ${y}px)`; const r = p.firstChild; r.style.opacity = ring > 0 ? (1 - ring).toFixed(3) : 0; r.style.transform = `scale(${(0.5 + ring).toFixed(3)})`; };
  });
  await page.evaluate(() => { window.__reelSkip = false; });
  const s = new L.Shot(page, 'launcher'); await s.start();
  let x = 1990, y = 620, ring = -1;
  const center = sel => page.evaluate(q => { const e = document.querySelector(q); const r = e.getBoundingClientRect(); return [r.left + Math.min(r.width * 0.5, 150), r.top + r.height * 0.55]; }, sel);
  const draw = () => page.evaluate(([x, y, r]) => window.__ptr(x, y, r), [x, y, ring]);
  const frames = async (sec, fn) => { await s.run(sec, async i => { if (fn) fn(i, Math.round(sec * L.FPS)); if (ring >= 0) ring = ring + 1 / 14 > 1 ? -1 : ring + 1 / 14; await draw(); }); };
  const move = async (sel, sec) => {
    const [tx, ty] = await center(sel), x0 = x, y0 = y;
    await frames(sec, (i, n) => { const f = (i + 1) / n, e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2; x = x0 + (tx - x0) * e; y = y0 + (ty - y0) * e; });
  };
  const click = async (sel, act) => { ring = 0; if (act) await act(); else await page.evaluate(q => document.querySelector(q).click(), sel); };
  await draw();
  s.cue('start'); await frames(0.3);
  await move('#subjectSelect', 0.45); s.cue('subject'); await click('#subjectSelect', () => page.selectOption('#subjectSelect', 'chemistry')); await frames(0.3);
  await move('#boardSelect', 0.3); s.cue('board'); await click('#boardSelect', () => page.selectOption('#boardSelect', 'edexcel')); await frames(0.3);
  await move('#courseSelect', 0.3); s.cue('course'); await click('#courseSelect', () => page.selectOption('#courseSelect', 'combined')); await frames(0.3);
  await move('#packBtn', 0.3); s.cue('topics'); await click('#packBtn'); await frames(0.3);
  for (const k of [1, 2, 3, 4]) {          // four topics: Category Clash's four categories
    const sel = await page.evaluate(k => { const l = [...document.querySelectorAll('#packMenu label.pk-opt:not(.mixed)')][k - 1]; l.id = 'demo-pk' + k; return '#demo-pk' + k; }, k);
    await move(sel, k === 1 ? 0.28 : 0.16); await click(sel, () => page.evaluate(id => document.querySelector(id + ' input').click(), sel)); await frames(0.14);
  }
  await move('#packMenu [data-done]', 0.3); await click('#packMenu [data-done]'); await frames(0.2);
  await move('[data-play="category-clash"]', 0.4); s.cue('play'); await click('[data-play="category-clash"]'); await frames(0.4);
  await move('#cc-startBtn', 0.35); s.cue('startgame'); await click('#cc-startBtn'); await frames(0.25);
  await page.evaluate(() => { document.getElementById('demo-ptr').style.display = 'none'; });
  s.cue('game'); await frames(1.0);
  s.cue('end');
  await s.done({});
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
