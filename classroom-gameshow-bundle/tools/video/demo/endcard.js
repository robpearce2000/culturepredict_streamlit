/*
 * The demo's end card (recording only), 5 s at 60 fps on the controlled clock: the Showtime wordmark, a still
 * from each of the four games, "Try it free on Tes", "No logins · Works offline", and the host waving. Drawn by
 * the app over the launcher's own background, in its fonts and colours. Also the YouTube thumbnail.
 *   THUMB_ONLY=1 node tools/video/demo/endcard.js  redraws only thumbnail.png
 *   node tools/video/demo/endcard.js            ->  marketing/video/work/endcard/ (frames) and marketing/video/thumbnail.png
 * The stills are frames of the rendered shots (see STILLS), so render the shots first.
 */
'use strict';
const path = require('path');
const fs = require('fs');
const L = require('./lib.js');

const W = L.WORK;
const cue = (shot, k, off) => { const r = JSON.parse(fs.readFileSync(path.join(W, shot, 'sounds.json'))); return Math.max(0, Math.min(r.frames - 1, r.cues[k] + (off || 0))); };
const frame = (shot, n) => path.join(W, shot, String(n).padStart(5, '0') + '.jpg');
const STILLS = () => [
  ['Over the Edge', frame('jackpot', cue('jackpot', 'fall', -30))],
  ['Outpace', frame('outpace', cue('outpace', 'move1', 60))],
  ['Category Clash', frame('cc', cue('cc', 'pick', -10))],
  ['Hex Hunt', frame('hh', cue('hh', 'win', 120))]
];
const data = f => 'data:image/jpeg;base64,' + fs.readFileSync(f).toString('base64');

const CSS = `
#end { position: fixed; inset: 0; z-index: 2147483000; overflow: hidden; }
#end .t { position: absolute; left: 0; right: 0; text-align: center; }
#end .wm { position: absolute; left: 330px; top: 26px; width: 760px; }
#end .wm svg, #end .lg svg { display: block; width: 100%; height: auto; filter: drop-shadow(0 10px 0 rgba(0,0,0,0.35)); }
#end .row { position: absolute; left: 70px; top: 448px; width: 1280px; display: flex; gap: 24px; }
#end .g { flex: 1; text-align: center; }
#end .g .im { height: 172px; border-radius: 18px; border: 6px solid #FFC93C; background-size: cover; background-position: center; box-shadow: 0 10px 0 rgba(0,0,0,0.35); }
#end .g b { display: block; margin-top: 12px; font-family: var(--display, 'Lilita One'), sans-serif; font-weight: 400; font-size: 36px; color: #fff; text-shadow: 0 4px 0 rgba(0,0,0,0.35); }
#end .cta { position: absolute; left: 70px; width: 1280px; top: 728px; text-align: center; }
#end .box { display: inline-block; font-family: var(--display, 'Lilita One'), sans-serif; color: #fff; font-size: 104px; line-height: 1; padding: 16px 52px 24px;
  background: rgba(20,24,52,0.92); border: 8px solid #FFC93C; border-radius: 36px; box-shadow: 0 14px 0 rgba(0,0,0,0.35); text-shadow: 0 6px 0 rgba(0,0,0,0.35); }
#end .pill { display: inline-block; margin-top: 26px; font-family: var(--body, 'Nunito'), sans-serif; font-weight: 900; font-size: 48px; color: #1B1F3B; background: #FFF9F0;
  border: 6px solid #1B1F3B; border-radius: 999px; padding: 8px 36px 12px; box-shadow: 0 8px 0 rgba(0,0,0,0.3); }
#end #pip { position: absolute; left: 1430px; top: 250px; width: 440px; height: 800px; }
#end .in { opacity: 0; }
`;

(async () => {
  const stills = STILLS().map(([n, f]) => [n, data(f)]);
  const { browser, page } = await L.open({ seed: 1 });
  const T = L.timer(page);
  await T.wait(800);
  // the launcher's background stays; everything on it is replaced by the end card
  await page.addStyleTag({ content: CSS + '#launcher > *, .launcher > * { visibility: hidden !important; } #mascot { visibility: hidden !important; }' });
  await page.evaluate(stills => {
    const e = document.createElement('div'); e.id = 'end';
    e.innerHTML = `<div class="wm in" data-at="0">${CGB.brand.wordmark()}</div>
      <div class="row">${stills.map(([n, d], i) => `<div class="g in" data-at="${0.25 + i * 0.12}"><div class="im" style="background-image:url('${d}')"></div><b>${n}</b></div>`).join('')}</div>
      <div class="cta"><div class="box in" data-at="0.9">Try it free on Tes</div><br><div class="pill in" data-at="1.2">No logins · Works offline</div></div>
      <div id="pip" class="in" data-at="0.1"></div>`;
    document.body.appendChild(e);
    const host = CGB.createHost2D(document.getElementById('pip'), { className: 'host-launcher' });
    window.__endHost = host;
    // each part pops in at its time, on the page's (controlled) clock
    const t0 = performance.now();
    const tick = () => {
      const t = (performance.now() - t0) / 1000;
      e.querySelectorAll('.in').forEach(el => {
        const k = Math.max(0, Math.min(1, (t - +el.dataset.at) / 0.35)), b = 1 - Math.pow(1 - k, 3);
        el.style.opacity = Math.min(1, k * 2.5);
        el.style.transform = `scale(${(0.8 + 0.2 * b + 0.06 * Math.sin(k * Math.PI)).toFixed(4)})`;
      });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, stills);
  await page.evaluate(() => document.fonts.ready);
  if (!process.env.THUMB_ONLY) {
  const s = new L.Shot(page, 'endcard'); await s.start();
  await s.run(0.4);
  await page.evaluate(() => { window.__endHost.gesture('wave', 2600); window.__endHost.talk(1200); });
  await s.run(2.8);
  await page.evaluate(() => { window.__endHost.gesture('present', 2000); });
  await s.run(1.8);
  await s.done({});
  }
  // the YouTube thumbnail (1280×720): the most dramatic Over the Edge frame, big words, the host
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.evaluate(img => {
    document.getElementById('end').remove();
    const e = document.createElement('div'); e.id = 'end';
    e.innerHTML = `<div style="position:absolute;inset:0;background:url('${img}') center/cover"></div>
      <div style="position:absolute;inset:0;background:linear-gradient(90deg, rgba(14,16,44,0.9) 0%, rgba(14,16,44,0.65) 45%, rgba(14,16,44,0) 70%)"></div>
      <div style="position:absolute;left:44px;top:40px;width:760px;font-family:var(--display,'Lilita One'),sans-serif;color:#fff;font-size:150px;line-height:0.92;text-shadow:0 8px 0 rgba(0,0,0,0.5);-webkit-text-stroke:6px #1B1F3B;paint-order:stroke fill">4 GAME<br><span style="color:#FFC93C">SHOWS</span></div>
      <div style="position:absolute;left:52px;top:388px"><span class="box" style="font-size:84px;padding:10px 36px 16px">GCSE Revision</span></div>
      <div style="position:absolute;left:52px;top:536px;width:520px"><div class="wm" style="position:static;width:330px">${CGB.brand.wordmark()}</div></div>
      <div id="pip2" style="position:absolute;left:930px;top:150px;width:330px;height:600px"></div>`;
    document.body.appendChild(e);
    const h = CGB.createHost2D(document.getElementById('pip2'), { className: 'host-launcher' }); h.gesture('wave', 60000);
  }, data(frame('jackpot', cue('jackpot', 'fall', -20))));
  await T.wait(900);
  await L.still(page, path.join(L.ROOT, 'marketing', 'video', 'thumbnail.png'));
  console.log('thumbnail');
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
