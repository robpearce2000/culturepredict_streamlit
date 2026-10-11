/*
 * Best-take search for the jackpot reel: the final round with the gold counter put close to the edge; finds when
 * it falls and how long it hung over the edge first.   node tools/video/reels/search-jackpot.js <seed> <count>
 */
'use strict';
const { open } = require('./reel.js');
const ote = require('./ote-lib.js');
(async () => {
  const s0 = +(process.argv[2] || 1), count = +(process.argv[3] || 4);
  for (let seed = s0; seed < s0 + count; seed++) for (const near of [0.6, 0.35]) {
    const { browser, page } = await open({ seed, quality: 'high', css: ote.STAGE_CSS });
    const T = ote.timer(page, true);
    await ote.setup(page, T);
    const frames = [];
    await ote.playFinal(page, T, near, 3, async n => { const f = await ote.advance(page, n); frames.push(...f); return f; });
    const fell = frames.findIndex(f => f.jf);
    let hang = 0;
    if (fell > 0) for (let j = fell - 1; j >= 0; j--) { const b = frames[j].edge.find(e => e[2]); if (b && b[1] > 400 - 34 * 0.6) hang++; else if (!b || b[1] < 400 - 34) break; }
    console.log(JSON.stringify({ seed, near, fell, hang, total: frames.length }));
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
