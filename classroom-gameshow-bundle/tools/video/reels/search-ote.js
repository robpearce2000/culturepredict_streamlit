/*
 * Best-take search for the Over the Edge reels: replays seeded games (physics only, no pictures) and scores every
 * moment a counter falls off the edge: how long something hung over the edge first (the teeter) and how many
 * counters followed it within 1.5 s (the cascade). Prints the best takes as JSON lines.
 *   node tools/video/reels/search-ote.js <firstSeed> <count> [questions]
 */
'use strict';
const { open } = require('./reel.js');
const ote = require('./ote-lib.js');
(async () => {
  const s0 = +(process.argv[2] || 1), count = +(process.argv[3] || 5), Q = +(process.argv[4] || 6);
  for (let seed = s0; seed < s0 + count; seed++) {
    const { browser, page } = await open({ seed, quality: 'high', css: ote.STAGE_CSS });
      const T = ote.timer(page, true);
    await ote.setup(page, T);
    const frames = [];
    await ote.play(page, T, Q, async n => { const f = await ote.advance(page, n); frames.push(...f); return f; });
    // falls: the lower-tray count goes down (a drop wave adds counters, so only decreases count)
    const takes = [];
    for (let i = 1; i < frames.length; i++) {
      const lost = frames[i - 1].n - frames[i].n;
      if (lost <= 0) continue;
      // the counter that fell: in the edge band before, gone after; follow it back frame by frame
      const gone = frames[i - 1].edge.filter(([x, y]) => !frames[i].edge.some(([x2, y2]) => Math.abs(x2 - x) < 3 && Math.abs(y2 - y) < 3));
      if (!gone.length) continue;
      let [x, y, jk] = gone.sort((a, b) => b[1] - a[1])[0];
      const r = jk ? 34 : 18;
      let t = 0;
      for (let j = i - 1; j >= 0; j--) {
        const m = frames[j].edge.find(([x2, y2]) => Math.abs(x2 - x) < 4 && Math.abs(y2 - y) < 4);
        if (!m) break; x = m[0]; y = m[1];
        if (y > 400 - r * 0.15) t++; else if (y < 400 - r * 0.6) break;
      }
      let casc = 0; for (let j = i; j < Math.min(frames.length, i + 90); j++) casc += Math.max(0, frames[j - 1].n - frames[j].n);
      takes.push({ seed, frame: i, teeter: t, cascade: casc, x: Math.round(x), jackpot: !!jk });
      i += 60;
    }
    takes.sort((a, b) => (Math.min(b.teeter, 240) / 40 + b.cascade) - (Math.min(a.teeter, 240) / 40 + a.cascade));
    takes.slice(0, 3).forEach(t => console.log(JSON.stringify(t)));
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
