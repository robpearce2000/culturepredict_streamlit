/*
 * Which question opens a game for each seed (no pictures drawn), so a shot can start on a short, clear question
 * from a popular topic:  node tools/video/demo/search-questions.js <game> <subject> <firstSeed> <count>
 */
'use strict';
const L = require('./lib.js');
(async () => {
  const [game, subject] = process.argv.slice(2, 4), s0 = +(process.argv[4] || 1), n = +(process.argv[5] || 10);
  const pre = { 'over-the-edge': 'ote', outpace: 'op', 'category-clash': 'cc', 'hex-hunt': 'hh' }[game];
  for (let seed = s0; seed < s0 + n; seed++) {
    const { browser, page } = await L.open({ seed });
    const T = L.timer(page);
    await page.evaluate(() => { window.__reelSkip = true; });
    await page.evaluate(([g, s]) => { CGB.bank.setBoard('aqa'); CGB.bank.setSubject(s); CGB.app.show(g); }, [game, subject]);
    await T.wait(300);
    await page.evaluate(p => { const b = document.querySelector(`#${p}-segTeams button[data-v="6"], #${p}-segGroups button[data-v="6"]`); if (b) b.click(); document.getElementById(p + '-startBtn').click(); }, pre);
    await T.until(g => { const s = CGB.games[g].state(); return s.q && (s.q.q || s.q.text); }, 60000, game);
    const q = await page.evaluate(g => { const s = CGB.games[g].state(); return s.q; }, game);
    console.log(seed, JSON.stringify(q).slice(0, 260));
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
