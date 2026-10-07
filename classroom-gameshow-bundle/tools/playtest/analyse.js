/* Summarises playtest runs into the report's tables. usage: node tools/playtest/analyse.js runsDir */
const fs = require('fs'), path = require('path');
const dir = process.argv[2];
const runs = fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')));
const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
const r1 = x => Math.round(x * 10) / 10, mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
const out = { timing: [], fairness: [], moments: [], errors: [], repeats: [], fps: [], crashes: [] };
for (const r of runs) {
  const L = r.lesson;
  if (r.crash) out.crashes.push(`${L.name}: ${r.crash.split('\n')[0]}`);
  const errs = (r.errors || []).filter(e => !/software WebGL|GL Driver Message|GPU stall|GroupMarkerNotSet/.test(e));
  if (errs.length) out.errors.push(`${L.name}: ${errs.slice(0, 5).join(' | ')}`);
  if (r.fps && r.fps.length) { const f = r.fps.slice().sort((a, b) => a - b); out.fps.push(`${L.name} (${L.quality || 'low'}, ${L.w || 1366}×${L.h || 768}): median ${f[f.length >> 1]} fps, lowest 5 s ${f[0]}`); }
  (r.games || []).forEach((g, gi) => {
    const q = g.questions || [];
    const w = q.map(x => x.writing), m = q.map(x => x.marking), sh = q.map(x => x.showAnim);
    // Over the Edge: the drop as a classroom laptop would show it (game clock, plus the captains' real time)
    let otherWaits = (g.waits || []).map(x => x.sim != null ? { what: x.what, s: Math.max(0, x.sim - x.humanPicks * (x.sim / Math.max(0.1, x.real))) } : x);
    const longest = otherWaits.reduce((b, x) => x.s > (b ? b.s : -1) ? x : b, null);
    // estimated length on a normal-speed machine for Over the Edge: real time minus the slowed-down drop time
    let total = g.total || 0, est = total;
    // a game that restarted itself or hit the time limit: it ended when its last question was marked
    if ((total > 1800 || g.restartedAt) && q.length) { total = q[q.length - 1].tConfirm - g.tStart + 10; est = total; }
    if (g.game === 'over-the-edge') {
      const slow = (g.waits || []).filter(x => x.sim != null).reduce((a, x) => a + Math.max(0, (x.real - x.humanPicks) - Math.max(0, x.sim - x.humanPicks * (x.sim / Math.max(0.1, x.real)))), 0);
      est = total - slow;
    }
    out.timing.push({ lesson: L.name, game: g.game, teams: g.teams, subject: `${L.subject} ${L.board || ''}`.trim(), maths: g.mathsTime || '', mode: L.mode || '', questions: q.length,
      total: mmss(total), est: g.game === 'over-the-edge' ? mmss(est) : '', perQ: q.length ? r1((est) / q.length) : '', writing: r1(avg(w)), show: r1(avg(sh)), marking: r1(avg(m)),
      longestWait: longest ? `${longest.what} ${r1(longest.s)} s` : '', ended: g.final && g.final.phase });
    // fairness
    const f = g.final || {};
    let spread = '', note = '';
    if (g.game === 'over-the-edge' && f.money) { spread = `£${Math.max(...f.money) - Math.min(...f.money)}`; note = `money ${f.money.join(' / ')}; class pot £${f.team}`; }
    if (g.game === 'category-clash' && f.scores) { spread = Math.max(...f.scores) - Math.min(...f.scores); note = `scores ${f.scores.join(' / ')}`; }
    if (g.game === 'hex-hunt' && f.owners) { note = `winner ${f.winner === 0 ? 'half 1' : f.winner === 1 ? 'half 2' : 'none'}; hexes ${f.owners.filter(o => o === 0).length}–${f.owners.filter(o => o === 1).length}`; }
    if (g.game === 'outpace') { note = `pot ${f.pot}; sprint ${f.net}/${f.target}`; }
    // when was it decided? the last question at which the leader changed
    let lastChange = '', leader = null;
    (g.scores || []).forEach((s, i) => { if (!s || g.game === 'outpace') return; const best = s.indexOf(Math.max(...s)); if (best !== leader) { leader = best; lastChange = i + 1; } });
    out.fairness.push({ lesson: L.name, game: g.game, teams: g.teams, spread, note, lastLeadChange: lastChange ? `${lastChange} of ${q.length}` : '' });
    (g.moments || []).forEach(mo => out.moments.push(`${L.name}: ${JSON.stringify(mo)}`));
    const ids = (g.qids || []).filter(Boolean), dup = ids.length - new Set(ids).size;
    out.repeats.push({ lesson: L.name, game: g.game, asked: ids.length, repeatsInGame: dup, ids });
  });
}
console.log(JSON.stringify(out, null, 1));
