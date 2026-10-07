/* Runs the playtest plan: at most one 3D game (Over the Edge, Outpace) at a time, plus two 2D lessons.
   Lessons sharing a "profile" run one after another in the same browser profile (a week of lessons).
   usage: node tools/playtest/run-all.js plan.json outDir */
const { spawn } = require('child_process'), fs = require('fs'), path = require('path');
const plan = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), OUT = process.argv[3];
fs.mkdirSync(OUT, { recursive: true });
const is3D = l => l.games.some(g => g.id === 'over-the-edge' || g.id === 'outpace');
const done = new Set(fs.readdirSync(OUT).filter(f => f.endsWith('.json')).map(f => f.slice(0, -5)));
// a week of lessons is one job
const jobs = []; const weeks = {};
plan.filter(l => !done.has(l.name)).forEach(l => {
  if (l.profile) { l.profile = path.join(OUT, 'profile-' + l.profile); if (!weeks[l.profile]) jobs.push(weeks[l.profile] = { list: [] }); weeks[l.profile].list.push(l); }
  else jobs.push({ list: [l] });
});
let running3D = 0, running = 0;
const runOne = l => new Promise(res => {
  const t = Date.now();
  const c = spawn('node', [path.join(__dirname, 'lesson.js'), JSON.stringify(l), path.join(OUT, l.name + '.json')], { stdio: ['ignore', 'ignore', 'pipe'] });
  let err = ''; c.stderr.on('data', d => { err += d; });
  c.on('exit', code => { console.log(new Date().toISOString().slice(11, 19), l.name, 'exit', code, ((Date.now() - t) / 60000).toFixed(1) + ' min', err.slice(0, 200)); res(); });
});
async function runJob(j) { for (const l of j.list) await runOne(l); }
(async () => {
  const queue = jobs.slice();
  await new Promise(resolve => {
    const tick = () => {
      if (!queue.length && !running) return resolve();
      for (let i = 0; i < queue.length && running < 3; i++) {
        const j = queue[i], d3 = is3D(j.list[0]);
        if (d3 && running3D >= (+process.env.MAX3D || 1)) continue;
        queue.splice(i--, 1); running++; if (d3) running3D++;
        runJob(j).then(() => { running--; if (d3) running3D--; tick(); });
      }
    };
    tick();
  });
  console.log('ALL DONE');
})();
