#!/usr/bin/env node
/*
 * Built-in question packs: one plain-text file per specification topic, in packs/<board>/<subject>/.
 *
 *   node tools/packs.js check        check every pack and print a report (exit 1 on any error)
 *   node tools/packs.js ids [path]   give every question without an Id: line a stable id (optionally only
 *                                    in files under a path, e.g. packs/aqa/biology)
 *   node tools/packs.js report       the per-pack table used in QUESTION-BANK-PROGRESS.md
 *
 * Pack file format (keys are case-insensitive; # starts a comment line):
 *
 *   Board: AQA                      header: the exam board (AQA or Edexcel)
 *   Spec: 8461                      the specification code (packs/specs/<board>-<code>.json)
 *   Topic: 4.1                      the specification topic this pack covers
 *
 *   Q: Which organelle is the site of aerobic respiration?
 *   A: Mitochondria                 the expected answer: short, at most about eight words
 *   Accept: Mitochondrion           other accepted answers, separated by ";"   (optional)
 *   Ref: 4.1.1.2                    the specification point, in the board's own numbering
 *   D: 1                            difficulty 1, 2 or 3 (see TIERS in src/shared/bank.js)
 *   HT: yes                         Higher tier only                            (optional)
 *   RP: 1                           a required practical's number               (optional)
 *   Calc: yes                       the question needs a calculation: 45 s to answer outside Maths (optional)
 *   Sep: yes                        separate science only, on a point that is otherwise in Combined Science;
 *                                   "Sep: no" keeps it in Combined Science (needed on the points the course
 *                                   map in tools/spec-map lists as "partial")
 *   Note: Not "mitochondrion's"     a note for the teacher, e.g. a misconception (optional)
 *   Id: aqa-bio-4.1-001             stable id, added by "node tools/packs.js ids"
 *
 * The subtopic is the specification heading three levels down from the Ref (e.g. 4.1.1 Cell
 * structure), and hexOk comes from the shared Hex Hunt letter rule (src/shared/hexletter.js),
 * so neither is typed by hand. Science questions also take their course tags from their Ref:
 * tools/spec-map/<board>-<subject>.json says, point by point, whether it is in Combined Science,
 * its Combined Science reference and its tier in each course.
 *
 * The same module is used by build.js (to put the packs into the single HTML file) and by
 * tests/packs.spec.js (the automated checks).
 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const hexLetter = require('../src/shared/hexletter.js');

const ROOT = path.join(__dirname, '..', 'packs');
const SPEC_DIR = path.join(ROOT, 'specs');
const BOARDS = { aqa: 'AQA', edexcel: 'Edexcel' };
const SUBJECT_IDS = { Biology: 'biology', Chemistry: 'chemistry', Physics: 'physics', Mathematics: 'maths', Maths: 'maths', History: 'history', Geography: 'geography', 'Geography A': 'geography', 'Geography B': 'geography' };
const SHORT = { biology: 'bio', chemistry: 'chem', physics: 'phys', maths: 'maths', history: 'hist', geography: 'geog' };

/* Limits from the brief: a mini-whiteboard answer in 20 seconds, marked at a glance */
const LIMITS = {
  answerWords: 10, answerWordsSoft: 8, answerChars: 80, questionChars: 220, acceptWords: 12,
  minPerTopic: 30, perLeaf: 3,          // at least 30 per topic, and about three per specification point
  minLeaf: 2,                           // every point is asked about at least twice
  minShare: 0.2,                        // each difficulty is at least a fifth of the topic
  hexShare: 1 / 3,                      // at least a third of each topic can go on a Hex Hunt board
  minRP: 2                              // each required practical has at least two questions
};

/* Names as shown in the games: without the specification's "(biology only)", "(HT only)" and
   similar labels (every pack is for the separate sciences, and Higher tier is a field of its own) */
const displayName = t => String(t).replace(/\s*\((?:[a-z]+ only|HT only|common content with [a-z]+)\)/gi, '').trim();
const refKey = r => r.split('.').map(Number);
const cmpRef = (a, b) => { const x = refKey(a), y = refKey(b); for (let i = 0; i < Math.max(x.length, y.length); i++) { const d = (x[i] === undefined ? -1 : x[i]) - (y[i] === undefined ? -1 : y[i]); if (d) return d; } return 0; };

// packs follow the specification's own topic order (refs such as 1AA or B1 are not numbers)
const topicIndex = p => loadSpec(p.board, p.specCode).topics.findIndex(t => t.ref === p.topicRef);

/* The course map for a separate science (tools/spec-map), or null for other subjects */
const MAP_DIR = path.join(__dirname, 'spec-map');
const mapCache = {};
function loadMap(board, subject) {
  const k = board.toLowerCase() + '-' + subject;
  if (!(k in mapCache)) { const f = path.join(MAP_DIR, k + '.json'); mapCache[k] = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null; }
  return mapCache[k];
}
/* A Ref's entry in the course map: the point itself or its nearest heading, with the
   Combined Science reference carried down (AQA renumbers 4.x as 4.x, 5.x or 6.x) */
function mapPoint(map, ref) {
  for (let r = ref; r; r = r.includes('.') ? r.slice(0, r.lastIndexOf('.')) : '') {
    const e = map.points[r];
    if (e) return Object.assign({}, e, { at: r, combRef: e.comb ? e.comb + ref.slice(r.length) : null });
  }
  return null;
}

const specCache = {};
function loadSpec(board, code) {
  const k = board.toLowerCase() + '-' + code;
  if (!specCache[k]) {
    const f = path.join(SPEC_DIR, k + '.json');
    if (!fs.existsSync(f)) return null;
    specCache[k] = JSON.parse(fs.readFileSync(f, 'utf8'));
  }
  return specCache[k];
}
function allSpecs() {
  if (!fs.existsSync(SPEC_DIR)) return [];
  return fs.readdirSync(SPEC_DIR).filter(f => f.endsWith('.json')).map(f => { const [b, ...c] = f.replace(/\.json$/, '').split('-'); return loadSpec(b, c.join('-')); });
}

/* ---------- Parsing ---------- */
function parseFile(file) {
  const text = fs.readFileSync(file, 'utf8');
  const head = {}, questions = [], errors = [];
  let cur = null;
  const lines = text.split('\n');
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line || line.startsWith('#')) return;
    const m = /^([A-Za-z]+):\s?(.*)$/.exec(line);
    if (!m) { errors.push(`line ${i + 1}: not a "Key: value" line: ${line.slice(0, 60)}`); return; }
    const key = m[1].toLowerCase(), val = m[2].trim();
    if (key === 'q') { cur = { q: val, line: i + 1 }; questions.push(cur); return; }
    if (!cur) {
      if (['board', 'spec', 'topic'].includes(key)) head[key] = val;
      else errors.push(`line ${i + 1}: "${m[1]}:" before the first Q:`);
      return;
    }
    const known = { a: 'a', accept: 'accept', ref: 'ref', d: 'd', ht: 'ht', rp: 'rp', calc: 'calc', sep: 'sep', note: 'note', id: 'id' };
    if (!known[key]) { errors.push(`line ${i + 1}: unknown key "${m[1]}:"`); return; }
    if (cur[key] !== undefined) errors.push(`line ${i + 1}: "${m[1]}:" given twice for one question`);
    cur[key] = val;
  });
  return { file, head, questions, errors, text };
}

const yes = v => /^(yes|y|true|1)$/i.test(String(v || '').trim());
const words = s => String(s).trim().split(/\s+/).filter(Boolean).length;
const norm = s => String(s).toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
const STOP = new Set('the a an of in is are what which name give state how does do to and for on by with from that this its it be as at or why when'.split(' '));
const tokens = s => new Set(norm(s).split(' ').filter(w => w && !STOP.has(w)));
function jaccard(a, b) { let n = 0; a.forEach(x => { if (b.has(x)) n++; }); return n / (a.size + b.size - n || 1); }

/* Turn a parsed file into a pack, checking everything that can be checked on its own */
function buildPack(p) {
  const errors = p.errors.map(e => `${rel(p.file)}: ${e}`), warnings = [];
  const err = (q, msg) => errors.push(`${rel(p.file)}:${q ? q.line : 1}: ${msg}`);
  const board = BOARDS[String(p.head.board || '').toLowerCase()];
  if (!board) err(null, `Board: must be AQA or Edexcel`);
  const spec = board && p.head.spec ? loadSpec(board, p.head.spec) : null;
  if (!spec) { err(null, `Spec: ${p.head.spec || '(missing)'} has no registry in packs/specs`); return { errors, warnings, pack: null }; }
  const topic = spec.topics.find(t => t.ref === p.head.topic);
  if (!topic) { err(null, `Topic: ${p.head.topic} is not a topic of ${board} ${spec.specCode}`); return { errors, warnings, pack: null }; }
  const subject = SUBJECT_IDS[spec.subject];
  const numbering = new RegExp(spec.numbering);
  const htRefs = Object.keys(spec.refs).filter(r => /\(HT only\)/.test(spec.refs[r]));
  // Higher tier only: a heading marked "(HT only)", or a point the registry lists in htRefs
  // (Edexcel marks Higher-only statements in bold rather than with a heading)
  const isHtOnly = ref => htRefs.some(h => ref === h || ref.startsWith(h + '.')) || (spec.htRefs || []).includes(ref);
  // Specifications whose point numbers do not nest under their headings (AQA Maths N1, Edexcel 2.10B,
  // History and Geography content) give each point's subtopic in "parents"
  const parentOf = ref => spec.parents ? spec.parents[ref] : null;
  const inTopic = ref => spec.parents ? topic.subtopics.some(t => t.ref === parentOf(ref)) : (ref + '.').startsWith(topic.ref + '.');
  const subRef = ref => { const parts = ref.split('.'); const depth = spec.subtopicDepth || 3; return parts.slice(0, depth).join('.'); };
  const topicName = displayName(topic.title);
  const map = loadMap(board, subject);
  const questions = p.questions.map(q => {
    ['a', 'ref', 'd'].forEach(k => { if (!q[k]) err(q, `missing ${k === 'd' ? 'D' : k[0].toUpperCase() + k.slice(1)}: for "${q.q.slice(0, 50)}"`); });
    const out = { id: q.id || '', board, subject, specCode: spec.specCode, specRef: q.ref || '', topic: topicName, subtopic: '', tier: yes(q.ht) ? 'Higher' : 'Foundation and Higher', difficulty: +q.d, q: q.q, a: q.a || '', accept: q.accept ? q.accept.split(';').map(s => s.trim()).filter(Boolean) : [], hexOk: false, notes: q.note || '', line: q.line };
    if (q.rp) out.rp = /^\d+$/.test(q.rp) ? +q.rp : q.rp;   // AQA numbers its practicals; Edexcel names them by specification point
    if (yes(q.calc)) out.calc = true;
    if (q.ref) {
      if (!numbering.test(q.ref)) err(q, `Ref ${q.ref} does not match the ${spec.specCode} numbering`);
      else if (!spec.refs[q.ref]) err(q, `Ref ${q.ref} is not a point in the ${spec.specCode} specification`);
      else if (!inTopic(q.ref)) err(q, `Ref ${q.ref} is outside topic ${topic.ref}`);
      else {
        const s = spec.parents ? topic.subtopics.find(t => t.ref === parentOf(q.ref))
          : topic.subtopics.find(t => t.ref === subRef(q.ref)) || topic.subtopics.find(t => q.ref.startsWith(t.ref + '.') || q.ref === t.ref);
        if (!s) err(q, `Ref ${q.ref} must be at least three levels deep (a subtopic)`);
        else out.subtopic = displayName(s.title);
        if (isHtOnly(q.ref) && out.tier !== 'Higher') err(q, `Ref ${q.ref} is Higher tier only in the specification: add "HT: yes"`);
        if (map) {
          // course tags: in Combined Science or not, its Combined Science reference, its tier there
          const e = mapPoint(map, q.ref);
          if (!e) err(q, `Ref ${q.ref} has no entry in tools/spec-map/${map.board.toLowerCase()}-${subject}.json`);
          else {
            if (e.partial && q.sep === undefined) err(q, `Ref ${q.ref} is only partly in Combined Science (${e.partial}): add "Sep: yes" or "Sep: no"`);
            if (!e.comb && q.sep !== undefined && !yes(q.sep)) err(q, `Ref ${q.ref} is not in Combined Science, so "Sep: no" is wrong`);
            if (e.tier === 'Higher' && out.tier !== 'Higher') err(q, `Ref ${q.ref} is Higher tier only: add "HT: yes"`);
            if (!e.comb || yes(q.sep)) out.sepOnly = true;
            else { out.combRef = e.combRef; out.combTier = e.combTier === 'Higher' ? 'Higher' : out.tier; }
          }
        }
      }
    } else if (q.sep !== undefined && !map) err(q, `"Sep:" is only for Biology, Chemistry and Physics`);
    if (map && !q.ref) { /* reported above as a missing Ref */
    }
    if (q.d && ![1, 2, 3].includes(out.difficulty)) err(q, `D: must be 1, 2 or 3`);
    if (q.id && !new RegExp(`^${p.head.board.toLowerCase()}-${SHORT[subject]}-[0-9A-Za-z.-]+-\\d{3}$`).test(q.id)) err(q, `Id ${q.id} has the wrong form`);
    if (q.rp && !(spec.practicals || []).some(r => String(r.n) === String(q.rp))) err(q, `RP: ${q.rp} is not a required practical of ${spec.specCode}`);
    if (out.a) {
      if (words(out.a) > LIMITS.answerWords) err(q, `answer is ${words(out.a)} words (at most ${LIMITS.answerWords}): ${out.a}`);
      else if (words(out.a) > LIMITS.answerWordsSoft) warnings.push(`${rel(p.file)}:${q.line}: answer is ${words(out.a)} words: ${out.a}`);
      if (out.a.length > LIMITS.answerChars) err(q, `answer is ${out.a.length} characters (at most ${LIMITS.answerChars})`);
    }
    if (out.q.length > LIMITS.questionChars) err(q, `question is ${out.q.length} characters (at most ${LIMITS.questionChars})`);
    if (!/[?.:]$|\)$/.test(out.q)) err(q, `question should end with "?", "." or ":"`);
    out.accept.forEach(x => { if (words(x) > LIMITS.acceptWords) err(q, `accepted answer too long: ${x}`); });
    out.hexOk = !!hexLetter(out.a);
    return out;
  });
  // within the pack: duplicates and near-duplicates
  for (let i = 0; i < questions.length; i++) for (let j = 0; j < i; j++) {
    const a = questions[i], b = questions[j];
    if (norm(a.q) === norm(b.q)) err(a, `duplicate of the question on line ${b.line}`);
    else { const s = jaccard(tokens(a.q), tokens(b.q)); if (s >= 0.9 || (s >= 0.75 && norm(a.a) === norm(b.a))) err(a, `near-duplicate of line ${b.line}: "${b.q.slice(0, 60)}"`); }
  }
  // coverage and balance
  const leaves = spec.parents ? Object.keys(spec.parents).filter(inTopic)
    : Object.keys(spec.refs).filter(r => (r + '.').startsWith(topic.ref + '.') && r.split('.').length >= 3 && !Object.keys(spec.refs).some(x => x.startsWith(r + '.')));
  const target = Math.max(LIMITS.minPerTopic, LIMITS.perLeaf * leaves.length);
  const n = questions.length;
  const byD = [1, 2, 3].map(d => questions.filter(q => q.difficulty === d).length);
  const hex = questions.filter(q => q.hexOk).length;
  if (n < target) err(null, `${n} questions; this topic needs at least ${target} (${leaves.length} specification points)`);
  byD.forEach((c, i) => { if (n && c < LIMITS.minShare * n) err(null, `only ${c} of ${n} questions at difficulty ${i + 1} (at least a fifth)`); });
  if (n && hex < LIMITS.hexShare * n) err(null, `only ${hex} of ${n} answers work on a Hex Hunt board (at least a third)`);
  leaves.forEach(r => { const c = questions.filter(q => q.specRef === r || q.specRef.startsWith(r + '.')).length; if (c < LIMITS.minLeaf) err(null, `point ${r} (${spec.refs[r]}) has ${c} question${c === 1 ? '' : 's'} (at least ${LIMITS.minLeaf})`); });
  (spec.practicals || []).filter(r => inTopic(r.ref)).forEach(r => { const c = questions.filter(q => String(q.rp) === String(r.n)).length; if (c < LIMITS.minRP) err(null, `required practical ${r.n} has ${c} question${c === 1 ? '' : 's'} (at least ${LIMITS.minRP})`); });
  // Category Clash, one topic chosen: every subtopic is a column with a question at each difficulty
  const thin = topic.subtopics.filter(s => questions.some(q => q.subtopic === displayName(s.title))).filter(s => [1, 2, 3].some(d => !questions.some(q => q.subtopic === displayName(s.title) && q.difficulty === d)));
  thin.forEach(s => warnings.push(`${rel(p.file)}: subtopic ${s.ref} ${s.title} lacks a question at some difficulty (Category Clash fills it from the nearest)`));
  const pack = {
    id: `${p.head.board.toLowerCase()}-${subject}-${spec.specCode}-${topic.ref}`, file: rel(p.file), board, subject, specCode: spec.specCode, specVersion: `${spec.version}, ${spec.date}`,
    specTitle: spec.title || '', topic: topicName, topicRef: topic.ref, order: spec.topics.indexOf(topic),
    subtopics: topic.subtopics.map(s => displayName(s.title)), questions,
    comb: combOf(map, topic.ref),
    stats: { comb: questions.filter(q => !q.sepOnly && map).length, combHt: questions.filter(q => map && !q.sepOnly && q.combTier === 'Higher').length, n, target, byD, hex, ht: questions.filter(q => q.tier === 'Higher').length, leaves: leaves.length }
  };
  return { errors, warnings, pack };
}

/* The topic in Combined Science: its specification, reference, name and place in the order (null if
   the whole topic is separate science only, or the subject has no Combined Science course) */
function combOf(map, ref) {
  if (!map) return null;
  const i = map.topics.findIndex(t => t.sep === ref);
  return i < 0 ? null : { specCode: map.combined.specCode, name: map.combined.name, version: map.combined.version, ref: map.topics[i].comb, topic: map.topics[i].combTitle, order: i };
}

const rel = f => path.relative(path.join(__dirname, '..'), f);
function packFiles(dir) {
  dir = dir || ROOT;
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory() && e.name !== 'specs') out.push(...packFiles(p));
    else if (e.isFile() && e.name.endsWith('.txt')) out.push(p);
  }
  return out.sort();
}

/* Every pack, with the checks that span packs: unique ids, and no duplicate question within a subject and board */
function loadAll() {
  const results = packFiles().map(f => buildPack(parseFile(f)));
  const errors = results.flatMap(r => r.errors), warnings = results.flatMap(r => r.warnings);
  const packs = results.map(r => r.pack).filter(Boolean);
  const ids = new Map();
  packs.forEach(p => p.questions.forEach(q => {
    if (!q.id) return;
    if (ids.has(q.id)) errors.push(`${p.id}: id ${q.id} is used twice`);
    ids.set(q.id, p.id);
  }));
  const seen = new Map();
  packs.forEach(p => p.questions.forEach(q => {
    const k = p.board + '|' + p.subject + '|' + norm(q.q);
    if (seen.has(k) && seen.get(k) !== p.id) errors.push(`${p.id}: "${q.q.slice(0, 60)}" is also in ${seen.get(k)}`);
    seen.set(k, p.id);
  }));
  packs.sort((a, b) => a.board.localeCompare(b.board) || a.subject.localeCompare(b.subject) || a.specCode.localeCompare(b.specCode) || topicIndex(a) - topicIndex(b));
  return { packs, errors, warnings };
}

/* ---------- Ids ---------- */
function assignIds(only) {
  let added = 0;
  packFiles().filter(f => !only || rel(f).startsWith(only)).forEach(f => {
    const p = parseFile(f);
    const prefix = `${String(p.head.board).toLowerCase()}-${SHORT[SUBJECT_IDS[(loadSpec(p.head.board, p.head.spec) || {}).subject]] || 'x'}-${/^1G[AB]0$/.test(p.head.spec) ? p.head.spec.toLowerCase() + '-' : ''}${p.head.topic}-`;   // Geography A and B share topic numbers
    let next = 1 + Math.max(0, ...p.questions.map(q => q.id && q.id.startsWith(prefix) ? parseInt(q.id.slice(prefix.length), 10) : 0));
    const lines = p.text.split('\n');
    // insert after each question's last line, from the bottom up so line numbers stay right
    const todo = p.questions.filter(q => !q.id);
    if (!todo.length) return;
    const ends = p.questions.map((q, i) => { const nextLine = i + 1 < p.questions.length ? p.questions[i + 1].line - 1 : lines.length; let e = nextLine; while (e > q.line && !lines[e - 1].trim()) e--; return e; });
    const ins = [];
    p.questions.forEach((q, i) => { if (!q.id) ins.push([ends[i], prefix + String(next++).padStart(3, '0')]); });
    ins.sort((a, b) => b[0] - a[0]).forEach(([at, id]) => lines.splice(at, 0, 'Id: ' + id));
    fs.writeFileSync(f, lines.join('\n'));
    added += ins.length;
  });
  return added;
}

/* ---------- For the build: compact data for the single HTML file ---------- */
/* A pack with any error is left out of the build (with a warning) rather than shipped; the test
   suite (tests/packs.spec.js) fails on any error, so nothing is left out of a release unnoticed. */
function bundleData() {
  const all = loadAll();
  const bad = new Set(all.errors.map(e => (all.packs.find(p => e.includes(p.id) || e.includes(p.file)) || {}).id).filter(Boolean));
  const packs = all.packs.filter(p => !bad.has(p.id) && p.questions.every(q => q.id));
  const left = all.packs.length - packs.length;
  if (left) console.warn(`Warning: ${left} question pack${left === 1 ? '' : 's'} left out of the build (errors or missing ids); run "node tools/packs.js check".`);
  const specs = {};
  packs.forEach(p => { specs[p.board + ' ' + p.specCode] = p.specVersion; });
  return {
    specs,
    packs: packs.map(p => ({
      id: p.id, board: p.board, subject: p.subject, specCode: p.specCode, topic: p.topic, ref: p.topicRef, order: p.order, subtopics: p.subtopics,
      comb: p.comb || undefined, nc: p.comb ? p.stats.comb : undefined,
      // [id, specRef, subtopic index, difficulty, Higher only, question, answer, accepted answers, hexOk, note, extras]
      n: p.questions.length,
      // the rows, DEFLATE-compressed and base64 (src/shared/inflate.js unpacks a pack when first used)
      z: zlib.deflateRawSync(Buffer.from(JSON.stringify(p.questions.map(q => {
        const row = [q.id, q.specRef, p.subtopics.indexOf(q.subtopic), q.difficulty, q.tier === 'Higher' ? 1 : 0, q.q, q.a, q.accept.join(';'), q.hexOk ? 1 : 0, q.notes];
        // extras: required practical, calculation, and the course tags of the sciences
        const x = { rp: q.rp, calc: q.calc, sepOnly: q.sepOnly, combRef: q.combRef, combTier: q.combTier && q.combTier !== q.tier ? q.combTier : undefined };
        if (Object.values(x).some(v => v !== undefined)) row.push(x);
        return row;
      })), 'utf8'), { level: 9 }).toString('base64')
    }))
  };
}

module.exports = { loadMap, mapPoint, parseFile, buildPack, loadAll, assignIds, bundleData, allSpecs, loadSpec, LIMITS, packFiles, norm, tokens, jaccard };

/* ---------- Command line ---------- */
if (require.main === module) {
  const cmd = process.argv[2] || 'check';
  if (cmd === 'ids') { console.log(`Added ${assignIds(process.argv[3])} ids.`); process.exit(0); }
  const { packs, errors, warnings } = loadAll();
  if (cmd === 'report' || cmd === 'check') {
    const rows = packs.map(p => `| ${p.board} | ${p.subject} | ${p.topicRef} ${p.topic} | ${p.stats.n} / ${p.stats.target} | ${p.stats.byD.join(' / ')} | ${Math.round(100 * p.stats.hex / (p.stats.n || 1))}% | ${p.stats.ht} |`);
    console.log('| Board | Subject | Topic | Questions / target | Difficulty 1 / 2 / 3 | Hex Hunt | Higher only |\n|---|---|---|---|---|---|---|\n' + rows.join('\n'));
    console.log(`\n${packs.length} packs, ${packs.reduce((s, p) => s + p.stats.n, 0)} questions.`);
  }
  if (cmd === 'check') {
    if (process.argv.includes('--warnings')) warnings.forEach(w => console.log('warning: ' + w));
    else if (warnings.length) console.log(`${warnings.length} warnings (add --warnings to list them).`);
    errors.forEach(e => console.log('ERROR: ' + e));
    console.log(errors.length ? `${errors.length} errors.` : 'No errors.');
    process.exit(errors.length ? 1 : 0);
  }
}
