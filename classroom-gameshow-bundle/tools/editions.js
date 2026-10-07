/*
 * The Showtime editions sold on Tes. Every edition has all four games, the question bank and
 * "write your own"; they differ only in which built-in packs are inside (see DECISIONS.md).
 *
 *   id        used for dist/listings/<id>/ and dist/downloads/
 *   file      the edition's single HTML file in dist/editions/
 *   name      shown on the launcher and in the About panel ("Showtime: <name>")
 *   subjects  the subjects in the picker (Other, for the teacher's own sets, is always added)
 *   packs     optional: only these built-in packs (pack ids from tools/packs.js); default: every
 *             pack of the edition's subjects
 */
'use strict';
const ALL = ['biology', 'chemistry', 'physics', 'maths', 'history', 'geography'];

/* The free taster: one widely taught topic per subject, on both boards where they share it */
const TASTER = [
  'aqa-biology-8461-4.1', 'edexcel-biology-1BI0-1',            // Cell biology / Key concepts in biology
  'aqa-chemistry-8462-4.1', 'edexcel-chemistry-1CH0-1',        // Atomic structure and the periodic table / Key concepts in chemistry
  'aqa-physics-8463-4.1', 'edexcel-physics-1PH0-3',            // Energy / Conservation of energy
  'aqa-maths-8300-3.5', 'edexcel-maths-1MA1-5',                // Probability
  'aqa-history-8145-1AB', 'edexcel-history-1HI0-31',           // Germany 1890-1945 / Weimar and Nazi Germany 1918-39
  'aqa-geography-8035-3.1.1', 'edexcel-geography-1GA0-2', 'edexcel-geography-1GB0-1'   // natural and weather hazards
];

const EDITIONS = [
  { id: 'free-taster', file: 'showtime-free-taster.html', name: 'Free taster', subjects: ALL, packs: TASTER, taster: true },
  { id: 'biology', file: 'showtime-biology.html', name: 'Biology edition', subjects: ['biology'] },
  { id: 'chemistry', file: 'showtime-chemistry.html', name: 'Chemistry edition', subjects: ['chemistry'] },
  { id: 'physics', file: 'showtime-physics.html', name: 'Physics edition', subjects: ['physics'] },
  { id: 'maths', file: 'showtime-maths.html', name: 'Maths edition', subjects: ['maths'] },
  { id: 'history', file: 'showtime-history.html', name: 'History edition', subjects: ['history'] },
  { id: 'geography', file: 'showtime-geography.html', name: 'Geography edition', subjects: ['geography'] },
  { id: 'science', file: 'showtime-science.html', name: 'Science mega pack', subjects: ['biology', 'chemistry', 'physics'] },
  { id: 'mega-bundle', file: 'showtime-mega-bundle.html', name: 'Mega bundle', subjects: ALL, all: true }
];

/* Does a built-in pack belong in an edition? */
const includes = (ed, pack) => ed.packs ? ed.packs.includes(pack.id) : ed.subjects.includes(pack.subject);
/* What the page is told about its edition (CGB.EDITION) */
const runtime = ed => ({ id: ed.id, name: ed.name, subjects: ed.subjects, taster: !!ed.taster, all: !!ed.all });

module.exports = { EDITIONS, TASTER, ALL, includes, runtime, byId: id => EDITIONS.find(e => e.id === id) };
