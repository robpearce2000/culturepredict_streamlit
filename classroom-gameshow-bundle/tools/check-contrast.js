#!/usr/bin/env node
/*
 * WCAG 2.1 contrast check for every text colour / background pair the UI uses.
 *   node tools/check-contrast.js
 * Normal text needs 4.5:1 (AA). Large display text (24px+, or 19px+ bold) needs 3:1.
 */
'use strict';
const lum = hex => {
  const n = parseInt(hex.slice(1), 16);
  return [16, 8, 0].map(s => (n >> s) & 255).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); })
    .reduce((a, v, i) => a + v * [0.2126, 0.7152, 0.0722][i], 0);
};
const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };

// [where, text, background, large?]
const PAIRS = [
  ['Panel body text', '#1B1F3B', '#FFF9F0'],
  ['Panel secondary text', '#4A4F6E', '#FFF9F0'],
  ['Secondary text on raised boxes', '#4A4F6E', '#FFF0DA'],
  ['Secondary text on white cards', '#4A4F6E', '#FFFFFF'],
  ['Correct button', '#FFFFFF', '#157A43'],
  ['Wrong button', '#FFFFFF', '#C62828'],
  ['Play buttons', '#FFFFFF', '#B83A06'],
  ['Teal button', '#FFFFFF', '#0A6663'],
  ['Gold button', '#2A1A00', '#FFC93C'],
  ['Selected toggle', '#FFFFFF', '#1B1F3B'],
  ['Links and accents on panels', '#0A6663', '#FFF9F0'],
  ['Team 1 text on white', '#1D4ED8', '#FFFFFF'], ['Team 2 text on white', '#C2410C', '#FFFFFF'],
  ['Team 3 text on white', '#7E22CE', '#FFFFFF'], ['Team 4 text on white', '#0F766E', '#FFFFFF'],
  ['Team 1 text on paper', '#1D4ED8', '#FFF9F0'], ['Team 2 text on paper', '#C2410C', '#FFF9F0'],
  ['Team 1 on the Category Clash board', '#93C5FD', '#1A0F2E'], ['Team 2 on the Category Clash board', '#FDBA74', '#1A0F2E'],
  ['Team 3 on the Category Clash board', '#D8B4FE', '#1A0F2E'], ['Team 4 on the Category Clash board', '#5EEAD4', '#1A0F2E'],
  ['Team marks on used Category Clash tiles (Team 1)', '#93C5FD', '#2B2045'], ['Team marks on used tiles (Team 3)', '#D8B4FE', '#2B2045'],
  ['Team 1 on the Hex Hunt board', '#93C5FD', '#0A1A2C'], ['Team 2 on the Hex Hunt board', '#FDBA74', '#0A1A2C'],
  ['Letters and marks on Team 1 hexagons', '#FFFFFF', '#2563EB', true], ['Letters and marks on Team 2 hexagons', '#FFFFFF', '#EA580C', true],
  ['Outpace step numbers on tiles', '#FFFFFF', '#252C6B', true],
  ['Host caption name', '#0A6663', '#FFFFFF'], ['Host caption text', '#1B1F3B', '#FFFFFF'],
  ['Outpace marked verdict (correct)', '#FFFFFF', '#157A43', true], ['Outpace marked verdict (wrong)', '#FFFFFF', '#C62828', true],
  ['Answer box (Over the Edge)', '#064442', '#E3F6F5'],
  ['Wrong count in summaries', '#C62828', '#FFFFFF'],
  ['Error status', '#C62828', '#FFF9F0'],
  ['Biology tag', '#1B1F3B', '#E3F4E6'], ['Chemistry tag', '#1B1F3B', '#FDE7D6'], ['Physics tag', '#1B1F3B', '#E2ECFB'],
  ['Team pot card', '#1B1F3B', '#FFF3CC'],
  ['Top bar buttons', '#1B1F3B', '#FFFFFF'],
  ['Launcher text', '#FFFFFF', '#141834'],
  ['Launcher secondary text', '#E3E6FF', '#1F2550'],
  ['Launcher footer', '#C9CDF0', '#141834'],
  ['Launcher settings labels on bar', '#E3E6FF', '#2C3159'],
  ['Coming soon heading', '#FFC93C', '#141834', true],
  ['Outpace text', '#F4F6FF', '#161C3D'],
  ['Outpace secondary text', '#C3C9EE', '#161C3D'],
  ['Outpace secondary text on inset boxes', '#C3C9EE', '#0F1430'],
  ['Outpace gold labels', '#FFC93C', '#161C3D'],
  ['Outpace gold labels on background', '#FFC93C', '#0B1026'],
  ['Outpace answer text', '#9FFCF0', '#161C3D'],
  ['Outpace links', '#9FFCF0', '#161C3D'],
  ['Outpace wrong counts', '#F0A3FF', '#0F1430'],
  ['Outpace caught verdict', '#F0A3FF', '#161C3D', true],
  ['Runner name tag', '#1B1F3B', '#FFC93C'],
  ['Hunter name tag', '#FFFFFF', '#C026D3'],
  ['Home tag', '#FFFFFF', '#0A6663'],
  ['Team score badge', '#1B1F3B', '#FFC93C'],
  ['Speech bubble name', '#0A6663', '#FFFFFF'],
  ['Over the Edge round name', '#0A6663', '#FFF9F0', true]
];
let fail = 0;
PAIRS.forEach(([where, fg, bg, large]) => {
  const r = ratio(fg, bg), need = large ? 3 : 4.5, ok = r >= need;
  if (!ok) fail++;
  console.log(`${ok ? 'pass' : 'FAIL'}  ${r.toFixed(2).padStart(5)}:1  (needs ${need})  ${where}  ${fg} on ${bg}`);
});
console.log(fail ? `\n${fail} pair(s) below WCAG AA.` : `\nAll ${PAIRS.length} pairs meet WCAG AA.`);
process.exit(fail ? 1 : 0);
