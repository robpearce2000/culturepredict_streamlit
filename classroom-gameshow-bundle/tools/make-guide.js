#!/usr/bin/env node
/*
 * Prints docs/teacher-guide.html to dist/teacher-guide.pdf (A4).
 * Run tools/make-listing.js first: the guide uses its screenshots.
 */
'use strict';
const path = require('path');
const { execFileSync } = require('child_process');
const { chromium } = require('@playwright/test');

const ROOT = path.join(__dirname, '..');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.join(ROOT, 'docs', 'teacher-guide.html'));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(500);
  const out = path.join(ROOT, 'dist', 'teacher-guide.pdf');
  await page.pdf({ path: out, format: 'A4', printBackground: true, preferCSSPageSize: true });
  await browser.close();
  let pages = '?';
  try { pages = execFileSync('pdfinfo', [out], { encoding: 'utf8' }).match(/Pages:\s+(\d+)/)[1]; } catch (e) { /* pdfinfo not installed */ }
  console.log(`Wrote dist/teacher-guide.pdf (${pages} pages)`);
  if (pages !== '?' && (+pages < 2 || +pages > 5)) { console.error('The guide must be 2 to 5 pages.'); process.exit(1); }
})().catch(e => { console.error(e); process.exit(1); });
