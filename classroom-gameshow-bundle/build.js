#!/usr/bin/env node
/*
 * Build: turns src/index.html into one self-contained file.
 *   node build.js            -> dist/showtime-classroom-gameshows.html   (the product)
 *                               test-build/showtime-test.html            (tests only, not shipped)
 *   node build.js --force    rebuild even if nothing in src/ has changed
 *   node build.js --editions every edition for sale (tools/editions.js) -> dist/editions/<file>,
 *                            with SHA256SUMS.txt, and test builds in test-build/editions/
 *
 * Every edition is built from the same source; they differ only in the built-in packs and the
 * edition name (CGB.EDITION). The main product file is the mega bundle (every subject).
 *
 * Code between the @test-only and @end-test-only comment markers (shortcuts and seeding for
 * tests) is kept in the test build and removed from the product. The build stops if any
 * of it is left in the product.
 *
 * It inlines, in order:
 *   <!-- @include path -->           HTML fragments
 *   <!-- @text path -->              plain text (HTML-escaped), e.g. licences
 *   <link rel="stylesheet" href>     as <style>, with url(font.woff2) turned into base64 data URIs
 *   <script src>                     as inline <script>
 *   <!-- @packdata -->               the built-in question packs from packs/ (see tools/packs.js)
 * No dependencies beyond Node itself.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, 'src');
const OUT_DIR = path.join(__dirname, 'dist');
const OUT = path.join(OUT_DIR, 'showtime-classroom-gameshows.html');
const TEST_DIR = path.join(__dirname, 'test-build');
const TEST_OUT = path.join(TEST_DIR, 'showtime-test.html');

// Skip the build when nothing it reads has changed since the last one
function newest(dir) {
  let t = 0;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    t = Math.max(t, e.isDirectory() ? newest(p) : fs.statSync(p).mtimeMs);
  }
  return t;
}
const EDITIONS_MODE = process.argv.includes('--editions');
if (!EDITIONS_MODE && !process.argv.includes('--force') && fs.existsSync(OUT) && fs.existsSync(TEST_OUT)) {
  const src = Math.max(newest(SRC), newest(path.join(__dirname, 'packs')), fs.statSync(path.join(__dirname, 'tools', 'packs.js')).mtimeMs, fs.statSync(__filename).mtimeMs);
  if (Math.min(fs.statSync(OUT).mtimeMs, fs.statSync(TEST_OUT).mtimeMs) > src) {
    console.log('Up to date: nothing in src/ has changed since the last build (use --force to rebuild).');
    process.exit(0);
  }
}

const read = p => fs.readFileSync(path.join(SRC, p), 'utf8');
const escapeHtml = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function inlineCss(rel) {
  const dir = path.dirname(rel);
  return read(rel).replace(/url\(([^)'"]+\.woff2)\)/g, (m, file) => {
    const b64 = fs.readFileSync(path.join(SRC, dir, file)).toString('base64');
    return `url(data:font/woff2;base64,${b64})`;
  });
}
function inlineJs(rel) {
  let js = read(rel);
  if (/<\/script/i.test(js)) js = js.replace(/<\/script/gi, '<\\/script');
  // drop source-map hints so browsers never try to fetch a .map file
  js = js.replace(/^\/\/# sourceMappingURL=.*$/gm, '');
  return js;
}

function buildHtml(ed) {
  // an edition: its name, and only its built-in packs (tools/editions.js)
  let html = read('index.html');
  html = html.replace(/<!-- @include ([^ ]+) -->/g, (m, p) => read(p));
  // The built-in question packs (packs/, checked by tools/packs.js) as one data script
  html = html.replace('<!-- @packdata -->', () => {
    const data = JSON.stringify(require('./tools/packs.js').bundleData(p => editions.includes(ed, p))).replace(/<\/(script)/gi, '<\\/$1');
    return `<script>\nCGB.EDITION = ${JSON.stringify(editions.runtime(ed))};\nCGB.PACKDATA = ${data};\n</script>`;
  });
  html = html.replace(/<!-- @text ([^ ]+) -->/g, (m, p) => escapeHtml(read(p)));
  html = html.replace(/<link rel="stylesheet" href="([^"]+)">/g, (m, p) => `<style>\n${inlineCss(p)}\n</style>`);
  html = html.replace(/<script src="([^"]+)"><\/script>/g, (m, p) => `<script>\n${inlineJs(p)}\n</script>`);

  // Guard rails: the shipped file must not reference anything external
  const external = html.match(/(?:src|href)\s*=\s*["']\s*(?:https?:)?\/\/[^"']+/gi) || [];
  const imports = html.match(/@import\s+url\(/gi) || [];
  if (external.length || imports.length) {
    console.error('Build stopped: external references found:\n' + external.concat(imports).join('\n'));
    process.exit(1);
  }
  // The product leaves out every test-only block
  const product = html.replace(/[ \t]*\/\* @test-only[\s\S]*?\/\* @end-test-only \*\/[ \t]*\n?/g, '');
  if (/@test-only|@end-test-only|CGB\.test\b|__SHOWTIME_/.test(product)) {
    console.error('Build stopped: test-only code would be left in the product.');
    process.exit(1);
  }
  return { product, html };
}

const editions = require('./tools/editions.js');
const kb = t => (Buffer.byteLength(t) / 1024).toFixed(0) + ' KB';
if (EDITIONS_MODE) {
  const crypto = require('crypto');
  const dir = path.join(OUT_DIR, 'editions'), tdir = path.join(TEST_DIR, 'editions');
  fs.mkdirSync(dir, { recursive: true }); fs.mkdirSync(tdir, { recursive: true });
  const sums = [];
  for (const ed of editions.EDITIONS) {
    const { product, html } = buildHtml(ed);
    fs.writeFileSync(path.join(dir, ed.file), product);
    fs.writeFileSync(path.join(tdir, ed.id + '.html'), html);
    sums.push(crypto.createHash('sha256').update(product).digest('hex') + '  ' + ed.file);
    console.log(`Built dist/editions/${ed.file} (${kb(product)})`);
  }
  fs.writeFileSync(path.join(dir, 'SHA256SUMS.txt'), sums.join('\n') + '\n');
} else {
  const { product, html } = buildHtml(editions.byId('mega-bundle'));
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT, product);
  fs.mkdirSync(TEST_DIR, { recursive: true });
  fs.writeFileSync(TEST_OUT, html);
  console.log(`Built ${path.relative(process.cwd(), OUT)} (${kb(product)}) and the test build`);
}
