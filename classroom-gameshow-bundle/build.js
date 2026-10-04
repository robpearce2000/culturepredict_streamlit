#!/usr/bin/env node
/*
 * Build: turns src/index.html into one self-contained file.
 *   node build.js            -> dist/showtime-classroom-gameshows.html
 *
 * It inlines, in order:
 *   <!-- @include path -->           HTML fragments
 *   <!-- @text path -->              plain text (HTML-escaped), e.g. licences
 *   <link rel="stylesheet" href>     as <style>, with url(font.woff2) turned into base64 data URIs
 *   <script src>                     as inline <script>
 * No dependencies beyond Node itself.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, 'src');
const OUT_DIR = path.join(__dirname, 'dist');
const OUT = path.join(OUT_DIR, 'showtime-classroom-gameshows.html');

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

let html = read('index.html');
html = html.replace(/<!-- @include ([^ ]+) -->/g, (m, p) => read(p));
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
fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(OUT, html);
console.log(`Built ${path.relative(process.cwd(), OUT)} (${(Buffer.byteLength(html) / 1024).toFixed(0)} KB)`);
