#!/usr/bin/env node
/*
 * The social profile picture (1080×1080, square: Instagram, Tes and others crop it to a circle, so everything that
 * matters sits in the middle): the host waving on the Showtime stage, drawn by the app in its own artwork.
 *   node tools/make-profile.js   ->  marketing/profile-picture.png  (and marketing/profile-picture-circle-preview.png)
 */
'use strict';
const path = require('path');
const { chromium } = require('@playwright/test');
const ROOT = path.join(__dirname, '..');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1080 } });
  await page.goto('file://' + path.join(ROOT, 'dist', 'editions', 'showtime-mega-bundle.html'));
  await page.waitForFunction(() => window.CGB && CGB.createHost2D && CGB.brand);
  await page.addStyleTag({ content: `
    html, body { margin: 0; overflow: hidden; background: #141834; }
    body > :not(#pf) { display: none !important; }
    #pf { position: fixed; inset: 0; overflow: hidden; background: radial-gradient(circle at 50% 38%, #4A50A8 0%, #2A2F6E 38%, #141834 78%); }
    #pf svg.rays { position: absolute; inset: 0; width: 100%; height: 100%; }
    #pf .ring { position: absolute; inset: 40px; border-radius: 50%; border: 14px solid #FFC93C; box-shadow: 0 0 0 6px #1B1F3B, inset 0 0 0 6px #1B1F3B; }
    #pf .bulbs { position: absolute; inset: 40px; }
    #pf .bulb { position: absolute; width: 26px; height: 26px; margin: -13px; border-radius: 50%; background: #FFF3B0; box-shadow: 0 0 18px 6px rgba(255,201,60,0.8); border: 3px solid #1B1F3B; }
    #pf #who { position: absolute; left: 50%; top: 140px; width: 760px; height: 1180px; transform: translateX(-50%); }
    #pf #who > * { width: 100% !important; height: 100% !important; }
    #pf .name { position: absolute; left: 0; right: 0; bottom: 118px; text-align: center; font-family: var(--display); font-size: 120px; line-height: 1; color: #fff; text-shadow: 0 8px 0 rgba(0,0,0,0.45); -webkit-text-stroke: 8px #1B1F3B; paint-order: stroke fill; }
    #pf .name b { font-weight: 400; color: #FFC93C; }` });
  await page.evaluate(() => {
    const e = document.createElement('div'); e.id = 'pf';
    const rays = '<svg class="rays" viewBox="0 0 100 100" preserveAspectRatio="none"><g fill="rgba(255,255,255,0.07)"><path d="M50 40 L30 110 L45 110 Z"/><path d="M50 40 L70 110 L55 110 Z"/><path d="M50 40 L0 90 L0 105 Z"/><path d="M50 40 L100 90 L100 105 Z"/><path d="M50 40 L8 -10 L22 -10 Z"/><path d="M50 40 L92 -10 L78 -10 Z"/></g></svg>';
    let bulbs = ''; const R = 500 - 40 - 0, n = 36;
    for (let i = 0; i < n; i++) { const a = i / n * Math.PI * 2; bulbs += `<span class="bulb" style="left:${(500 + Math.cos(a) * 460).toFixed(1)}px;top:${(500 + Math.sin(a) * 460).toFixed(1)}px"></span>`; }
    e.innerHTML = rays + '<div class="ring"></div><div class="bulbs" style="inset:0">' + bulbs + '</div><div id="who"></div>';
    document.body.appendChild(e);
    const h = CGB.createHost2D(document.getElementById('who'), { className: 'host-launcher' }); h.gesture('wave', 600000);
  });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(ROOT, 'marketing', 'profile-picture.png') });
  // what the circle crop will show
  await page.addStyleTag({ content: '#pf { clip-path: circle(540px at 50% 50%); background-color: transparent; }' });
  await browser.close();
  console.log('profile picture written');
})().catch(e => { console.error(e); process.exit(1); });
