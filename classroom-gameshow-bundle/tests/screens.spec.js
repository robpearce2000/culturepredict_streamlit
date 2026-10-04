// Screenshots of every screen at three sizes, plus an automatic check that
// banners, speech bubbles, floating labels and panels never overlap each other.
const path = require('path');
const { test } = require('@playwright/test');
const { openBundle, state, expect } = require('./helpers');

const SIZES = [
  { name: '1920x1080', width: 1920, height: 1080 },
  { name: '1366x768', width: 1366, height: 768 },
  { name: 'tablet-768x1024', width: 768, height: 1024 }
];
const OUT = path.join(__dirname, '..', 'screenshots');

/* Elements that must never collide with each other (when visible) */
const WATCH = [
  '.topbar',
  '#ote-bubble.show', '#ote-banner.show .verdict', '#ote-banner.show .detail', '#ote-labels .flabel',
  '.ote-overlay:not(.hidden) .ote-card',
  '.op-tag.show', '.op-hud.active .op-top', '.op-hud.active .op-dock',
  '.op-roundend.show .op-verdict', '.op-roundend.show .op-detail', '.op-roundend.show .btn',
  '.op-screen.active .op-panel',
  '#launcher .l-hero > *', '#launcher .gcard', '#launcher .l-bar', '#launcher .l-mascot .hello',
  '.cc-head .cc-turn', '.cc-head .cc-end', '.cc-q:not([hidden]) .cc-qcard', '.cc-screen.active .cgb-panel',
  '.hh-head > *', '.hh-q:not([hidden]) .hh-qcard', '.hh-win:not([hidden]) > *', '.hh-screen.active .cgb-panel', '.ote-host .host-head', '.ote-host .host-body'
];

async function findProblems(page) {
  return page.evaluate(sel => {
    const vis = el => {
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return null;
      let e = el;
      while (e && e !== document.body) {
        const cs = getComputedStyle(e);
        if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) < 0.15) return null;
        if (e.hidden) return null;
        e = e.parentElement;
      }
      if (r.bottom <= 0 || r.right <= 0 || r.top >= innerHeight || r.left >= innerWidth) return null;
      return r;
    };
    const items = [];
    sel.forEach(s => document.querySelectorAll(s).forEach(el => { const r = vis(el); if (r) items.push({ s, el, r }); }));
    const problems = [];
    for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
      const a = items[i], b = items[j];
      if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
      const ox = Math.min(a.r.right, b.r.right) - Math.max(a.r.left, b.r.left);
      const oy = Math.min(a.r.bottom, b.r.bottom) - Math.max(a.r.top, b.r.top);
      if (ox > 2 && oy > 2) problems.push(`overlap: ${a.s} ↔ ${b.s} (${Math.round(ox)}×${Math.round(oy)}px)`);
    }
    // text that spills off the side of the screen
    document.querySelectorAll('button, h1, h2, h3, p, .qtext, .qanswer, .op-tag, .flabel, .verdict, .detail, .name, .nm, label, .lab').forEach(el => {
      const r = vis(el); if (!r) return;
      if (el.closest('#ote-labels')) return;          // floating labels follow 3D objects and fade at the edges
      if (r.left < -1 || r.right > innerWidth + 1) problems.push(`off-screen: ${el.className || el.tagName} "${(el.textContent || '').trim().slice(0, 40)}"`);
    });
    return problems;
  }, WATCH);
}

/* Each size runs as three smaller batches (launcher, Over the Edge, Outpace),
   each in a fresh page with its own time limit, so one slow screen can't stall the sweep. */
test.describe.configure({ mode: 'parallel' });
const SECTIONS = {
  launcher: async ({ page, shot }) => {
    // ---------- Launcher ----------
    await page.waitForTimeout(1500);
    await shot('01-launcher');
    await page.click('#openBank'); await shot('02-question-bank');
    await page.locator('#bankForm').scrollIntoViewIfNeeded(); await shot('03-question-bank-add');
    await page.keyboard.press('Escape');
    await page.click('#openAbout'); await shot('04-about'); await page.keyboard.press('Escape');
    await page.click('#launcherSettings [data-key="textSize"] button:nth-child(2)');
    await shot('05-launcher-large-text');

  },
  'over-the-edge': async ({ page, shot, until, fits }) => {
    // ---------- Over the Edge (large text stays on for the first screens) ----------
    await page.evaluate(() => CGB.settings.set('textSize', 'large'));
    await page.click('[data-play="over-the-edge"]');
    await shot('10-ote-setup-large-text', 1500);
    await fits('over-the-edge', '10-ote-setup-large-text');
    await page.click('#game-over-the-edge #ote-settingsBtn'); await shot('11-settings-modal'); await page.keyboard.press('Escape');
    await page.evaluate(() => CGB.settings.set('textSize', 'normal'));
    await fits('over-the-edge', '10b-ote-setup');
    await page.click('#ote-toggleHost'); await shot('12-ote-setup-host'); await page.click('#ote-toggleHost');
    await page.click('#ote-segR1 button[data-v="4"]');
    await page.click('#ote-segF button[data-v="8"]');
    await page.click('#ote-startBtn');
    await shot('13-ote-question', 900);
    await page.keyboard.press('a'); await shot('14-ote-answer-shown', 300);
    await page.keyboard.press('w'); await shot('15-ote-steal', 300);
    await page.keyboard.press('c'); await shot('16-ote-lanes', 400);
    await page.keyboard.press('2'); await shot('17-ote-dropping', 1600);
    await until(async () => (await state(page, 'over-the-edge')).step === 'next', 30000);
    await shot('18-ote-after-drop', 200);
  },
  'over-the-edge-final': async ({ page, shot, until }) => {
    await page.click('[data-play="over-the-edge"]');
    await page.waitForTimeout(1200);
    await page.click('#ote-segR1 button[data-v="4"]');
    await page.click('#ote-segF button[data-v="8"]');
    await page.click('#ote-startBtn');
    // straight to the final (Round 1 is covered by the other batch and the gameplay tests)
    await page.waitForTimeout(600);
    await page.evaluate(() => CGB.test.ote.toFinal());
    await until(async () => (await state(page, 'over-the-edge')).phase === 'final', 30000);
    await shot('19-ote-final', 1200);
    await page.keyboard.press('c'); await page.keyboard.press('3');
    await shot('20-ote-final-drop', 2500);
    await until(async () => {
      const s = await state(page, 'over-the-edge');
      if (s.phase === 'summary') return true;
      if (s.step === 'ask') await page.keyboard.press('w'); else if (s.step === 'steal') await page.keyboard.press('n'); else if (s.step === 'next') await page.keyboard.press('Space'); else if (s.step === 'chute') await page.keyboard.press('4');
      return false;
    }, 240000);
    expect((await state(page, 'over-the-edge')).phase, 'reached the results').toBe('summary');
    await shot('21-ote-summary', 800);

  },
  'category-clash': async ({ page, shot, until, fits }) => {
    await page.click('[data-play="category-clash"]');
    await shot('50-cc-setup', 800);
    await fits('category-clash', '50-cc-setup');
    await page.click('#cc-segTeams button[data-v="4"]');
    await page.click('#cc-segCats button[data-v="6"]');
    await shot('51-cc-setup-4-teams', 300);
    await fits('category-clash', '51-cc-setup-4-teams');
    await page.click('#cc-startBtn');
    await shot('52-cc-board', 500);
    await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
    await shot('53-cc-question', 400);
    await page.keyboard.press('w'); await shot('54-cc-steal', 300);
    await page.keyboard.press('2'); await shot('55-cc-stolen', 300);
    await page.keyboard.press('Enter');
    let n = 0;
    await until(async () => {
      const s = await state(page, 'category-clash');
      if (s.phase === 'summary') return true;
      if (s.phase === 'board') await page.keyboard.press('Enter');
      else if (s.step === 'ask') await page.keyboard.press(n++ % 3 ? 'c' : 'w');
      else if (s.step === 'steal') await page.keyboard.press('n');
      else if (s.step === 'done') { if (n === 8) await shot('56-cc-board-midgame', 200); await page.keyboard.press('Enter'); }
      return false;
    }, 120000);
    await shot('57-cc-results', 600);
  },
  'hex-hunt': async ({ page, shot, until, fits }) => {
    await page.click('[data-play="hex-hunt"]');
    await shot('60-hh-setup', 800);
    await fits('hex-hunt', '60-hh-setup');
    await page.click('#hh-startBtn');
    await shot('61-hh-board', 500);
    await page.keyboard.press('Enter');
    await shot('62-hh-question', 400);
    await page.keyboard.press('1'); await page.keyboard.press('w');
    await shot('63-hh-other-team', 300);
    await page.keyboard.press('2'); await page.keyboard.press('c');
    await shot('64-hh-claimed', 300);
    await page.keyboard.press('Enter');
    let n = 0;
    await until(async () => {
      const s = await state(page, 'hex-hunt');
      if (s.phase === 'won') return true;
      if (s.phase === 'board') { if (n === 10) await shot('65-hh-board-midgame', 200); await page.keyboard.press('Enter'); }
      else if (s.step === 'buzz') await page.keyboard.press(n++ % 2 ? '1' : '2');
      else if (s.step === 'answer') await page.keyboard.press('c');
      else if (s.step === 'claimed' || s.step === 'nobody') await page.keyboard.press('Enter');
      return false;
    }, 90000);
    await shot('66-hh-round-won', 900);
    await page.keyboard.press('Enter');
    await shot('67-hh-results', 600);
  },
  outpace: async ({ page, shot, until, fits }) => {
    // ---------- Outpace ----------
    await page.click('[data-play="outpace"]');
    await shot('30-op-setup', 1200);
    await fits('outpace', '30-op-setup');
    await page.click('#op-startBtn');
    await shot('31-op-deal-choose', 1200);
    await page.keyboard.press('2');
    await shot('32-op-deal-question', 1200);
    await page.keyboard.press('a'); await shot('33-op-deal-answer', 300);
    await page.keyboard.press('c'); await shot('33b-op-deal-marked', 300);
    await shot('34-op-deal-step', 1500);
    // get caught quickly
    await until(async () => {
      const s = await state(page, 'outpace');
      if (s.roundEnd) return true;
      if (s.phase === 'deal' && !s.awaitingNext) await page.keyboard.press(s.answerShown ? 'w' : 'a');
      return false;
    }, 60000);
    await shot('35-op-round-end', 300);
    await page.keyboard.press('Enter');
    await page.keyboard.press('3');
    await until(async () => {
      const s = await state(page, 'outpace');
      if (s.roundEnd) return true;
      if (s.phase === 'deal' && !s.awaitingNext) await page.keyboard.press(s.answerShown ? 'w' : 'a');
      return false;
    }, 60000);
    await page.keyboard.press('Enter');
    await shot('36-op-sprint', 1200);
    await page.keyboard.press('a'); await page.keyboard.press('c');
    await page.keyboard.press('a'); await shot('37-op-sprint-answer', 800);
    await page.evaluate(() => CGB.test.outpace.setTime(9));
    await shot('38-op-sprint-low-time', 600);
    await until(async () => (await state(page, 'outpace')).phase === 'summary', 60000);
    await shot('39-op-summary', 800);

  }
};

for (const size of SIZES) {
  for (const [section, run] of Object.entries(SECTIONS)) {
    test(`screens at ${size.name}: ${section}`, async ({ browser }) => {
      test.setTimeout(section === 'launcher' ? 90000 : 420000);
      const ctx = await browser.newContext({ viewport: { width: size.width, height: size.height } });
      const page = await ctx.newPage();
      const log = await openBundle(page, '', { quality: 'high' });   // screenshots at full quality
      const problems = [];
      async function shot(name, settle) {
        await page.waitForTimeout(settle == null ? 600 : settle);
        await page.screenshot({ path: path.join(OUT, size.name, name + '.png'), timeout: 60000 });
        (await findProblems(page)).forEach(p => problems.push(`${name}: ${p}`));
      }
      // the setup card, title to Start button, must be on screen below the top bar with no scrolling
      async function fits(game, name) {
        await page.waitForTimeout(300);
        const r = await page.evaluate(g => {
          const c = document.querySelector(`#game-${g} .setup`), cr = c.getBoundingClientRect();
          const bar = document.querySelector(`#game-${g} .topbar`).getBoundingClientRect();
          const logo = c.querySelector('[class$="-logo"]').getBoundingClientRect();
          const go = c.querySelector('.setup-go .btn').getBoundingClientRect();
          return { top: cr.top, bottom: cr.bottom, left: cr.left, right: cr.right, bar: bar.bottom, logo: logo.top, go: go.bottom, scroll: c.scrollHeight - c.clientHeight, scrollTop: c.scrollTop };
        }, game);
        const bad = [];
        if (r.top < r.bar) bad.push(`card top ${Math.round(r.top)} is under the top bar (${Math.round(r.bar)})`);
        if (r.logo < r.bar) bad.push('title hidden under the top bar');
        if (r.bottom > size.height || r.go > size.height) bad.push(`card bottom ${Math.round(r.bottom)} / Start ${Math.round(r.go)} below the fold`);
        if (r.left < 0 || r.right > size.width) bad.push('card wider than the screen');
        if (r.scroll > 1 || r.scrollTop > 0) bad.push(`card needs scrolling (${r.scroll}px)`);
        bad.forEach(b => problems.push(`${name} setup card: ${b}`));
      }
      const until = async (fn, ms) => { const end = Date.now() + (ms || 30000); while (Date.now() < end) { if (await fn()) return true; await page.waitForTimeout(150); } return false; };
      await run({ page, shot, until, fits });
      await ctx.close();
      expect(log.errors, 'console errors').toEqual([]);
      expect(log.requests, 'network requests').toEqual([]);
      expect(problems, 'layout problems').toEqual([]);
    });
  }
}
