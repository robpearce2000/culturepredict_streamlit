// Screenshots of every screen at three sizes, plus an automatic check that
// banners, speech bubbles, floating labels and panels never overlap each other.
const path = require('path');
const { test } = require('@playwright/test');
const { openBundle, state, expect } = require('./helpers');
const { brightShare } = require('./png');

const SIZES = [
  { name: '1920x1080', width: 1920, height: 1080 },
  { name: '1366x768', width: 1366, height: 768 },
  { name: 'tablet-768x1024', width: 768, height: 1024 }
];
const OUT = path.join(__dirname, '..', 'screenshots');
/* The sweep runs at High graphics by default; SWEEP=low runs it at Low graphics and SWEEP=reduced
   at High with reduced motion (each into its own folder) */
const VARIANT = process.env.SWEEP || 'high';
const SUFFIX = VARIANT === 'high' ? '' : '-' + VARIANT;

/* Elements that must never collide with each other (when visible) */
const WATCH = [
  '.topbar',
  '#ote-bubble.show', '#ote-banner.show .verdict', '#ote-banner.show .detail', '#ote-labels .flabel',
  '.ote-overlay:not(.hidden) .ote-card',
  '.op-tag.show', '.op-hud.active .op-top', '.op-hud.active .op-dock',
  '.op-roundend.show .op-verdict', '.op-roundend.show .op-detail', '.op-roundend.show .btn',
  '.op-screen.active .op-panel',
  '#launcher .l-hero > *', '#launcher .gcard', '#launcher .l-bar', '#launcher .l-mascot .hello', '#launcher .l-host .btn',
  '.cc-head .cc-turn', '.cc-head .cc-end', '.cc-q:not([hidden]) .cc-qcard', '.cc-screen.active .cgb-panel',
  '.hh-head > *', '.hh-q:not([hidden]) .hh-qcard', '.hh-win:not([hidden]) > *', '.hh-screen.active .cgb-panel', '.ote-host .host-head', '.ote-host .host-body',
  '.host-foot .hc-bust', '.host-foot .hc-bubble.show', '.op-hostrow .hc-bust', '.op-hostrow .hc-bubble.show',
  '.cm-team', '.hh-side',
  '.hh-curhint:not([hidden])', '.op-hud.active .op-gap'
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
      // content scrolled out of a scrolling panel is clipped away: only the part still showing counts
      let c = { left: r.left, right: r.right, top: r.top, bottom: r.bottom };
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        const o = getComputedStyle(p).overflowY;
        if (o === 'auto' || o === 'scroll') {
          const pr = p.getBoundingClientRect();
          c = { left: c.left, right: c.right, top: Math.max(c.top, pr.top), bottom: Math.min(c.bottom, pr.bottom) };
          if (c.bottom - c.top < 2) return null;
        }
      }
      return Object.assign(c, { width: c.right - c.left, height: c.bottom - c.top });
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
    // cards and panel blocks whose text spills out of them
    document.querySelectorAll('.ote-panel > *, .op-qcard, .cc-qcard, .hh-qcard, .cgb-panel').forEach(el => {
      const r = vis(el); if (!r) return;
      const cs = getComputedStyle(el);
      if (cs.overflowY === 'visible' && el.scrollHeight > el.clientHeight + 2) problems.push(`text spills out of ${el.className || el.id}`);
    });
    // team names on the team panels must be readable, not cut off
    document.querySelectorAll('.cm-nm').forEach(el => {
      if (vis(el) && el.scrollWidth > el.clientWidth + 1) problems.push(`team name cut off: "${el.textContent.trim()}"`);
    });
    // text that spills off the side of the screen
    document.querySelectorAll('button, h1, h2, h3, p, .qtext, .qanswer, .op-tag, .flabel, .verdict, .detail, .name, .nm, label, .lab').forEach(el => {
      const r = vis(el); if (!r) return;
      if (el.closest('#ote-labels')) return;          // floating labels follow 3D objects and fade at the edges
      if (r.left < -1 || r.right > innerWidth + 1) problems.push(`off-screen: ${el.className || el.tagName} "${(el.textContent || '').trim().slice(0, 40)}"`);
    });
    return problems;
  }, WATCH);
}

/* Each size runs as smaller batches (the launcher, then each game), each in a fresh page with
   its own time limit, so one slow screen can't stall the sweep. Games are shown with six teams. */
test.describe.configure({ mode: 'parallel' });
const SECTIONS = {
  launcher: async ({ page, shot }) => {
    // ---------- Launcher: subject, exam board, questions, host ----------
    await page.waitForTimeout(1500);
    await shot('01-launcher');
    await page.click('#openBank'); await shot('02-question-bank');
    await page.locator('#bankForm').scrollIntoViewIfNeeded(); await shot('03-question-bank-add');
    await page.keyboard.press('Escape');
    await page.click('#openAbout'); await shot('04-about'); await page.keyboard.press('Escape');
    await page.click('#openHost'); await shot('06-host-menu', 900); await page.keyboard.press('Escape');
    await page.selectOption('#subjectSelect', 'history'); await page.selectOption('#boardSelect', 'edexcel');
    await shot('07-launcher-no-pack', 300);
    await page.selectOption('#subjectSelect', 'combined'); await page.selectOption('#boardSelect', 'aqa');
    await page.click('#launcherSettings [data-key="textSize"] button:nth-child(2)');
    await shot('05-launcher-large-text');
  },
  'over-the-edge': async ({ page, shot, until, fits }) => {
    // ---------- Over the Edge, six teams (large text stays on for the first screens) ----------
    await page.evaluate(() => CGB.settings.set('textSize', 'large'));
    await page.click('[data-play="over-the-edge"]');
    await page.click('#ote-segTeams button[data-v="6"]');
    await shot('10-ote-setup-large-text', 1500);
    await fits('over-the-edge', '10-ote-setup-large-text');
    await page.click('#game-over-the-edge #ote-settingsBtn'); await shot('11-settings-modal'); await page.keyboard.press('Escape');
    await page.evaluate(() => CGB.settings.set('textSize', 'normal'));
    await fits('over-the-edge', '10b-ote-setup');
    await page.click('#game-over-the-edge details.tnames summary');
    await shot('12-ote-setup-names', 400); await fits('over-the-edge', '12-ote-setup-names');
    await page.click('#ote-startBtn');
    await shot('13-ote-question-countdown', 1200);
    await page.keyboard.press('Space');
    await shot('14-ote-show-me', 700);
    await until(async () => (await state(page, 'over-the-edge')).round === 'mark', 10000);
    await page.keyboard.press('c'); await page.keyboard.press('2');
    await shot('15-ote-marking', 300);
    await page.keyboard.press('Enter');
    await shot('16-ote-lanes', 600);
    await page.keyboard.press('r');
    await shot('17-ote-dropping', 1800);
    await until(async () => (await state(page, 'over-the-edge')).step === 'next', 90000);
    await shot('18-ote-after-drop', 300);
  },
  'over-the-edge-final': async ({ page, shot, until }) => {
    await page.click('[data-play="over-the-edge"]');
    await page.click('#ote-segTeams button[data-v="6"]');
    await page.waitForTimeout(1200);
    await page.click('#ote-startBtn');
    // straight to the final, with one team far behind (Round 1 is covered by the other batch)
    await page.waitForTimeout(600);
    await page.evaluate(() => { CGB.test.ote.setMoney([600, 400, 300, 0, 200, 500]); CGB.test.ote.toFinal(); });
    await until(async () => (await state(page, 'over-the-edge')).phase === 'final', 30000);
    await shot('19-ote-final-catch-up', 1500);
    await page.keyboard.press('2');
    await until(async () => (await state(page, 'over-the-edge')).step === 'next', 90000);
    await page.keyboard.press('Space');
    await shot('20-ote-final', 1200);
    await page.keyboard.press('Space'); await page.keyboard.press('Space');
    await until(async () => (await state(page, 'over-the-edge')).round === 'mark', 10000);
    await page.keyboard.press('1'); await page.keyboard.press('4'); await page.keyboard.press('Enter');
    await page.keyboard.press('r');
    await shot('21-ote-final-drop', 2500);
    // the rest: nobody right, so no more drops
    await until(async () => {
      const s = await state(page, 'over-the-edge');
      if (s.phase === 'summary') return true;
      if (s.round === 'think' || s.round === 'show') await page.keyboard.press('Space');
      else if (s.round === 'mark') { await page.keyboard.press('w'); await page.keyboard.press('Enter'); }
      else if (s.step === 'lanes') await page.keyboard.press('r');
      else if (s.step === 'next') await page.keyboard.press('Space');
      return false;
    }, 240000);
    expect((await state(page, 'over-the-edge')).phase, 'reached the results').toBe('summary');
    await shot('22-ote-summary', 800);
  },
  outpace: async ({ page, shot, until, fits }) => {
    // ---------- Outpace, six teams ----------
    await page.click('[data-play="outpace"]');
    await page.click('#op-segGroups button[data-v="6"]');
    await shot('30-op-setup', 1200);
    await fits('outpace', '30-op-setup');
    await page.click('#game-outpace details.tnames summary');
    await shot('30b-op-setup-names', 400); await fits('outpace', '30b-op-setup-names');
    await page.click('#op-startBtn');
    await shot('31-op-vote', 1200);
    await page.keyboard.press('2');
    await shot('32-op-question-countdown', 1500);
    await page.keyboard.press('Space'); await page.keyboard.press('Space');
    await until(async () => (await state(page, 'outpace')).round === 'mark', 10000);
    for (const k of ['1', '2', '4', '5']) await page.keyboard.press(k);
    await shot('33-op-marking', 300);
    await page.keyboard.press('Enter');
    await shot('34-op-result', 1500);
    // get caught quickly: one team right each time
    await until(async () => {
      const s = await state(page, 'outpace');
      if (s.roundEnd) return true;
      if (s.round === 'think' || s.round === 'show') await page.keyboard.press('Space');
      else if (s.round === 'mark') { await page.keyboard.press('1'); await page.keyboard.press('Enter'); }
      else if (s.round === 'done') await page.keyboard.press('Enter');
      return false;
    }, 90000);
    await shot('35-op-round-end', 1200);
    await page.evaluate(() => CGB.test.outpace.toSprint(600));
    await shot('36-op-sprint', 1500);
    await page.keyboard.press('Space');
    await until(async () => (await state(page, 'outpace')).round === 'mark', 10000);
    await page.keyboard.press('c'); await page.keyboard.press('3'); await page.keyboard.press('Enter');
    await shot('37-op-sprint-marked', 500);
    await page.evaluate(() => CGB.test.outpace.setTime(9));
    await shot('38-op-sprint-low-time', 1600);
    await page.evaluate(() => CGB.test.outpace.setTime(1));
    await until(async () => (await state(page, 'outpace')).phase === 'summary', 60000);
    await shot('39-op-summary', 1000);
  },
  'category-clash': async ({ page, shot, until, fits }) => {
    await page.click('[data-play="category-clash"]');
    await shot('50-cc-setup', 800);
    await fits('category-clash', '50-cc-setup');
    await page.click('#cc-segTeams button[data-v="6"]');
    await page.click('#game-category-clash details.tnames:not(.cc-topicpick) summary');
    await shot('51-cc-setup-names', 300);
    await fits('category-clash', '51-cc-setup-names');
    await page.click('#cc-startBtn');
    await shot('52-cc-board', 500);
    await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
    await shot('53-cc-question-countdown', 800);
    await page.keyboard.press('Space');
    await shot('54-cc-show-me', 700);
    await until(async () => (await state(page, 'category-clash')).round === 'mark', 10000);
    await shot('55-cc-marking', 200);
    for (const k of ['1', '3', '4', '6']) await page.keyboard.press(k);
    await shot('56-cc-marked', 200);
    await page.keyboard.press('Enter');
    await shot('57-cc-result', 700);
    await page.keyboard.press('Enter');
    let n = 0;
    await until(async () => {
      const s = await state(page, 'category-clash');
      if (s.phase === 'summary') return true;
      if (s.phase === 'board') { if (n === 5) await shot('58-cc-board-midgame', 200); await page.keyboard.press('Enter'); }
      else if (s.round === 'done') await page.keyboard.press('Enter');
      else if (s.round === 'think' || s.round === 'show') await page.keyboard.press('Space');
      else if (s.round === 'mark') { n++; await page.keyboard.press(String(1 + n % 6)); await page.keyboard.press('5'); await page.keyboard.press('Enter'); }
      return false;
    }, 180000);
    await shot('59-cc-results', 700);
  },
  'hex-hunt': async ({ page, shot, until, fits }) => {
    await page.click('[data-play="hex-hunt"]');
    await shot('60-hh-setup', 800);
    await fits('hex-hunt', '60-hh-setup');
    await page.click('#game-hex-hunt details.tnames summary');
    await shot('60b-hh-setup-names', 300); await fits('hex-hunt', '60b-hh-setup-names');
    await page.click('#hh-startBtn');
    await shot('61-hh-board', 500);
    await page.keyboard.press('ArrowRight');
    await shot('61b-hh-board-cursor', 200);
    await page.keyboard.press('Enter');
    await shot('62-hh-question-countdown', 600);
    await page.keyboard.press('Space'); await page.keyboard.press('Space');
    await until(async () => (await state(page, 'hex-hunt')).round === 'mark', 10000);
    await shot('63-hh-sides', 300);
    await page.keyboard.press('2');
    await shot('64-hh-claimed', 600);
    await page.keyboard.press('Enter');
    let k = 0;
    await until(async () => {
      const s = await state(page, 'hex-hunt');
      if (s.phase === 'won') return true;
      if (s.phase === 'board' || s.round === 'done') { if (k === 4 && s.phase === 'board') await shot('65-hh-board-midgame', 200); await page.keyboard.press('Enter'); }
      else if (s.round === 'think' || s.round === 'show') await page.keyboard.press('Space');
      else if (s.round === 'mark') { k++; await page.keyboard.press(String(1 + (k % 2))); }
      return false;
    }, 180000);
    await shot('66-hh-round-won', 900);
    await page.keyboard.press('Enter');
    await shot('67-hh-results', 700);
  },
  // the Start game button that waits after setup, in every game
  'start-gate': async ({ page, shot }) => {
    for (const [id, n] of [['over-the-edge', '70-ote'], ['outpace', '71-op'], ['category-clash', '72-cc'], ['hex-hunt', '73-hh']]) {
      await page.click(`[data-play="${id}"]`, { timeout: 60000 });
      await page.click(`#game-${id} .setup-go .btn`);
      await shot(`${n}-start-game`, 1200);
      // the Start game card sits over the dimmed game on purpose: it must be whole, on screen and below the top bar
      const g = await page.evaluate(i => { const c = document.querySelector(`#game-${i} .start-gate-card`).getBoundingClientRect(), bar = document.querySelector(`#game-${i} .topbar`).getBoundingClientRect(); return { top: c.top, bottom: c.bottom, left: c.left, right: c.right, bar: bar.bottom, w: innerWidth, h: innerHeight }; }, id);
      expect(g.top >= g.bar && g.bottom <= g.h && g.left >= 0 && g.right <= g.w, `${n} Start game card on screen`).toBe(true);
      await page.click(`#game-${id} .start-gate-btn`);
      await page.waitForTimeout(400);
      await page.keyboard.press('Escape'); const leave = page.locator('#leaveConfirm'); if (await leave.isVisible()) await leave.click();
    }
  }
};

for (const size of SIZES) {
  for (const [section, run] of Object.entries(SECTIONS)) {
    test(`screens at ${size.name}${SUFFIX}: ${section}`, async ({ browser }) => {
      test.setTimeout(section === 'launcher' ? 90000 : 420000);
      const ctx = await browser.newContext({ viewport: { width: size.width, height: size.height } });
      const page = await ctx.newPage();
      if (VARIANT === 'reduced') await page.addInitScript(() => { try { localStorage.setItem('cgb.settings', JSON.stringify({ reducedMotion: true })); } catch (e) { /* no storage */ } });
      const log = await openBundle(page, '', { quality: VARIANT === 'low' ? 'low' : 'high', gate: section === 'start-gate' });
      const problems = [];
      async function shot(name, settle) {
        await page.waitForTimeout(settle == null ? 600 : settle);
        await page.screenshot({ path: path.join(OUT, size.name + SUFFIX, name + '.png'), timeout: 60000 });
        (await findProblems(page)).forEach(p => problems.push(`${name}: ${p}`));
        // Over the Edge: the host must never cover the machine (shelf front edge, lanes, peg board)
        const ote = await page.evaluate(() => {
          const sec = document.getElementById('game-over-the-edge');
          if (!sec || sec.hidden || !CGB.games['over-the-edge'].layout) return null;
          const host = document.getElementById('ote-host');
          if (host.classList.contains('away') || host.classList.contains('nofit') || getComputedStyle(host).opacity < 0.15) return null;
          return CGB.games['over-the-edge'].layout();
        });
        if (ote) {
          const m = ote.machine, h = ote.host;
          const ox = Math.min(m.right, h.right) - Math.max(m.left, h.left), oy = Math.min(m.bottom, h.bottom) - Math.max(m.top, h.top);
          if (ox > 0 && oy > 0) problems.push(`${name}: host overlaps the machine (${Math.round(ox)}×${Math.round(oy)}px)`);
        }
        // the 3D scene really is drawn (not lost in fog or a blank canvas): enough lit pixels in the stage
        const stage = await page.evaluate(() => {
          const el = ['#game-over-the-edge .ote-stage', '#game-outpace'].map(s => document.querySelector(s)).find(e => e && e.offsetParent !== null && !e.closest('[hidden]'));
          if (!el) return null;
          const live = document.querySelector('#game-outpace .op-hud.active') || document.querySelector('#game-over-the-edge:not([hidden]) #ote-home.hidden');
          if (!live) return null;   // setup and results cards cover the scene
          const r = el.getBoundingClientRect(); return { x: r.left, y: r.top + r.height * 0.25, width: r.width, height: r.height * 0.45 };
        });
        if (stage && stage.width > 50) {
          // measure the canvas alone: hide the page's own panels, tags and bubbles for this one capture
          await page.addStyleTag({ content: '.game > :not(.ote-app):not(.op-stage), .ote-app > :not(.ote-stage), .ote-stage > :not(canvas), .op-stage > :not(canvas) { visibility: hidden !important; }' }).then(h => h.evaluate(el => { el.id = 'scene-check'; }));
          const share = brightShare(await page.screenshot({ clip: stage, timeout: 60000 }), 70);
          await page.evaluate(() => document.getElementById('scene-check').remove());
          if (share < 0.006) problems.push(`${name}: the 3D scene looks blank (${(share * 100).toFixed(2)}% lit pixels)`);
        }
        // the launcher's "More shows coming soon" card is either fully on screen or starts below the fold
        const soon = await page.evaluate(() => {
          const el = document.querySelector('#launcher:not([hidden]) .gcard.soon'); if (!el) return null;
          const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right };
        });
        if (soon && soon.top < size.height && (soon.bottom > size.height + 1 || soon.left < 0 || soon.right > size.width)) {
          problems.push(`${name}: the coming-soon card is cut off (${Math.round(soon.top)}–${Math.round(soon.bottom)} of ${size.height})`);
        }
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
