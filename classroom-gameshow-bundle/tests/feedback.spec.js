// Fixes from Rob's classroom feedback (version 1.6.0): no second Start game step, the Final Sprint's own Start button,
// Over the Edge landings and the single question, Hex Hunt repeats and layout, cards below the top bar.
const { test } = require('@playwright/test');
const { openBundle, state, mark, playHexHunt, afterResults, expect } = require('./helpers');

test('the Final Sprint waits on its own "Start the Final Sprint" button in every subject, Maths included', async ({ page }) => {
  await openBundle(page);
  for (const [sj, clock] of [['biology', '1:30'], ['maths', '5:00']]) {
    await page.selectOption('#subjectSelect', sj);
    await page.click('[data-play="outpace"]');
    await page.click('#op-startBtn');
    await page.evaluate(() => CGB.test.outpace.toSprint(600));
    const card = page.locator('#game-outpace .sprint-gate:not([hidden])');
    await expect(card.locator('button')).toContainText('Start the Final Sprint');
    await expect(card.locator('button .kbd')).toHaveText('Enter');
    await expect(page.locator('#op-sprintTimer')).toHaveText(clock);
    await page.waitForTimeout(1500);
    await expect(page.locator('#op-sprintTimer')).toHaveText(clock);        // the clock has not started
    await page.keyboard.press('Enter');
    await expect(card).toHaveCount(0);
    await expect.poll(() => page.locator('#op-sprintTimer').textContent()).not.toBe(clock);
    await page.evaluate(() => CGB.app.show('launcher'));
  }
});

test('Over the Edge: the question shows once, in the right-hand panel; there is no big box over the machine', async ({ page }) => {
  await openBundle(page, '#over-the-edge');
  await page.click('#ote-startBtn');
  await expect.poll(async () => (await state(page, 'over-the-edge')).round).toBe('think');
  const q = (await state(page, 'over-the-edge')).q.q;
  await expect(page.locator('#ote-bigq')).toHaveCount(0);
  await expect(page.locator('#ote-qText')).toHaveText(q);
  expect(await page.getByText(q, { exact: true }).locator('visible=true').count()).toBe(1);
});

test('Over the Edge: a counter that lands where there is space always lands flat in it; the landing involves no chance', async ({ page }) => {
  test.setTimeout(240000);
  await openBundle(page, '#over-the-edge');
  await page.click('#ote-startBtn');
  await expect.poll(async () => (await state(page, 'over-the-edge')).round).toBe('think');
  await page.evaluate(() => CGB.test.ote.manual(true));
  const r = await page.evaluate(() => {
    let checked = 0, withRoom = 0, onTop = [], varied = 0;
    const T = CGB.test.ote;
    for (let step = 0; step < 160; step++) {
      T.advance(0.05 + (step * 0.037) % 1.1);                    // different moments of the shelf's movement
      for (let x = 20; x <= 460; x += 26) {
        const a = T.landing(x), b = T.landing(x), c = T.landing(x);
        if (a.x !== b.x || a.y !== b.y || a.onto !== b.onto || b.x !== c.x || b.onto !== c.onto) varied++;   // same answer every time
        checked++;
        if (T.gapFree(x)) { withRoom++; if (a.onto) onTop.push([step, x]); }
      }
    }
    return { checked, withRoom, onTop, varied };
  });
  expect(r.checked).toBeGreaterThan(2000);
  expect(r.withRoom).toBeGreaterThan(200);                      // plenty of moments with space on the shelf
  expect(r.varied).toBe(0);
  expect(r.onTop).toEqual([]);                                  // never on top of another counter when there was room
});

test('Hex Hunt: a second match on the same topic mostly asks questions the class has not seen', async ({ page }) => {
  test.setTimeout(480000);
  await openBundle(page, '#hex-hunt');
  await page.evaluate(() => { CGB.bank.setSubject('biology'); CGB.bank.setBoard('aqa'); CGB.bank.setCourse('separate'); CGB.bank.setSelection({ ids: ['aqa-biology-8461-4.1'] }); });
  const asked = () => page.evaluate(() => { const m = (CGB.store.getJSON('hh.recent', {}) || {})['biology|aqa|separate']; return m ? m.q : {}; });
  async function match() {
    await page.click('#hh-startBtn');
    const s = await playHexHunt(page, { picks: ['1'] });
    expect(s.phase).toBe('summary');
    await afterResults(page);
    await page.evaluate(() => CGB.app.show('launcher'));
    await page.click('[data-play="hex-hunt"]');
  }
  await match();
  const first = Object.keys(await asked());
  await match();
  const all = await asked(), second = Object.keys(all).filter(k => all[k] > first.length);
  expect(first.length).toBeGreaterThanOrEqual(8);
  expect(second.length).toBeGreaterThanOrEqual(8);
  const repeats = second.filter(k => first.includes(k)).length;
  expect(repeats / second.length, `${repeats} of ${second.length} repeated`).toBeLessThanOrEqual(0.1);
});

test('Hex Hunt: whose turn it is is large, below the top bar, and the top bar stays across the top', async ({ page }) => {
  for (const [w, h] of [[960, 600], [1366, 768], [1920, 1080]]) {
    await page.setViewportSize({ width: w, height: h });
    await openBundle(page, '#hex-hunt');
    await page.click('#hh-startBtn');
    await expect(page.locator('#hh-turn')).toContainText('pick a hexagon');
    const r = await page.evaluate(() => {
      const box = e => { const b = e.getBoundingClientRect(); return { l: b.left, t: b.top, r: b.right, b: b.bottom }; };
      const bar = Array.from(document.querySelectorAll('#game-hex-hunt .topbar .btn')).filter(b => b.offsetParent).map(box);
      return { bar, turn: box(document.getElementById('hh-turn')), size: parseFloat(getComputedStyle(document.getElementById('hh-turn')).fontSize) };
    });
    expect(new Set(r.bar.map(b => Math.round(b.t))).size, `${w}: one row`).toBe(1);
    expect(r.size, `${w}: turn text size`).toBeGreaterThanOrEqual(36);
    for (const b of r.bar) expect(b.b <= r.turn.t + 1 || b.r <= r.turn.l || b.l >= r.turn.r, `${w}: top bar clear of the turn text`).toBe(true);
    await page.reload();
  }
});

for (const [w, h] of [[640, 800], [960, 600], [1366, 768], [1920, 1080]]) {
  test(`question cards stay below the top bar buttons at ${w}×${h}: Hex Hunt, Category Clash and Outpace`, async ({ page }) => {
    await page.setViewportSize({ width: w, height: h });
    await openBundle(page, '#hex-hunt');
    const clear = async (game, card) => {
      await page.waitForTimeout(1200);                         // the card grows from its tile: measure it where it settles
      const r = await page.evaluate(([g, c]) => {
        const bar = Array.from(document.querySelectorAll(`#game-${g} .topbar .btn`)).filter(b => b.offsetParent).map(b => b.getBoundingClientRect());
        const k = document.querySelector(c); if (!k || !k.offsetParent) return null;
        const q = k.getBoundingClientRect();
        return { bar: bar.map(b => [b.left, b.top, b.right, b.bottom]), card: [q.left, q.top, q.right, q.bottom] };
      }, [game, card]);
      expect(r, `${game} question card is showing`).not.toBeNull();
      for (const b of r.bar) expect(b[2] <= r.card[0] || b[0] >= r.card[2] || b[3] <= r.card[1] || b[1] >= r.card[3], `${game}: a top bar button overlaps the question card`).toBe(true);
    };
    // Hex Hunt
    await page.click('#hh-startBtn');
    await page.locator('#hh-board .hh-hex').first().click();
    await expect.poll(async () => (await state(page, 'hex-hunt')).round).toBe('think');
    await clear('hex-hunt', '#hh-qcard');
    // Category Clash
    await page.evaluate(() => CGB.app.show('category-clash'));
    await page.click('#cc-startBtn');
    await page.locator('#cc-board .cc-tile[data-c]').first().click();
    await expect.poll(async () => (await state(page, 'category-clash')).round).toBe('think');
    await clear('category-clash', '#cc-qcard');
    // Outpace: the Deal Round question card
    await page.evaluate(() => CGB.app.show('outpace'));
    await page.click('#op-startBtn');
    await page.keyboard.press('3');
    await expect.poll(async () => (await state(page, 'outpace')).round).toBe('think');
    await clear('outpace', '#op-qcard');
  });
}
