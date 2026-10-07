// Outpace: winning as soon as the target is reached, the subject looks, and the frame rate.
const { test } = require('@playwright/test');
const { openBundle, state, mark, PICK, expect } = require('./helpers');
const note = (name, value) => test.info().annotations.push({ type: name, description: String(value) });

test('the Final Sprint is won as soon as the target is reached, after a short undo window', async ({ page }) => {
  const log = await openBundle(page, '#outpace');
  await page.click(PICK('#op-segGroups', 4));
  await page.click('#op-startBtn');
  await page.evaluate(() => CGB.test.outpace.toSprint(600));          // target: 7 steps for 4 teams on a Standard deal
  let s = await state(page, 'outpace');
  expect(s.target).toBe(7);
  expect(s.timeLeft).toBe(90);                                         // a 90-second sprint
  await mark(page, 'outpace', () => true);                             // 4
  await expect.poll(async () => (await state(page, 'outpace')).round, { timeout: 5000 }).toBe('think');
  await mark(page, 'outpace', i => i < 2);                             // 6
  await expect.poll(async () => (await state(page, 'outpace')).round, { timeout: 5000 }).toBe('think');
  await mark(page, 'outpace', i => i < 2);                             // 8: target reached
  s = await state(page, 'outpace');
  expect(s.net).toBe(8); expect(s.frozen).toBe(true);
  // the clock has stopped
  const t0 = s.timeLeft; await page.waitForTimeout(600);
  expect((await state(page, 'outpace')).timeLeft).toBe(t0);
  // undo inside the window: the clock starts again and the sprint carries on
  await page.keyboard.press('u');
  s = await state(page, 'outpace');
  expect(s.net).toBe(6); expect(s.frozen).toBe(false); expect(s.phase).toBe('sprint');
  await page.waitForTimeout(500);
  expect((await state(page, 'outpace')).timeLeft).toBeLessThan(t0);
  // reach it again and let the window pass: the escape plays with time still on the clock
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await state(page, 'outpace')).frozen).toBe(true);
  const left = (await state(page, 'outpace')).timeLeft;
  expect(left).toBeGreaterThan(60);
  await expect.poll(async () => (await state(page, 'outpace')).phase, { timeout: 20000 }).toBe('summary');
  await expect(page.locator('#op-finalVerdict')).toContainText('Escaped');
  expect((await state(page, 'outpace')).timeLeft).toBe(left);
  expect(log.errors).toEqual([]);
});

test('the look follows the subject chosen on the main screen', async ({ page }) => {
  const log = await openBundle(page);
  const expected = { biology: 'science', chemistry: 'science', physics: 'science', maths: 'maths', history: 'history', geography: 'geography', other: 'general' };
  await page.click('[data-play="outpace"]', { timeout: 60000 });
  for (const [sj, look] of Object.entries(expected)) {
    await page.evaluate(sj => CGB.bank.setSubject(sj), sj);
    expect((await state(page, 'outpace')).look, sj).toBe(look);
  }
  // there is no look setting on the setup card
  await expect(page.locator('#op-setupCard')).not.toContainText(/look/i);
  // every look draws a race: play into the Deal Round in each and check the racers are on screen
  for (const sj of ['biology', 'maths', 'history', 'geography', 'other']) {
    await page.keyboard.press('Escape');
    await page.evaluate(sj => { CGB.bank.setSubject(sj); if (!CGB.bank.active()) CGB.bank.addSet('Look ' + sj, 'Subject: S\nTopic: T\nQ: One?\nA: Yes\nQ: Two?\nA: No', sj); }, sj);
    await page.click('[data-play="outpace"]');
    await page.click('#op-startBtn');
    await expect(page.locator('#op-tagYou')).toBeVisible();
    await expect(page.locator('#op-tagHunter')).toBeVisible();
    await page.keyboard.press('Escape'); await page.click('#leaveConfirm');
  }
  expect(log.errors).toEqual([]);
});

/* Frames per second in the Deal Round at 1366×768. The test machine draws with a software
   renderer, so the numbers are far below a real laptop's; they are recorded with the result,
   and the test checks the automatic step-down engages when frames are slow. */
for (const quality of ['low', 'high']) {
  test(`Outpace frame rate at ${quality} graphics (recorded)`, async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await openBundle(page, '#outpace', { quality });
    await page.click('#op-startBtn');
    await page.keyboard.press('2');
    await page.waitForTimeout(1500);
    const fps = await page.evaluate(() => new Promise(res => {
      let n = 0; const t0 = performance.now();
      const tick = () => { n++; if (performance.now() - t0 < 4000) requestAnimationFrame(tick); else res(n / ((performance.now() - t0) / 1000)); };
      requestAnimationFrame(tick);
    }));
    const s = await state(page, 'outpace');
    note(`fps at ${quality}`, fps.toFixed(1));
    note('automatic quality step-down level', s.perfLevel);
    console.log(`Outpace ${quality}: ${fps.toFixed(1)} fps (software renderer), step-down level ${s.perfLevel}`);
    // a sanity check only: with other test workers sharing the CPU the software renderer can drop to 2–3 fps
    expect(fps).toBeGreaterThan(1.5);
    // slow frames for a sustained spell step the quality down (at most two levels); under a
    // heavily loaded test machine the first frames come so slowly that this can take a few seconds more
    if (fps < 30) await expect.poll(async () => (await state(page, 'outpace')).perfLevel, { timeout: 15000 }).toBeGreaterThanOrEqual(1);
    expect((await state(page, 'outpace')).perfLevel).toBeLessThanOrEqual(2);
  });
}

/* Smoothness: on a perfectly steady 60 Hz clock, a move after marking must glide. The camera
   eases in and out (no sudden change of speed) and the picture never jumps to make room for the
   question panel (it used to step a few pixels every sixth frame, which read as jitter). */
test('the camera and picture glide smoothly through a move', async ({ page }) => {
  await page.addInitScript(() => {
    let vt = 0; const q = new Map(); let id = 0;
    const realNow = performance.now.bind(performance), origRaf = window.requestAnimationFrame.bind(window);
    let on = false;
    performance.now = () => on ? vt : realNow();
    window.requestAnimationFrame = cb => { if (!on) return origRaf(cb); const i = ++id; q.set(i, cb); return i; };
    window.cancelAnimationFrame = i => q.delete(i);
    window.__virtOn = () => { on = true; vt = realNow(); };
    window.__queued = () => q.size;
    window.__step = n => { const out = []; for (let k = 0; k < n; k++) { vt += 1000 / 60; const cbs = [...q.values()]; q.clear(); cbs.forEach(cb => cb(vt)); out.push(CGB.test.outpace.motion()); } return out; };
  });
  await page.setViewportSize({ width: 1366, height: 768 });
  await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  await page.keyboard.press('2');
  await page.waitForTimeout(1500);
  await page.evaluate(() => window.__virtOn());
  await page.waitForFunction(() => window.__queued() > 0, null, { polling: 50 });   // the game's loop now runs on the steady clock
  let rec = await page.evaluate(() => window.__step(60));
  await page.keyboard.press('Space'); rec = rec.concat(await page.evaluate(() => window.__step(40)));   // (a second Space within half a second is a repeat)
  await page.keyboard.press('Space'); rec = rec.concat(await page.evaluate(() => window.__step(5)));
  for (let i = 0; i < 4; i++) await page.keyboard.press(String(i + 1));
  await page.keyboard.press('Enter');
  rec = rec.concat(await page.evaluate(() => window.__step(180)));
  const d = f => rec.slice(1).map((r, i) => f(r) - f(rec[i]));
  const camMoved = Math.max(...d(r => Math.hypot(r.cam[0], r.cam[2])).map(Math.abs));
  expect(camMoved).toBeGreaterThan(0.003);                          // the camera did follow the class
  const shiftSteps = d(r => r.shift[0]).map(Math.abs);
  expect(Math.max(...shiftSteps)).toBeLessThan(3.5);                 // the picture slides (a few px a frame), never jumps
  const vel = d(r => r.cam[0]), acc = vel.slice(1).map((v, i) => Math.abs(v - vel[i]));
  expect(Math.max(...acc)).toBeLessThan(0.02);                       // the camera's speed changes gradually
});

test('Outpace: the Final Sprint waits on a card that explains it, and only starts the clock when Start is pressed', async ({ page }) => {
  const log = await openBundle(page, '#outpace', { gate: true });
  await page.click('#op-startBtn');
  await page.click('#game-outpace .start-gate:not(.sprint-gate) .start-gate-btn');   // the Start game card
  await page.evaluate(() => CGB.test.outpace.toSprint(600));
  const card = page.locator('#game-outpace .sprint-gate:not([hidden])');
  await expect(card).toContainText('Final Sprint');
  await expect(card).toContainText('1 minute 30 seconds');
  await expect(card.locator('button')).toContainText('Start the Final Sprint');
  await expect(page.locator('#op-sprintCard')).toBeHidden();
  const t0 = (await state(page, 'outpace')).timeLeft;
  await page.waitForTimeout(1200);
  expect((await state(page, 'outpace')).timeLeft).toBe(t0);         // nothing ticks yet
  await page.keyboard.press('Enter');
  await expect(card).toHaveCount(0);
  await expect(page.locator('#op-sprintCard')).toBeVisible();
  await expect.poll(async () => (await state(page, 'outpace')).timeLeft).toBeLessThan(t0);
  expect(log.errors).toEqual([]);
});

test('Outpace: the class and Hunter labels sit over their own racers, in the Deal Rounds and in the Final Sprint', async ({ page }) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  const log = await openBundle(page, '#outpace');
  await page.click('#op-startBtn');
  const gaps = () => page.evaluate(() => {
    const m = CGB.test.outpace.motion();
    const c = id => { const b = document.getElementById(id).getBoundingClientRect(); return b.left + b.width / 2; };
    return [Math.abs(c('op-tagYou') - m.rs[0]), Math.abs(c('op-tagHunter') - m.hs)];
  });
  await page.waitForTimeout(2500);
  (await gaps()).forEach(g => expect(g).toBeLessThan(30));
  await page.evaluate(() => CGB.test.outpace.toSprint(600));          // the picture slides left of the question column
  await page.waitForTimeout(3000);
  (await gaps()).forEach(g => expect(g).toBeLessThan(30));
  expect(log.errors).toEqual([]);
});
