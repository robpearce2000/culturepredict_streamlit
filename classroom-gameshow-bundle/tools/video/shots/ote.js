/* Shot "ote": Over the Edge, one class question: the question large with its countdown, "3, 2, 1, show me!",
   six teams marked, a team's counter dropping and pushing counters off the edge. Seed and script below. */
const { open, Shot } = require('../harness.js');
const SEED = +(process.env.SEED || 20261004);
const TEAMS = ['Team Electrons', 'Team Neutrons', 'Team Protons', 'Team Photons', 'Team Ions', 'Team Atoms'];
(async () => {
  const { browser, page } = await open({ seed: SEED, hash: '#over-the-edge' });
  await page.setViewportSize({ width: 960, height: 600 });
  await page.evaluate(() => { CGB.bank.setBoard('aqa'); CGB.bank.setSubject('biology'); CGB.bank.setSelection({ ids: ['aqa-biology-8461-4.1'] }); });
  await page.click('#ote-segTeams button[data-v="6"]');
  const box = page.locator('#game-over-the-edge .setup details.tnames').first();
  if (!(await box.evaluate(d => d.open))) await box.locator('summary').click();
  for (let i = 0; i < 6; i++) await page.fill('#ote-cname' + i, TEAMS[i]);
  await page.click('#ote-startBtn');
  await page.waitForFunction(() => CGB.games['over-the-edge'].state().step === 'ask', null, { timeout: 120000 });
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.waitForTimeout(3000);
  console.log('question:', await page.evaluate(() => CGB.games['over-the-edge'].state().q));
  const s = new Shot(page, 'ote'); await s.start();
  await s.caption('Every student answers every question');
  s.mark('think'); await s.run(2.6);
  await s.key('Space'); await s.run(0.4);                      // "3, 2, 1, show me!"
  s.mark('show'); await s.run(2.4);
  await s.caption('Mark the whole class in seconds');
  await s.run(0.7);
  s.mark('mark');
  for (const k of ['1', '2', '4', '5']) { await s.key(k); await s.run(0.35); }
  await s.run(0.6);
  await s.key('Space'); await s.run(0.8);                       // confirm
  await s.caption('Over the Edge');
  s.mark('drops');
  for (let i = 0; i < 40 && (await page.evaluate(() => CGB.games['over-the-edge'].state().step)) !== 'lanes'; i++) await s.run(0.25);
  await s.key('r');                                             // the captains' lanes
  await s.run(7);
  s.mark('end');
  console.log('frames', await s.finish());
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
