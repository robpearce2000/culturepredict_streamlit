# Showtime: Classroom Gameshows

Four revision quiz games for the front of the classroom, **Over the Edge**, **Outpace**, **Category Clash** and **Hex Hunt**, packaged as one offline HTML file with a shared launcher, a shared question bank and a host mascot.

Teachers choose their subject, exam board and question set once on the main screen. In every game 2 to 6 teams answer every question on mini whiteboards, the teacher marks each team with the number keys, and play moves on; the only setup choice is the number of teams.

- Product: [`dist/showtime-classroom-gameshows.html`](dist/showtime-classroom-gameshows.html). Double-click it: no server, no internet connection.
- Teacher guide: [`dist/teacher-guide.pdf`](dist/teacher-guide.pdf)
- Tes listing images and text: [`dist/listing/`](dist/listing/)
- Why things are the way they are: [`DECISIONS.md`](DECISIONS.md). What changed: [`CHANGELOG.md`](CHANGELOG.md). Third-party licences: [`LICENSES.md`](LICENSES.md).

## Source layout

```
src/
  index.html                 launcher shell, modals; build markers pull everything else in
  fonts/                     Lilita One and Nunito (woff2, SIL OFL) + fonts.css + licences
  vendor/                    three.js r128 and the post-processing scripts used by the 3D games
  shared/
    storage.js               localStorage wrapper (every call in try/catch; memory fallback)
    settings.js              sound, text size, reduced motion, graphics quality
    packs.js                 built-in question packs in the plain-text format
    bank.js                  question bank: subjects and exam boards, sets, set in use, picker, wrong-answer history, backup
    sfx.js                   Web Audio synthesised sound effects (no audio files)
    host.js                  Marty Marquee, the customisable 2D (SVG) host, mascot and corner host
    brand.js                 wordmark, game logos and launcher art, drawn as SVG
    ui.css                   shared UI kit: tokens, buttons, panels, score cards, banners, modals, launcher
    app.js                   launcher (subject panel, host editor), routing, bank manager, settings, about, setup-card fitting
    teams.js                 the team palette (six teams) and shared team names
    classmode.js             the class round: countdown, "show me", team marking board, undo, misconceptions
  games/
    over-the-edge/           game.html, game.css, game.js
    outpace/                 game.html, game.css, game.js
    category-clash/          game.html, game.css, game.js (2D, DOM)
    hex-hunt/                game.html, game.css, game.js (2D, SVG)
build.js                     makes the single file
docs/teacher-guide.html      source of the PDF guide
tools/                       listing images, guide PDF, banned-term and contrast checks
tests/                       Playwright tests
```

Each game registers itself with `CGB.registerGame(id, { init, enter, exit, inProgress })`. It is created the first time it is opened, its render loop only runs while it is on screen, and leaving resets it to its setup card.

## Building

Needs Node 18 or later. The build itself has no dependencies.

```bash
node build.js            # writes dist/showtime-classroom-gameshows.html
```

The build inlines every `<script src>`, every stylesheet (turning the font files into base64 data URIs), the game HTML fragments (`<!-- @include ... -->`) and the licence texts (`<!-- @text ... -->`). It stops with an error if the result refers to anything external. The page also carries a Content-Security-Policy that blocks all network connections.

For development, edit files in `src/`, run `node build.js` and refresh the dist file in the browser.

## Tests and release checks

```bash
npm install                       # Playwright test runner only
npm test                          # everything before a push: gameplay tests, two High-graphics playthroughs and the screenshot sweep
npm run test:quick                # gameplay tests only (about a minute)
npm run test:screens              # the screenshot sweep only
SEED=7 npm run test:quick         # the same tests with another random sequence
node tools/make-listing.js        # dist/listing/*.png (and docs/mascot.png for the guide)
node tools/make-guide.js          # dist/teacher-guide.pdf
node tools/check-banned.js        # searches dist, guide and listing text for terms we must not use
node tools/check-contrast.js      # WCAG AA contrast for every colour pair in the UI
```

Playwright runs `build.js` first; it does nothing if `src/` hasn't changed since the last build.

**Two builds.** `dist/showtime-classroom-gameshows.html` is the product. `test-build/showtime-test.html` (not committed) is the same file plus the code between `@test-only` and `@end-test-only` markers: a seedable random number generator and shortcuts such as "jump to the final round", "jackpot near the edge", a manual physics clock and setting the sprint timer. The build removes those blocks from the product and stops if any are left; a test checks the product is exactly the test build without them.

**What the tests cover.** The shipped file loads from `file://` with zero network requests and console errors, and the launcher reaches every game and back. Full keyboard games of all four games, played with the same seed every run. The setup cards offer only the number of teams; the subject and exam board choice filters the sets everywhere and is remembered; the host is customised from the main screen (`tests/setup.spec.js`). Every game's 20-second countdown. Category Clash difficulty rows. Whole-class games of every game with 2, 4 and 6 teams (Hex Hunt with two halves), checking marking, undo, catch-up rules, the misconceptions summary and that Over the Edge's counters finish dropping within 10 seconds of marking (`tests/classmode.spec.js`). The jackpot win. Over the Edge physics giving identical results on two runs. A pasted set appearing in every game, JSON backup round trips, shared wrong-answer history and weak topics coming up more often, and blocked storage. Gameplay tests use low graphics and a 960×600 window. One full-length Over the Edge game and one full Outpace game run at High graphics on the shipped file (tagged `@hq`, in `npm test` only).

`tests/screens.spec.js` takes screenshots of every screen at 1920×1080, 1366×768 and 768×1024 (batches with their own time limits) into `screenshots/`. Every game is shown with six teams (two halves in Hex Hunt). It fails if:
- banners, bubbles, labels, captions or panels overlap;
- a setup card needs scrolling or hides under the top bar;
- the Over the Edge host overlaps the machine;
- the coming-soon strip is cut off;
- a team name is cut off on a team panel;
- a card or panel block has text spilling out of it;
- a 3D scene comes out blank.

## Question format

```
Subject: Chemistry
Topic: Bonding
Q: What type of bonding is in sodium chloride?
A: Ionic
```

Subject and Topic apply until changed. An optional `Tier: 1`, `2` or `3` line applies to the questions below it (1 recall, 2 describe and apply, 3 explain and extend; see DECISIONS.md); Category Clash puts tiers 1, 2 and 3 on its 100, 200 and 300 rows. Every built-in question is tagged. Older `Difficulty: 1`–`5` lines still work. This is the same format the original games used, so existing sets paste straight in.

## Data and privacy

Everything is stored in the browser's `localStorage` under keys starting `cgb.`. Nothing is sent anywhere. Backups are JSON files the teacher saves and loads by hand.
