# Showtime: Classroom Gameshows

Four revision quiz games for the front of the classroom, **Over the Edge**, **Outpace**, **Category Clash** and **Hex Hunt**, packaged as one offline HTML file with a shared launcher, a shared question bank and a host mascot.

- Product: [`dist/classroom-gameshow-bundle.html`](dist/classroom-gameshow-bundle.html). Double-click it: no server, no internet connection.
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
    bank.js                  question bank: sets, active set, picker, wrong-answer history, backup
    sfx.js                   Web Audio synthesised sound effects (no audio files)
    host.js                  Professor Pip, the customisable 2D (SVG) host and mascot
    brand.js                 wordmark, game logos and launcher art, drawn as SVG
    ui.css                   shared UI kit: tokens, buttons, panels, score cards, banners, modals, launcher
    app.js                   launcher, routing (#over-the-edge, #outpace), bank manager, settings, about
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
node build.js            # writes dist/classroom-gameshow-bundle.html
```

The build inlines every `<script src>`, every stylesheet (turning the font files into base64 data URIs), the game HTML fragments (`<!-- @include ... -->`) and the licence texts (`<!-- @text ... -->`). It stops with an error if the result refers to anything external. The page also carries a Content-Security-Policy that blocks all network connections.

For development, edit files in `src/`, run `node build.js` and refresh the dist file in the browser.

## Tests and release checks

```bash
npm install                       # Playwright test runner only
npm test                          # build + all browser tests (about 10 minutes with software rendering)
npx playwright test tests/bundle.spec.js tests/bank.spec.js tests/boardgames.spec.js    # the functional tests only
node tools/make-listing.js        # dist/listing/*.png (and docs/mascot.png for the guide)
node tools/make-guide.js          # dist/teacher-guide.pdf
node tools/check-banned.js        # searches dist, guide and listing text for terms we must not use
node tools/check-contrast.js      # WCAG AA contrast for every colour pair in the UI
```

The tests open the built file from `file://` and check: zero network requests and console errors; launcher to each game and back; a full keyboard-only game of each game to the summary; a pasted set appearing in both games; JSON backup export and re-import with no loss; wrong answers from one game appearing in the other; working with storage blocked. `tests/screens.spec.js` takes screenshots of every screen at 1920×1080, 1366×768 and 768×1024 (12 batches, each with its own time limit) into `screenshots/` and fails if banners, speech bubbles, floating labels or panels overlap.

## Question format

```
Subject: Chemistry
Topic: Bonding
Q: What type of bonding is in sodium chloride?
A: Ionic
```

Subject and Topic apply until changed. An optional `Difficulty: 1`–`5` line sets Category Clash points for the questions below it. This is the same format the original games used, so existing sets paste straight in.

## Data and privacy

Everything is stored in the browser's `localStorage` under keys starting `cgb.`. Nothing is sent anywhere. Backups are JSON files the teacher saves and loads by hand.
