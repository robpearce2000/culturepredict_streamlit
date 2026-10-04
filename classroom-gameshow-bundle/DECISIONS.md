# Decisions

Choices made while building the first edition without the owner available. Each has a one-line reason. Anything here can be changed.

## Names and branding

| Decision | Reason |
|---|---|
| **Game 2 is called Outpace**, not the working title in the brief. | The working title is the name of a long-running SEGA racing video game series, so it clashes with an existing trade mark in the same class (game software). "Outpace" keeps the meaning and found no clash in a web search. |
| **Game 1 keeps the name Over the Edge.** | A web search found no UK classroom quiz product with this name. A 1990s tabletop role-playing game uses it, but in a different product category. |
| **Product renamed Showtime: Classroom Gameshows** (owner's instruction during the build; replaces the briefed bundle name everywhere a user sees it). The wordmark is "SHOWTIME" in a marquee with a "CLASSROOM GAMESHOWS" ribbon. | Owner's choice. A web search found no classroom quiz product with this name, but SHOWTIME is a well-known US TV network brand, so always use the full name and run a UK IPO check (classes 9, 28, 41) before selling. |
| The shipped file is `dist/showtime-classroom-gameshows.html` (renamed in 1.2). The backup format id stays `classroom-gameshow-bundle-backup`. | The file name now matches the product name. The id keeps old backups loading. Backups download as `showtime-backup-<date>.json`. |
| Show-specific terms replaced: "lanes 1–4", "wildcard counters" (marked ★), "the Hunter", "Deal Round" with Cautious / Standard / Bold deals, "Final Sprint", "pass" (was "help"). | Brief's suggested terms, plus deal names that say what they do. |
| The "Lab grant" wildcard is now "Cash bonus". | Works for any subject, not just science. |
| Over the Edge set redesigned: deep-teal studio, aqua and coral light strips, sunflower shelves with a hazard-stripe front edge, a dark slate peg board with brass pegs, an "OVER THE EDGE" header sign. Geometry, tiers, peg layout and every physics constant are unchanged. | Removes the lilac-and-gold set, red shelves, white board and header sign of the TV look. The physics code was diffed against v1: the only change is the field name `mystery` → `wildcard`. |
| Outpace runner is **gold** and the Hunter is **magenta**, on a deep-indigo track with a teal home cell. | The original blue-versus-red pairing echoed the TV board. Gold/magenta also stays distinct for red–green colour blindness. Atom visuals kept, as briefed. |
| Default host named **Professor Pip** (renamed **Marty Marquee** in 1.2, see below) with curly ginger hair, glasses, a teal suit and an orange bow tie (new). | The previous default carried the owner's name and a generic look; the bow tie gives the mascot an identity that is not modelled on any presenter. All customisation options kept. |
| Showtime identity: midnight navy, tangerine, sunflower and teal; wordmark built as SVG in code; display font Lilita One, body font Nunito. | Bright, readable from the back of a room, fully offline. |
| Fonts embedded as base64 woff2 (~92 KB) rather than system fonts. | Gives the same look on every school PC; both fonts are SIL OFL. |
| Marketing and guide say "inspired by classic TV quiz formats" and never name a show. | As briefed. |
| Credits read "Game design and question packs by Rob Pearce". | The games are Rob's; remove from the About panel (`src/index.html`) and guide if he prefers. |

## Rules and gameplay

| Decision | Reason |
|---|---|
| No rule, scoring or physics changes in either game. | Out of scope for this edition. |
| Over the Edge keeps £ prize values; Outpace keeps points. | These are part of each game's existing rules. |
| **Outpace gained keyboard shortcuts** (1/2/3 deals, A or Space answer, C/W mark, H pass, Enter continue, Esc menu). | The brief asks for both games to be fully usable by keyboard; v1 of Outpace had no shortcuts. Keys mirror Over the Edge. |
| Outpace still requires "Show answer" before Correct/Wrong. | Unchanged v1 behaviour; keeps the answer visible to the class before marking. |
| Outpace gained the "pick more questions from weak topics" option. | It is a shared-bank feature now, not a rule change; off by default. |
| Outpace now logs wrong answers the moment they happen (v1 saved them at the end). | Needed so history is shared live between games and is not lost if a game is abandoned. |
| Outpace gained short synthesised sound effects (deal picked, caught, escaped, time up). No countdown ticking. | v1 was silent; sounds are original and avoid anything that imitates a TV format. |
| Outpace's catch flash is softer (45 % instead of 95 % opacity) and is off with Reduced motion. | Photosensitivity. |
| Enter starts a game from the setup card and plays again from the results. | Faster to run from the keyboard. |
| Leaving a game in progress asks for confirmation; returning to a game starts from its setup card. | Avoids losing a game by accident, and keeps state simple. |

## Question bank and data

| Decision | Reason |
|---|---|
| One active set shared by both games. | As briefed; changing it in either game or on the menu changes it everywhere. |
| Built-in packs: GCSE Combined Science starter pack (Biology 65, Chemistry 68, Physics 58, plus an all-in-one mix of 191) and Homeostasis L1 (61). | The brief's AQA-style pack, building on the v1 mixed set (all 80 of its questions are included). The v1 Homeostasis set was identical in both games, so it appears once. |
| Starter pack topics use AQA specification topic names, so a few v1 topic labels changed (e.g. "Inheritance" → "Inheritance, variation and evolution"). | Matches teachers' schemes of work; old history entries keep their old labels. |
| Built-in packs are read-only; **Copy to edit** makes an editable copy. | Updates to the product can then improve built-in packs without overwriting a teacher's work. |
| Bank manager also offers **Edit questions** for a teacher's own sets (shows the set in the paste format). | Rename and delete alone made fixing a typo awkward. |
| Backup import **merges** and never deletes. Sets match by id; history entries are counted, so importing the same file twice adds nothing. | Safe to use on a shared machine; round-trips exactly (tested). |
| History is keyed by player name, ignoring capitals and surrounding spaces, capped at 300 entries per player. | Same rule as v1. |
| Storage keys start with `cgb.`. On first run, data saved by the two stand-alone v1 games in the same browser (their sets, history, host and names) is imported once. | Teachers who used v1 on the same computer keep their work. |
| If storage is blocked the bundle keeps data in memory for the session and says so on the menu and when saving. | The page must still work. |
| Player names are remembered and shared between games. | So the same names (and history) carry across. |

## Accessibility and performance

| Decision | Reason |
|---|---|
| Every text colour pair checked against WCAG AA (`tools/check-contrast.js`, 42 pairs). Button colours were darkened from v1 (v1's green and red buttons were below 4.5:1 with white text). | Brief requirement. |
| Nothing relies on colour alone: players marked ● and ■, wildcard ★, labelled runner/Hunter tags, ✓/✗ on buttons, "◀ turn" markers, selected toggles show ✓. | Colour-blind pupils. |
| Text size Large scales everything by 1.1875 (panels are sized in rem). | Back-of-room reading. |
| Reduced motion defaults to the operating-system setting. | Respect existing preferences. |
| Graphics defaults to Low on devices reporting ≤ 2 CPU cores or ≤ 2 GB memory. Low = no shadows, no bloom, pixel ratio 1. | Older classroom PCs. |
| Render loops stop when a game is not on screen; the launcher mascot pauses inside games. | Smooth frame rate and less fan noise. |
| Outpace's camera now frames both atoms at any aspect ratio. | v1 could push an atom off-screen on a portrait tablet. |
| Outpace's HUD is a top area and a bottom dock in one flex column. | v1 positioned the help buttons at a fixed 210 px above the bottom, so they collided with the question card when the answer was shown. |
| Over the Edge speech bubble is kept below the menu bar and its tail points at the host when pushed in from an edge. | Prevents the bubble covering the menu buttons. |

## Technical

| Decision | Reason |
|---|---|
| The project lives in `classroom-gameshow-bundle/` inside the repository this session was given (a separate Streamlit project). | It was the only repository and branch available; the folder is self-contained. |
| Games run in the same page (no iframes), each wrapped in its own function scope with prefixed element ids (`ote-`, `op-`). | One file, shared fonts and Three.js, no duplicated 600 KB library. |
| A Content-Security-Policy meta tag blocks all network connections. | Guarantees "no network calls" even if a future edit adds one by mistake. |
| Automated tests run in Chromium only. | Firefox and WebKit browser builds are not available in the build environment (no downloads allowed). The code avoids features missing from current Firefox and Safari (no ES modules, no import maps, no `structuredClone`). **Manual check recommended in Firefox and Safari before publishing.** |
| Small test hooks remain in the shipped file (`CGB.games.<id>.state()` and Outpace `setTime()`). | They let the tests play the games by keyboard and shorten the 60-second sprint; they do nothing unless called. |
| `dist/` is committed. | It holds the deliverables. |

## Version 1.1: Category Clash, Hex Hunt and the new host

| Decision | Reason |
|---|---|
| **Names kept as chosen: Category Clash and Hex Hunt.** | Owner's choice. A web search found a small Etsy seller offering a kids' "Category Clash" classroom word game, and a popular daily geography web game called "HexHunt". Neither is a registered mark that I could confirm, but both are in nearby markets, so check before selling. Renaming is cheap: change the `title` in each game's `game.js`, the card in `src/index.html` and the logo in `src/shared/brand.js`. |
| Neither game copies a TV set: no blue board with gold values, no gold hexagons, no show names, catchphrases or "double" round names. The banned-term check now also covers the two formats' show and presenter names. | Same rule as the first two games: mechanics may stay, branding may not. |
| Category Clash categories are **topics** from the active question set (4 to 6, chosen automatically across subjects or by the teacher). | Topics already tag every question and drive the weak-topic history, so no new data is needed. |
| Values are 100 to 500 in five difficulty bands per category. Order comes from an optional `Difficulty:` line, otherwise from an estimate (longer, explanatory and calculation answers count as harder). | The question format has no difficulty field, and tagging 252 built-in questions by hand overnight would be guesswork; the estimate plus an override is honest and quick to correct. |
| Category Clash turns rotate between teams, rather than the winner picking next. | Fairer for whole-class play with mixed-ability teams. |
| Steals on by default; wrong-answer penalty off by default. | Keeps every team engaged without negative scores discouraging weaker teams. |
| Hex Hunt teams link opposite edges of a square hexagon board (5 × 5 or 6 × 6); the hexagon's winner picks next. | Equal-sized classroom teams need an even contest. On a hexagon board one team always connects, so a round can never end without a winner. |
| Hex Hunt letters come from the answer's first letter, ignoring "a", "an" and "the". Answers starting with a number or symbol, "Any two: ...", or yes/no/true/false are left off the board. | These would make meaningless clues. 148 of the 191 starter-pack questions qualify (53 of 61 in Homeostasis L1), plenty for a 36-hexagon board. |
| If nobody can answer a hexagon, it gets a fresh question (and possibly a new letter). | Stops a board stalling on a question nobody knows. |
| Wrong answers in both new games are logged against team names, in the same shared history. | The brief's cross-game tracking works for teams as well as individuals. |
| **Pip redrawn in 2D (SVG)** instead of improving the 3D model. | A smoothed, cel-shaded 3D version was tried first: it was better than the blocks but still looked like a stiff mannequin, and its outlines disappeared against the dark sets. A flat ink-outlined cartoon matches the Showtime logos, reads clearly from the back of a room and costs no WebGL on the launcher. All customisation options and gestures were kept. |
| In Over the Edge Pip now stands in front of the 3D set in the bottom-right corner, and fades out when a setup or results card would cover him. | He no longer depends on the camera angle, and never peeks out from behind a card. |

## Version 1.2: Pre-launch polish

| Decision | Reason |
|---|---|
| **Host renamed Marty Marquee.** A saved default name of "Professor Pip" is changed to the new default; a name the teacher typed is kept. The CSS classes were renamed from `pip-` to `host-`. | "Professor Pip" is an existing character (the headmaster bear in the Badanamu children's learning brand). "Marty Marquee" ties in with the marquee wordmark, and a web search found no character, product or presenter of that name. |
| **One team palette for every game:** Team 1 blue ●, Team 2 orange ■, Team 3 purple ▲, Team 4 teal ◆ (`shared/teams.js`, `--team0`…`--team3`). Each colour has a text shade for light panels and a light shade for the dark boards. | Team 1 now looks the same in every game. Blue/orange stays distinct for the common colour-vision types, and the shapes mean nothing relies on colour. Hex Hunt's edges and owned hexagons use the palette, with white letters for contrast. |
| **"Team" everywhere** in the interface, guide and listing (code keeps its internal `players` names). Team names are stored once (`cgb.teams`) and prefill every game; on first run the names saved by any older game are carried over. Old "Player 1/2" defaults become "Team 1/2". | Consistent wording, and teachers type names once. |
| **Difficulty tags written by hand** for all 252 built-in questions (1 = recall of a single term, 5 = multi-step explanation or calculation). Every topic in every pack covers all five levels. Within a topic the questions are grouped by level in the pack text. | Category Clash rows now match their points. Grouping keeps the pack readable; games shuffle anyway. |
| **Category Clash fills rows in two passes:** first each row takes a question of exactly its level, then any empty row takes the nearest level left, then any question left. Untagged questions get a level from the estimator's order within the topic. | A short level never uses up another row's question, and old sets without tags behave as before. |
| **Outpace marking matches the other games:** Correct (C) and Wrong (W) work at any time; marking shows the answer with a ✓/✗ verdict for 1.3 s (0.9 s in the sprint, where the clock keeps running) before play moves on. Show answer (A) stays as an option. | Faster to run and consistent. The pause lets the class read the answer. |
| **Outpace track:** a HOME finish post with a sign, the number of steps to home on every tile with chevrons pointing home, and a camera that frames the whole track from the finish post to the Hunter's furthest start, set at the start of each round. Tags are about 70% larger. | Readability from the back of the room. Rules, distances and timings are unchanged. |
| **Setup cards share one layout:** two columns on screens 1000px and wider, one column below; the Start button stays at the bottom of the card. If a card would need scrolling, it first tightens its spacing, then closes "How to play" (one click opens it), then shrinks the logo and the topic list. Over the Edge's card stays clear of its score panel. | Every card fits without scrolling at 1920×1080, 1366×768 and 768×1024, with normal and large text. The order follows the brief's preferred approaches. |
| **Over the Edge host is sized to the gap beside the machine.** The machine's on-screen outline (bottom shelf with its front edge, lanes, peg board, header sign) is worked out for the current camera shot, including the sway. Marty shrinks at once for a closer shot and only grows back after the camera has settled. If the gap is too small for a 140px figure he is hidden rather than allowed to overlap. | He never covers the machine at any size or in any shot; the sweep checks it on every Over the Edge screenshot. |
| **Corner host in the other three games** (head and shoulders, with a caption bubble): in Category Clash and Hex Hunt he sits in a bar along the bottom, above the question overlay, whose content keeps clear of the bar. In Outpace he sits in the top panel's own row. On the results screens he sits at the top of the card. Captions cover the start, correct, wrong, steals, star tiles, round wins and the final result. No voice. | The bar and the panel row are part of the layout, so he cannot cover the board, track, questions or controls; the sweep checks overlaps too. |
| **Question-set dropdowns use short names** ("Combined Science: Biology"); the full name is in the tooltip and the count is in the field label. | The long names were cut off. |
| **Review request:** "Enjoying Showtime? A short review on Tes helps other teachers find it. Thank you!" at the foot of every results screen, as plain text with no link. | Unobtrusive, and the file stays free of links. |
| **Launcher on laptop screens** (1000px+ wide, up to 860px tall): a slightly shorter hero so the "More shows coming soon" strip is fully in view at 1366×768. At 1920×1080 it is in view; on a portrait tablet it starts below the fold and is reached by scrolling. | Never a sliver at the bottom edge. |

### Testing

| Decision | Reason |
|---|---|
| **Test build and product from one source.** Code between `@test-only` and `@end-test-only` markers (seeding and shortcuts) goes into `test-build/showtime-test.html` only. `build.js` makes the product by removing those blocks and stops if any test code is left; a test checks that the product equals the test build minus those blocks. Read-only `state()` and `layout()` hooks stay in the product. | Tests can jump to the final round or set the sprint clock, but nobody can in the shipped file. The read-only hooks change nothing and let the full playthroughs run on the shipped file. |
| **Seeded play:** every random choice that affects play (question order, boards, counters, physics nudges) goes through `CGB.random`, which tests seed (default 20261004, or `SEED=n`; the seed is attached to each test result). Purely visual randomness stays on `Math.random`. | Failures reproduce exactly. Seed 99 failing every time while 20261004 passed is how a weak jackpot test was found and fixed. |
| **Over the Edge simulation runs in fixed 1/60 s steps** (up to 4 per frame), and counter releases and the settle wait use simulation time. Tests can stop the clock and step it. | The same inputs give the same physics at any frame rate. On slow PCs the game catches up better than before (up to 0.067 s per frame, was 0.033 s). No physics constant changed. |
| **Gameplay tests:** low graphics and a 960×600 window. Over the Edge rounds use the manual clock. One full-length game of Over the Edge and one full game of Outpace (with the full 60-second sprint) run at High graphics on the shipped file before every push (`npm test`; tagged `@hq`). `npm run test:quick` skips those two and the screenshot sweep. | Rules don't depend on screen size or shadows, so this is several times faster. The High playthroughs still cover the shadow and glow code paths. |
| **The build is skipped when nothing in `src/` has changed** (`node build.js --force` rebuilds anyway). Playwright runs the build before every test run. | No stale files, no wasted rebuilds. |
| **The screenshot sweep** runs at full quality on the test build and jumps straight to the Over the Edge final (Round 1 is screenshotted in its own batch). It now also checks that each setup card fits, that the Over the Edge host never overlaps the machine, that the coming-soon strip is never cut off, that no card or panel block has text spilling out of it, and that the 3D scene is really drawn (it measures the share of lit pixels on the canvas with the page's own overlays hidden). | Shorter batches, the same coverage. The last two were added after reviewing the screenshots by eye found a portrait-tablet Outpace screen with an empty scene (the fog hid the whole track once the camera pulled back to fit it) and a squashed Over the Edge question card; the old checks passed both. |

## Things not done

| Item | Reason |
|---|---|
| Manual playtesting on a real interactive whiteboard and in Safari. | Not possible from the build environment; recommended before release. |
| Hand-checked difficulty levels for the built-in packs. | The automatic estimate orders them sensibly; adding `Difficulty:` lines to `src/shared/packs.js` would make Category Clash values exact. |
| Trade-mark searches beyond a quick web search. | A formal UK IPO search for "Over the Edge" and "Outpace" in classes 9, 28 and 41 is worth doing before selling. |
