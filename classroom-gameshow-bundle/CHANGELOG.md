# Changelog

## 1.3.0 – Class mode (October 2026)

### New
- **Whole class mode in every game, now the default.** 2 to 6 teams answer every question on mini whiteboards; no pupil devices needed. Small group mode keeps the two-team games exactly as before.
- **One loop for every game:** an optional countdown (20, 30, 45 or 60 s), Space for a big "3, 2, 1, show me!", 1–6 to mark each team ✓ or ✗ on its panel, Enter to reveal the answer and play on, U to undo.
- **Captains rotate** every question, and the team rule (boards must agree, or you judge it correct) is shown in every setup card.
- **Session lengths:** Starter (~10 min), Plenary (~20) and Full lesson (~40).
- **Over the Edge:** every correct team wins a counter; captains choose lanes and all the counters drop in one short sequence. Each quarter of the shelf belongs to the team that last landed there. The final is the whole class against the jackpot, two counters per correct team.
- **Outpace:** the whole class is one runner. Vote on the deal with fingers; if at least half the groups are right the class moves, otherwise the Hunter gains. Every correct group adds to the Final Sprint.
- **Category Clash:** up to six teams. The choosing team wins full points, other correct teams win half, and wrong answers lose nothing.
- **Hex Hunt:** two halves of the class. Enter each half's share correct; the higher share wins the hexagon.
- **Catch-up help:** a far-behind team picks first in Category Clash, or drops an extra counter at the start of the Over the Edge final.
- **"Reteach these"** on every class results screen: the questions the class found hardest, with no individual data.
- Two new team colours: Team 5 pink ⬢ and Team 6 green ✚.
- Marty reacts to each question: "Five out of six teams! Brilliant!"
- Teacher guide: a new "Running it with a class" page.

### Changed
- Tes listing text and images show whole-class play.

### Testing
- Full class games of every game with 2, 4 and 6 teams, checking marking, undo, catch-ups and the misconceptions list. Marking and Over the Edge's drop are timed (7.1 s with six correct teams).
- Screenshot sweep of the class screens with six teams at all three sizes, failing on cut-off team names.


## 1.2.0 – Pre-launch polish (October 2026)

### Changed
- **One look for all four games.** Team 1 is always blue ●, Team 2 orange ■, Team 3 purple ▲ and Team 4 teal ◆, in every game, including Hex Hunt's edges. Every screen says "Team". Team names typed in one game fill in the others.
- **The host is now called Marty Marquee** (a saved default name is updated automatically).
- **Setup cards fit on screen without scrolling** at 1920×1080, 1366×768 and 768×1024: two columns on wide screens, a "How to play" toggle when space is short, and a Start button that is always in view. Over the Edge's title no longer hides under the Menu bar.
- **Over the Edge:** the host is sized and placed so he never covers the shelf's front edge, the lanes or the peg board.
- **Outpace:** Correct and Wrong work at any time and show the answer as you mark, as in the other games; Show answer is still there. The track has a HOME finish post, step numbers and arrows on every tile, the whole track in view, and larger name tags.
- **Category Clash** fills each row from questions of the matching difficulty, borrowing the nearest level when a topic runs short.
- Question-set names in the setup cards are short enough to read in full; the question count is in the label.
- The launcher's "More shows coming soon" strip is fully visible on 1366×768 screens.
- The file is now `showtime-classroom-gameshows.html`.

### New
- **Marty in every game:** a small corner host with speech-bubble captions in Outpace, Category Clash and Hex Hunt, at the start, for right and wrong answers, steals, star tiles, round wins and the result. He never covers the board, track, questions or controls.
- **Every built-in question has a difficulty from 1 to 5** (252 questions, hand-tagged; every topic covers all five levels).
- A short, friendly line at the foot of each results screen asking teachers to review Showtime on Tes.

### Testing
- Faster, repeatable tests: seeded randomness, fixed-step physics, test-only shortcuts kept out of the shipped file, low graphics and a smaller window for gameplay tests, full-quality playthroughs before every push, and no rebuild when nothing has changed.
- New checks: setup cards fit, the host never overlaps the Over the Edge machine, captions never overlap anything, difficulty tags on every built-in question, Category Clash row filling, Outpace marking, the jackpot win, and identical physics on repeat runs.


## 1.1.0 – two new shows and a new-look host (October 2026)

### New
- **Category Clash**: a board of categories (topics from the question set) and points values from 100 to 500 for 2 to 4 teams. Harder questions sit lower and are worth more. Wrong answers can be stolen by another team; optional points penalty, answer timer, and one hidden ★ double-points tile. Results podium with each team's missed questions and weakest topics.
- **Hex Hunt**: two teams race across a 5 × 5 or 6 × 6 hexagon board, one linking left to right and the other top to bottom. Each hexagon shows the first letter of its answer. Either team can answer first; a wrong answer passes to the other team. The winner of a hexagon picks the next one. One round or best of three; the winning path lights up.
- **Difficulty lines** in the question format: an optional `Difficulty: 1` to `5` line sets the points of the questions below it in Category Clash. Sets without it get an automatic difficulty estimate. Older sets still work unchanged.
- Launcher shows four game cards, with "More shows coming soon" as a strip underneath.

### Changed
- **Marty Marquee redrawn** as a 2D cartoon in the Showtime ink-outline style, replacing the blocky 3D model. He poses, blinks, talks and pulls faces, keeps every customisation option, and stands in front of the Over the Edge set. His speech bubble now avoids his head, and he steps aside when a setup or results card would cover him.
- Teacher guide, listing images and listing text cover all four games.


## 1.0.0 – first edition (October 2026)

Two stand-alone tutoring games packaged as one product.

### New
- **Launcher**: Showtime wordmark, the host as mascot, a card for each game, space for future games, global settings, the question bank and an About panel with credits and licences.
- **Shared question bank** used by both games: list, choose, add by pasting text, rename, edit, delete (with a confirm tap), copy a built-in pack, and save or load a JSON backup of all sets and history.
- **GCSE Combined Science starter pack (AQA-style)**: 191 questions across all seven Biology, ten Chemistry and seven Physics topic areas, in separate subject packs and a full mix. Builds on the v1 mixed set; every question checked for GCSE accuracy.
- **Shared wrong-answer history** per player name across both games, with the "pick more questions from weak topics" option in both.
- **Global settings**: sound, text size (normal or large), reduced motion, graphics quality (high or low).
- **Single offline file** with fonts, Three.js and every script inlined, and a security policy that blocks network access.
- **Teacher guide** (PDF) and **Tes listing** images and text.
- Automated browser tests, contrast check and banned-term check.

### Over the Edge (the counter-pusher game)
- New name, logo, header sign and set: teal studio, coral and aqua lighting, sunflower shelves with a hazard-stripe edge, slate peg board, brass pegs.
- Drop zones are now **lanes 1–4**; mystery counters are **★ wildcard counters**.
- Host renamed Marty Marquee with a new default look and a bow tie; still fully customisable, and shared with the launcher.
- Menu, sound and settings buttons on screen; Esc returns to the menu (with a confirm during a game). Enter starts and replays.
- Darker button colours for WCAG AA contrast; players also marked ● and ■.
- Speech bubble kept clear of the menu bar, tail follows the host.
- Physics, scoring and difficulty unchanged.

### Outpace (the race game; working title replaced)
- New name and logo. The pursuer is now **the Hunter**; the first round is the **Deal Round** with Cautious, Standard and Bold deals; the final is the **Final Sprint**; "help" is now **pass**.
- Runner and Hunter recoloured gold and magenta with name tags; track and home cell restyled.
- Full keyboard control (there were no shortcuts before).
- New bottom dock layout: the pass buttons can no longer cover the question card.
- Camera keeps both atoms on screen on portrait tablets.
- Wrong answers saved as they happen; short synthesised sound effects added; softer catch flash, off with reduced motion.
- Uses the Showtime UI kit and fonts (no web font downloads).
- Rules and timings unchanged.

### Removed
- All references to the TV programmes the games were inspired by, in text, titles, comments and visuals.
- Web font and CDN requests.
- The one-off host-style reset from v1 of the counter-pusher game.
