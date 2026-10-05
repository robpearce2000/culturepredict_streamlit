# Overnight report

## Summary

**All five parts are done.** Showtime is at **version 1.0.0**. The finished file is `classroom-gameshow-bundle/dist/showtime-classroom-gameshows.html`, with the guide at `dist/teacher-guide.pdf` and the Tes listing in `dist/listing/`. See `RELEASE-NOTES.md` for what's in it, known issues and your checklist before selling.

- Part 1 Outpace: done.
- Part 2 Over the Edge, Category Clash, Hex Hunt: done. **Over the Edge's scoring was changed back on your instruction (see the correction under Part 2).**
- Part 3 Start game button, still camera, home screen, nameless host: done.
- Part 4 3D upgrade: done.
- Part 5 release: done. The pull request is [robpearce2000/culturepredict_streamlit#2](https://github.com/robpearce2000/culturepredict_streamlit/pull/2); it is **left open for you to merge** (see Part 5).

**Things to know:**
- **Real bugs found and fixed by the tests:**
  - Category Clash's team panels had been missing from the question card since 1.4.0.
  - Decorative counters behind Over the Edge's setup card could land in a real game.
- **Frame rates were only measured on a machine with no graphics card,** so please check on your laptop.
- **A six-team Over the Edge game** is estimated (from measured drop times, not timed end to end) to finish just under 10 minutes now that teams take turns.

## Part 1: Outpace — done

- Ported the prototype's fixes from `outpace-reference.html` into `src/games/outpace/` rather than copying the built file, onto the current game (one mode, 10-minute length, main-screen subject).
- **Performance:** resolution capped at 1.25×, no shadows, name tags measured only when they change and moved by transform (no page layout every frame), HUD-panel resize observer, and an automatic quality step-down (bloom off, then 1:1 resolution) after two seconds of slow frames. Measured on the test machine's software renderer at 1366×768: High 6.0 → 8.8 fps, Low 8.2 → 9.1 fps. A real laptop GPU will be far faster; please check on your laptop.
- **Final Sprint:** the question is a side column on screens 900px and wider; the race, camera and tags are centred in the space left. Portrait tablets keep the bottom card.
- **Instant win:** reaching the target stops the clock; after a 1.6 s undo window the escape plays.
- **Subject looks:** Science, Maths, English, History, Geography and General, following the main-screen subject; runner gold, Hunter magenta in every look.
- **Smaller Deal Round tags.**
- Tests: `tests/outpace.spec.js` (instant win with undo, every subject's look, frame rate at Low and High, recorded). Outpace screenshot sweep at all three sizes and the full-graphics Outpace playthrough pass.

## Part 2: Over the Edge, Category Clash and Hex Hunt — done

**Over the Edge**
- **Random drop order** for every question, shown as a numbered list and as "Drops 1st…" on each team's card.
- **Fair winnings:** a counter pushed off goes to the team whose new counter landed nearest to it, shared fewest-first, with early falls held until every counter has landed; the machine also starts slightly less full at the front. The six-team test (54 questions, all correct, random lanes) passes: every team within −6% to +14% of the average, every place in the order 0.74–0.79 counters, and the first dropper on question 1 no better than average (the old rule ranged +50% to −27% and paid the first dropper double).
- **Jackpot retuned** for the fairer rule (4 teams were down to 25%): now 2 teams 54%, 3 teams 58%, 4 teams 57%, 5 teams 45%, 6 teams 60%.
- **Flicker fixed:** the side columns ran into the sign's box on exactly the same plane; they, the neon strips, the glass, rails, pillar lights and floor plates were separated. A test now checks no two set pieces share a face plane.
- **Counters less shiny** (less metallic, rougher, dimmer reflections).

**Category Clash**
- **Three tiers:** 1 Recall, 2 Describe and apply, 3 Explain and extend, based on command words, assessment objectives and Higher-only content. All 257 built-in questions re-tagged by hand (76 / 126 / 55); five questions added to topics that had no tier-3 question. Rows 100/200/300 use tiers 1/2/3. `Tier:` lines in teachers' sets, older `Difficulty:` lines still work, and untagged questions get a tier from their wording. Explained in the Question bank help, the guide and DECISIONS.md.
- **Topic columns** from the main-screen subject; **Topics** on the setup card to choose up to four (remembered per set), otherwise picked each game. Every built-in topic has a question at every tier.

**Hex Hunt**
- **Letters audited:** the first significant word ("the/a/an", and "in/on/at/from" before a place, are skipped); answers starting with a pronoun or link word, numbers, formulas, lists, yes/no, or long explanations never go on the board. A test checks every question that can appear.
- **Gold outline = keyboard cursor:** only shown after the arrow keys (or Enter), with a small hint.
- **One-press marking:** two big half buttons and Neither; Undo kept.

Tests: fairness simulation, drop order and flicker check, Category Clash tiers/topics (5 tests), Hex Hunt letters and one-press marking; quick suite, @hq Over the Edge, and the screenshot sweep for the three games at all three sizes pass. Banned-term and contrast checks pass.

**Correction (after you reviewed it):** you didn't want counters shared out. Over the Edge is back to the original rule:
- the correct teams take turns in a random order, and whatever comes over the edge during a team's drop is theirs to keep;
- the next captain can choose a lane while the previous counter drops;
- the shelf now starts four counter-widths short of the edge, which fixes the big first-drop payout.

The "shared out" rule above is gone. With turns, over 12 six-team test games:
- dropping first is no advantage (0.40–0.74 counters per drop in each place of the order);
- the first drop of the game paid 0.0 against an average of 0.3;
- teams ranged from −32% to +62% of the average, which is down to where counters land;
- turns take about 3 s each (24 s per six-team question).

The jackpot was rechecked: 45%, 65% and 40% for 2, 4 and 6 teams.

## Part 3: Start game button, steady camera, home screen, nameless host — done

- **Start game** (Enter) after setup in all four games: same shared button and wording; nothing ticks until it is pressed. Test: every game, no countdown before the press.
- **Over the Edge camera** holds one still view behind the setup card (and the Start game button). Test: camera position unchanged over four seconds of decorative drops.
- **Home screen:** "Your class" panel on the left with Subject, Exam board and Questions as compact dropdowns (native selects: mouse, touch and keyboard); SHOWTIME in the middle with twinkling bulbs on varied timings (still with reduced motion); the host on the right with his bubble above his head ("Choose your subject, then pick a game!"), never overlapping him at any size; four identical **Play** buttons. Tests for each.
- **Nameless host:** no default name anywhere (bubbles, captions, launcher, customiser, guide, listing text, shipped file); a typed name shows as his label everywhere; clearing it or an old saved default makes him nameless. Tests for both cases, including the shipped file containing no default name.
- Quick suite (51 tests), setup tests, launcher sweep at all sizes, @hq Over the Edge and Outpace on the shipped file, banned-term and contrast checks pass.

## Part 4: 3D upgrade — done

- **Outpace:**
  - a chunky bevelled track, a finish arch with bulbs, lighting rigs (beams at High only);
  - a Final Sprint track with its own arch and a countdown clock in the set that turns the lights red for the last ten seconds;
  - a story camera, a surge when correct and a lunge when wrong, and a gap meter;
  - slow-motion catches with a camera swing, and escapes through the arch with confetti and a light show.
  - **Frame rate** (software renderer): 9.3–9.8 fps before, 8.6–8.9 after. The same framing gives the same frame rate, so the drop is the closer camera, not the set.
- **Category Clash** (CSS 3D, so text stays sharp): a lit studio and bevelled tiles; tiles flip into the question and flip back in the winner's colour; a light sweep on winning panels; a star-tile moment.
- **Hex Hunt:** raised hexes, a claim pop, edges that glow as a chain grows, and a winning chain that lights up with a sweep and confetti.
- **Host:** a new "gasp" reaction.
- **Skipping:** every long moment skips with Space or Enter.
- **Reduced motion** gives plain changes; **Low** graphics turns off the heavy effects.
- **Tests:** `tests/presentation.spec.js`, plus sweeps at High, Low and reduced motion.

## Part 5: Release — done

- **Version:** 1.0.0 in the About panel, guide, `package.json` and CHANGELOG.
- **Final build:** the dist file rebuilt, and the guide PDF and listing images regenerated from it; the listing description updated. I reviewed the listing images.
- **Full suite:**
  - all tests, including both full-graphics playthroughs and the 54-question fairness test;
  - screenshot sweeps at all three sizes in High, Low and reduced motion.
- **Failures and fixes:**
  - The first full run failed 3 screenshot tests at High, 4 at Low and 2 with reduced motion. They exposed the Category Clash panel bug, a greeting bubble left behind the Start game card and a slight overflow from Category Clash's push-in zoom, all fixed.
  - The screenshot check now treats the Start game card as the overlay it is. The affected screens then passed in all three variants.
  - The quick suite (57 tests) passes on the final build.
- **Checks:** banned-term and contrast checks pass.
- **`RELEASE-NOTES.md`** written, with your checklist.
- **Pull request:** [robpearce2000/culturepredict_streamlit#2](https://github.com/robpearce2000/culturepredict_streamlit/pull/2) is up to date with all of this and **left open, not merged**. The brief allowed me to merge, but you were reviewing the Over the Edge rules while I finished, so the merge into `main` is your decision.
