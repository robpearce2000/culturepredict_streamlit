# Overnight brief: finish Showtime: Classroom Gameshows

Rob is asleep. Work through every part below **in order**, autonomously, until the bundle is finished and released. Don't ask questions: make sensible decisions and record them in `DECISIONS.md`.

## Working rules (apply to every part)

- **Start** once any work already in progress (the 10-minute game lengths) is finished and pushed.
- **Keep everything from earlier work working:** the offline single file, no banned terms, layouts that fit at 1366×768, the simplified setup screens, the main-screen subject choice, the 10-minute game lengths (Over the Edge 6 + 4-question final, Outpace one Deal Round then the Final Sprint, Category Clash 4 × 3, Hex Hunt 4 × 4), and all existing tests passing.
- **Out of scope:** new question banks. Work with the packs already in the bundle.
- **After each part:** run its tests (including the full-graphics playthroughs where 3D games changed), fix anything that fails, rebuild the dist file, commit, push, and add a short summary of what changed to `OVERNIGHT-REPORT.md` before starting the next part.
- **If a test still fails after several genuine attempts,** record it in `KNOWN-ISSUES.md` with what you tried, and carry on, so the rest of the night isn't wasted.
- **If you hit a usage limit,** carry on from where you stopped as soon as you can.
- Respect the reduced-motion setting for every new animation.

---

## Part 1: Outpace

A reference file, `outpace-reference.html`, is included. It is a built bundle in which several of these fixes have already been prototyped directly in the dist file: performance changes, smaller Deal Round labels, the side-column Final Sprint layout, winning as soon as the target is reached, and six subject looks. Port these into `src/` properly rather than copying the dist file. Treat the prototype as a starting point: improve it where needed and make it consistent with the rest of the codebase. Note that the earlier setup work removed per-game settings, so the prototype's "Subject look" selector should not come across; the look follows the main-screen subject instead.

1. **Performance.** It is very laggy on Rob's laptop. Find and fix the causes: cap the render resolution, drop shadows (the racers float, so they add little), avoid forcing a page layout every frame (the name tags were reading their size every frame), and add automatic quality step-down when frames are slow for a sustained spell. Target a smooth frame rate on a typical school laptop at High quality. The prototype includes all of these.
2. **Final Sprint layout.** The question card covers the racers. On wide screens, put the question in a column at the side and centre the race in the remaining space, including the camera framing and name-tag positions. On narrow and portrait screens, keep a layout where the racers stay visible.
3. **Win as soon as the target is reached.** Interpretation: when the class reaches the Final Sprint target, the win animation should play straight away, not after the 60-second clock runs out. Freeze the clock, allow the usual brief undo window, then play the escape sequence.
4. **Subject looks.** The graphics change with the subject. Science stays as atoms. Make a distinct, polished look for Science (Biology, Chemistry and Physics can share it or have their own), Maths, English, History and Geography, plus a general look for anything else, so the game is ready when packs for those subjects are added later. The runner stays gold and the Hunter stays magenta in every look, so the race reads the same. The look follows the subject chosen on the main screen (added in the earlier setup work), with no extra setting in Outpace. The prototype has Science, Maths, English, History, Geography and General looks to build on.
5. **Labels too big in the Deal Round.** The class name, HUNTER and HOME labels are too large while the whole track is in view. Make them smaller in that round. They can stay larger in the Final Sprint close-up.

Tests for this part: the immediate win on reaching the target, each subject look, and Outpace's frame rate at Low and High (record the result).

---

## Part 2: Over the Edge, Category Clash and Hex Hunt

### Over the Edge

Teams keep dropping their counters one at a time, as now.

1. **Random drop order.** After marking, the order in which the correct teams drop is random each question, and the order is shown clearly on screen.
2. **Fair winnings.** At the moment the team that drops first on the first question wins far too many counters, because the shelf starts loaded. Teams should win roughly the same amount per drop, with some randomness to keep it close and tense. Fix the cause (for example, how the shelf starts and how the shelf settles between drops) rather than just capping. Acceptance test: simulate at least 50 questions with six teams all correct every time. No team's average winnings may differ from the overall average by more than about 15%, and the first team to drop must not be systematically ahead.
3. **Flashing set pieces.** Parts of the set just under the OVER THE EDGE sign flicker or flash, probably z-fighting from overlapping surfaces. Find them and fix them so nothing flickers.
4. **Coin reflections are too bright.** Turn down the light reflections on the counters a little. They should still look metallic, but not glare.

### Category Clash
1. **A real difficulty scale.** The tiered difficulty from the polish brief isn't working properly. Research how to build a reliable tiered system of question difficulty within a question set (for example, tiers based on exam board assessment objectives, command words, or Foundation and Higher tier content). Apply it to every question in the existing packs, and make sure each points row draws from the matching tier. Explain the system in `DECISIONS.md` and in the bank manager help, so teachers writing their own questions can tag them.
2. **Categories are topics, not subjects.** Category Clash is played in one lesson, for example Biology, so the columns should be topics within the subject chosen on the main screen (added in the earlier setup work), such as Cell biology or Ecology. The teacher picks which topics appear, with a sensible default. Each topic must have enough questions at every tier to fill its column.

### Hex Hunt
1. **Answers must truly start with the hexagon's letter.** Some answers begin with "the" or "a", so the letter clue is wrong. Ignore leading articles when choosing the letter (use the first significant word), and audit every answer in the existing packs so each hex letter is genuinely the first letter of its answer. Add a test that checks this for every question that can appear on the board.
2. **The gold outline.** There is a gold outline around one hexagon that moves each time. It is probably the keyboard selection cursor. If so, make its purpose obvious: show it only when the teacher uses the arrow keys, and add a tiny key hint. If it is something else, fix whatever causes it.
3. **Simpler marking.** Replace the percentage marking with two big buttons, one for each side. The teacher compares the whiteboards and presses the side that had more correct answers. Add a "Neither" option for when nobody gets it, and keep undo.

Tests for this part: the Over the Edge fairness simulation, Category Clash topic columns and tiers, and Hex Hunt two-button marking and the first-letter check.

---

## Part 3: Home screen, Start game button and steady camera

### 1. A "Start game" button before the first countdown

In the games where a 20-second countdown begins as soon as the teacher finishes team setup, the clock starts before the class is ready.

- After setup, show the game screen with a large **Start game** button (Enter) and nothing ticking. The countdown for the first question begins only when it is pressed.
- This gives the teacher a moment to explain the rules and get whiteboards out.
- Apply it to every game that has a countdown, and use the same button style and wording in each.

### 2. Over the Edge: steady camera during team setup

While the teacher is on the setup screen choosing teams, the camera behind it moves back and forth, which is distracting.

- Keep the camera still while setup is showing: a fixed, attractive view of the machine.
- The usual camera moves resume once the game starts.

### 3. Home screen (launcher)

1. **Professor Pip's speech bubble.**
   - He doesn't introduce himself. His bubble says: "Choose your subject, then pick a game!"
   - The name label on his bubble only appears if a name has been entered in the host customiser (for example a teacher's own name). With no name entered, show no name label at all.
   - The bubble currently covers half of his head. Position it so it never overlaps him at any of the three screen sizes. Add this to the overlap tests.
2. **Layout.** Put the SHOWTIME wordmark in the centre and the subject picker on the left.
3. **Subject picker as a dropdown.** Replace the visible list of subjects with a compact dropdown that shows the chosen subject and only opens to reveal the others when clicked. It must work by mouse, touch and keyboard. Keep the exam board choice nearby in the same compact style.
4. **Twinkling lights.** The bulbs around the SHOWTIME sign should twinkle gently, at varied timing so they don't all flash together. With reduced motion on, they stay lit and still.
5. **Play buttons.** The game cards already show each game's title, so the buttons just say **Play**. All four Play buttons must be exactly the same size. At the moment the Category Clash button is bigger.

Tests for this part: the Start game button in every game with a countdown (no countdown before it is pressed), a still camera on the Over the Edge setup screen, the launcher speech bubble not overlapping Pip, the dropdown subject picker, and equal-sized Play buttons.

---

## Part 4: 3D upgrade

Check `CHANGELOG.md` first. If a 3D upgrade has already been done, skip to Part 5 and note that in the report.

This is a presentation upgrade only: **do not change any game's rules, rounds, scoring or question flow.** It was planned before Parts 1–3, so it must fit around them:

- **Outpace** keeps its subject looks and performance fixes. The 3D additions (track, finish arch, camera, catch and escape moments) must work in every subject look, and must not make it laggy again. Check frame rate at High and Low before and after, and simplify anything that costs too much.
- **Category Clash** uses the 4 × 3 board with topic columns and difficulty tiers.
- **Hex Hunt** uses the 4 × 4 board and two-button marking, and keeps the keyboard cursor behaviour from Part 2.
- **Readability from the back of a classroom beats spectacle.** If an effect makes text harder to read or slows the teacher down, drop it. Any animation longer than about a second must be skippable with Space or Enter.

### Principles for all three games

- **Flat boards in a 3D world.** Keep game boards face-on and readable. Put them in a lit 3D studio set with depth, a backdrop and subtle camera motion. Do not tilt boards at angles that distort text.
- **One "wow" moment per game,** as memorable as Over the Edge's cascade. Each game's is specified below.
- **Motion with purpose.** Animate state changes (picks, reveals, scoring, wins) so the class can see what happened. Keep each animation short; the teacher must be able to press Space or Enter to skip any animation longer than about a second.
- **Each game keeps its own identity** (colours, logo) but shares lighting quality, material style, the team palette and the Professor Pip mascot from the polish brief.
- **Performance:** the Low graphics setting must turn off shadows, bloom and heavy effects and stay smooth on older classroom PCs. Reduced motion must cut camera moves and large animations to simple fades.

### 1. Outpace: top priority

It is already 3D but currently the weakest game: a thin track in a mostly empty scene.

- **A proper track and set.** A chunky, well-lit track with clear step tiles, step numbers, and chevrons pointing home. A big glowing **finish arch** at the home end. A set around the track (lighting rigs, floor, backdrop) so the scene doesn't feel empty.
- **Camera that tells the story.** Frame the whole track during the Deal Round choice. During questions, follow behind or beside the runner with the Hunter visible behind them, so the gap between them is obvious.
- **Make the gap felt.** On a correct answer the runner surges forward one step. On a wrong answer the Hunter lunges forward with a visible burst of speed. A gap meter or distance readout updates with each step.
- **Wow moment:** if the Hunter catches the runner, a slow-motion catch with a camera swing and a flash. If the runner reaches home, they burst through the finish arch with confetti and a light show. The Final Sprint gets a countdown clock built into the set, and the lighting turns red in the last ten seconds.

### 2. Category Clash

The current 2D board is clean and readable; keep that.

- **Board in a studio.** The face-on board becomes a wall of lit 3D tiles with depth and bevelled edges, sitting in a studio set with spotlights and a backdrop.
- **Tile flip reveal.** A picked tile flips over in 3D and expands into the question card, and the camera pushes in slightly. Answered tiles flip back to show the winning team's colour.
- **Scoring feedback.** A light sweep across the team's score panel when they win points, and a short "stolen!" effect when a steal succeeds.
- **Wow moment:** the hidden ★ double tile. When picked, the camera pushes in, the lights dim, the tile spins and bursts with gold light before the question appears.

### 3. Hex Hunt

- **Chunky 3D hexes.** Each hexagon becomes a raised 3D tile with bevelled edges and the letter clearly on its face. Keep the board face-on so every letter is easy to read.
- **Claiming a hex.** A won hexagon pops up, flips or rises slightly and changes to the team's colour with a short burst of light. The team's two target edges glow more strongly as their path grows.
- **Wow moment:** when a team completes a path, the winning chain lights up hex by hex from edge to edge, and the camera sweeps along it, followed by a celebration burst.

### 4. Professor Pip in 3D scenes

Wherever Pip appears in these games (per the polish brief), place him in the 3D scene or as a well-integrated overlay. Give him a few reactions for the new moments (cheer, gasp, groan, point). He must never cover the board, track, questions or controls.

Tests for this part: every screen at all three sizes at High and Low graphics and with reduced motion on, skipping animations, frame rate on Low, and the overlap tests extended to the new 3D elements.

---

## Part 5: Release

1. Set the version to 1.0.0 everywhere it appears (About panel, guide, CHANGELOG).
2. Rebuild the dist file. Regenerate the teacher guide PDF and the Tes listing images from the final build, and update the listing description.
3. Run the complete test suite, including both full-graphics playthroughs and the batched screenshot sweep at all three sizes. Review the screenshots yourself. Rerun the banned-term search and the contrast checks.
4. Push, then open a pull request into `main` and merge it if you have permission. If you can't merge, leave the pull request open and say so in the report.
5. Write `RELEASE-NOTES.md`: what's in the bundle, what changed overnight, any known issues, and a checklist of what Rob still needs to do himself before selling (classroom trials, checking the questions, trade mark searches, his employment contract, and Tes setup).
6. Finish `OVERNIGHT-REPORT.md` with a short summary at the top: which parts are done, which aren't, and where the finished file is.
