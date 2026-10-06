# Changelog

## 1.2.0 – Every board and subject (October 2026)

- **Built-in packs for every GCSE specification in scope:** AQA and Pearson Edexcel Biology, Chemistry, Physics, Maths, History and Geography (Edexcel Geography A and B). 135 topic packs, 7,792 questions, one pack per specification topic; in History and Geography every option is its own pack, so you tick the options you teach. Fieldwork, pre-release material and AQA's annually changing historic environment site are left out because they depend on the school.
- Every question is tied to a numbered point of its specification, has a difficulty from 1 to 3 and is tagged when it is Higher tier only (Maths and the sciences). Each pack was written from the specification text and then checked separately against it and for errors; maths and science calculations were re-worked in code.
- **Smaller file:** the questions are stored compressed inside the single file and unpacked when a topic is first used, so the whole product is about 1.8 MB.
- **Outpace Final Sprint:** a card now explains the sprint (60 seconds, one step per correct team, the target and the pot) and nothing starts until you press Start the Final Sprint (or Enter). The sprint has no track: just the two racers and the clock.
- **Fix:** in Category Clash, clicking "Back to the board" or "See the results" after marking did nothing (only the Enter key worked, and not once the button had been clicked). Both now work with the mouse, a tap or Enter.
- Hex Hunt no longer puts algebra answers (such as "x² + 3x") on the board as if they were words.

## 1.1.0 – Topic packs (October 2026)

- **Built-in packs for every topic of AQA GCSE Biology, Chemistry and Physics.** One pack per specification topic (25 packs, 1,425 questions), replacing the Combined Science starter packs and the Homeostasis lesson set (their good questions were rewritten into the new packs). Every question is tied to a numbered specification point, has a difficulty from 1 to 3 and is tagged when it is Higher tier only; the separate-science content and every required practical are covered. Each pack was written from the specification text and then checked separately against it and for errors.
- **Choose topics, not sets.** On the main screen, the Questions dropdown lists "Mixed: all topics" and then the topic packs in specification order, with your own sets below; tick one topic for a lesson, several for a recap, or Mixed for revision. Remembered for each subject and exam board.
- **Higher tier switch.** Untick "Include Higher tier only questions" for a Foundation class and every game leaves them out.
- **Category Clash with one topic** takes its columns from that topic's subtopics in the specification, so a one-topic lesson still gets a full board.
- **Accepted answers and notes.** Every game shows the other accepted answers and a short teacher's note (a common mistake, say) under the answer. Your own sets can have them too, with Accept: and Note: lines.
- **Subjects:** Biology, Chemistry, Physics, Maths, History and Geography, plus Other. Combined Science and English are no longer listed; sets you filed under them are untouched and appear under Other.
- **"Difficulty" replaces "tier"** for the 1-to-3 scale (tier now means Foundation or Higher, as exam boards use it). Old Tier: 1-3 lines still work.
- The Question bank shows each built-in pack's specification code and version.

## 1.0.1 – Smoother Outpace (October 2026)

- **Outpace runs about twice as smoothly.** It no longer uses a full-screen glow pass or moving lights; the glows are drawn far more cheaply, every surface uses simpler shading, and it draws at standard resolution. If a computer still struggles, it now steps its quality down sooner (after a second and a half below about 45 frames a second): first the floating dust, set dressing and light beams go, then the resolution drops to three-quarters.
- **No more stutter when the camera and racers move.** Every movement now runs on the clock rather than counting frames, so the camera glide and the racers' steps are smooth and the same speed on 60, 120 and 144 Hz screens, and one slow frame no longer makes them jump. A browser drawing 3D without the graphics chip (hardware acceleration switched off) now starts in the lightest mode.
- **No more jolts as the camera swings.** Everything in the set is loaded onto the graphics chip before play, instead of the first time the camera brings it into view (the browser used to pause for that mid-move). The camera shake on each answer is now a short smooth sway instead of a new random jump every frame, and the camera glides once to where the racers are heading rather than wobbling with them as they bounce into place. The picture is no longer rebuilt when nothing about its size has changed.
- **The picture no longer jitters as the camera moves.** When the question panel changed size (as it does on marking), the 3D picture was nudged to stay clear of it in small jumps ten times a second; it now slides there smoothly. The camera now eases in and out of every move (it used to set off at full speed in a single frame), the class's counter no longer slides back on screen as it lands, and the name tags glide with the racers instead of snapping from pixel to pixel.
- **The home-screen sign is livelier:** a band of light runs round the bulbs, lit bulbs glow and the rest dim right down. With reduced motion every bulb stays lit and still.

## 1.0.0 – First release (October 2026)

The first public release. (The 1.1 to 1.4.1 entries below were development builds before release.)

### Outpace
- **Much smoother:** a capped render resolution, no shadows, name tags that no longer make the browser lay out the page every frame, and an automatic step-down in quality if a computer can't keep up.
- **The Final Sprint question sits in a column at the side** on wide screens, so the race stays in view.
- **The class escapes as soon as it reaches the sprint target**, without waiting for the clock (a short pause allows an undo first).
- **The racers follow the subject:** atoms for science, and new looks for Maths, English, History and Geography, plus a general one. The class is always gold and the Hunter always magenta.
- Smaller name tags in the Deal Round, when the whole track is in view.

### Over the Edge
- **Teams drop in a random order for every question**, shown on screen.
- **Teams take turns to drop again, and whatever comes over the edge on your drop is yours.** (In 1.3 every team's counter dropped at once, so it wasn't clear whose drop pushed what off.)
- **No more huge payout on the very first drop:** the shelf starts a little less full at the front.
- The jackpot is retuned for the fairer rule.
- **No more flickering** set pieces under the OVER THE EDGE sign, and the counters are a little less shiny.

### Category Clash
- **The columns are topics from your subject.** Choose up to four on the setup screen, or let the game pick a different mix each game.
- **A real difficulty scale:** every question is tagged Tier 1 (recall), 2 (describe and apply) or 3 (explain and extend), and the 100, 200 and 300 rows use the matching tier. Add a `Tier:` line to your own sets; older `Difficulty:` lines still work.

### Hex Hunt
- **One press to mark:** press the half that had more right answers, or Neither. Undo is still there.
- **Every letter really is where the answer starts** ("the", "a" and "an" are skipped, and answers that are explanations, lists or numbers stay off the board).
- The gold outline is the keyboard cursor, so it only appears when you use the arrow keys, with a small hint.

### Every game
- **A Start game button after setup.** Nothing ticks until you press it, so you can explain the rules and get the whiteboards out.
- **The host has no name** unless you give him one in Customise the host; with a name, it shows on his speech bubbles.

### Home screen
- The subject, exam board and question set are compact dropdowns on the left; the SHOWTIME sign is in the middle, and its bulbs twinkle (still with reduced motion).
- The host's speech bubble no longer covers his head.
- The four game buttons just say Play and are all the same size.
- Over the Edge's camera stays still behind its setup card.

### 3D upgrade
- **Outpace** has a proper studio set: a chunky step track, a finish arch, lighting rigs, a Final Sprint track with its own arch and a countdown clock in the set that turns the lights red for the last ten seconds. The camera follows the class during questions, the class surges and the Hunter lunges, and a gap meter shows the distance. Catches play in slow motion; escapes burst through the arch with confetti and a light show.
- **Category Clash** is a lit wall of 3D tiles: picked tiles flip into the question, answered tiles flip back in the winner's colour, and the star tile gets its own moment.
- **Hex Hunt** has raised hexagons that pop when claimed, edges that glow as a chain grows, and a winning chain that lights up edge to edge.
- The host has new reactions, including a gasp.
- Any moment longer than about a second can be skipped with Space or Enter. Reduced motion turns them into simple changes; Low graphics turns off the heavier effects.

### Fixed
- Decorative counters dropping behind Over the Edge's setup card could land in the real game.
- Category Clash's team panels were missing from the question card (since 1.4.0), so teams could only be marked by number key. They are back and can be tapped.


## 1.4.1 – Ten-minute games (October 2026)

### Changed
- **Every game takes 10 minutes at most with a class**, with time for teams to think, talk and write: Over the Edge 6 questions and a 4-question final; Outpace one Deal Round and the Final Sprint; Category Clash 4 categories of 3 questions (100, 200 and 300 points); Hex Hunt one round on a 4 × 4 board.
- **Over the Edge's jackpot is set for the number of teams**, so the class wins it about half the time whether there are 2 teams or 6.
- **Outpace's Final Sprint target** is retuned for the shorter game: about half of classes make it, and a bolder deal makes it harder.


## 1.4.0 – Simpler setup, subject choice and host menu (October 2026)

### New
- **Choose your subject first.** The menu now has a subject choice (Biology, Chemistry, Physics, Combined Science, Maths, English, History, Geography), an exam board choice (AQA or Edexcel) and the question set to use. Every game and the Question bank show only that subject's sets. All three are remembered.
- Subjects without a built-in pack yet say so and take your own sets. Your own sets now belong to a subject (sets made before this version show under every subject until you give them one).
- **Customise the host** from the menu, right under Marty, with a live preview. Changes apply in every game.

### Changed
- **Every setup card asks only for the number of teams.** Team names are behind "Edit team names", and the card shows the questions in use with a link back to the menu.
- **One set of rules in every game:** every team answers every question, for 2 to 6 teams. The Whole class / Small group switch is gone; two teams play the same way, which suits a pair or a tutoring session.
- Fixed settings, chosen for about 20 minutes each: a 20-second countdown before "show me"; Over the Edge 14 questions plus an 8-question final with the jackpot at Normal; Outpace two Deal Rounds and the Final Sprint; Category Clash a 5 × 5 board with one star tile; Hex Hunt one round on a 6 × 6 board.
- Questions from topics the playing teams got wrong before now come up a little more often automatically (the checkbox has gone).
- Outpace says "teams" rather than "groups", and the Final Sprint shows the steps so far against the target.
- The menu's game descriptions and the teacher guide describe the class rules.
- Starting a game by pressing Enter in a team-name box no longer leaves the keyboard in that box (the first Space could be typed into it instead of skipping the countdown).
- Over the Edge's results card no longer runs over the team panels on laptop screens.

### Removed
- The two-team turn-taking versions of each game: steals and penalties in Category Clash, buzzing in and best of three in Hex Hunt, each team's own deal and passes in Outpace, and Over the Edge's single-team turns, score cards and event feed.

### Testing
- New tests for the simple setup cards, the subject and exam board choice and the host menu on the main screen. All gameplay tests now play the class rules. The screenshot sweep shows every game with six teams (two halves in Hex Hunt), the host menu, and a subject with no built-in pack.


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
