# Showtime: Classroom Gameshows 1.1.0: release notes

## What's in the bundle

| File | What it is |
|---|---|
| `dist/showtime-classroom-gameshows.html` | The product: one offline HTML file with all four games, the question bank and the teacher's settings. Double-click to open; no internet, no installation, no accounts, no tracking. |
| `dist/teacher-guide.pdf` | The teacher guide (A4): getting started, running each game with a class, keyboard shortcuts, adding your own questions, settings, privacy and troubleshooting. |
| `dist/listing/` | The Tes listing: `description.md` (the listing text) and the images `cover.png`, `over-the-edge.png`, `outpace.png`, `category-clash.png`, `hex-hunt.png` and `question-bank.png`. |

**The games** (every one whole-class, about 10 minutes, inspired by classic TV quiz formats):

- **Over the Edge** (2 to 6 teams): every correct team drops a counter onto a moving 3D shelf; counters pushed over the edge win prize money; the whole class plays the final for the jackpot.
- **Outpace** (the class as one runner, 2 to 6 teams): vote for a deal, answer your way home before the Hunter catches you, then a 60-second Final Sprint.
- **Category Clash** (2 to 6 teams): four topics from your subject, three tiers of difficulty, a hidden double-points star tile.
- **Hex Hunt** (two halves of the class): lettered hexagons, each letter the first letter of the answer; the first half to link its edges wins.

**Questions:** AQA-style GCSE Biology (8461), Chemistry (8462) and Physics (8463) packs, one per specification topic, 1,425 questions, plus the teacher's own sets for any subject. The packs are AQA-style practice questions written for Showtime; they are not produced or endorsed by AQA.

## What changed in 1.1.0

- **Topic packs for AQA GCSE Biology, Chemistry and Physics:** one pack per specification topic (25 packs, 1,425 questions), every question tied to a specification point, with a difficulty and a Higher-tier tag, written from the specification and checked separately. They replace the Combined Science starter packs. Progress on the other boards and subjects is in QUESTION-BANK-PROGRESS.md.
- **Questions dropdown:** tick one topic, several, or "Mixed: all topics"; a Higher tier switch for Foundation classes; Category Clash uses subtopics as columns when one topic is ticked; accepted answers and teacher's notes shown under every answer.
- **Subjects:** Biology, Chemistry, Physics, Maths, History, Geography and Other (Combined Science and English removed; sets filed under them show under Other).

## What changed in 1.0.1

- **Outpace is smooth on real laptops.** About twice the frame rate (no full-screen glow pass or moving lights, simpler shading, standard resolution), and no more jitter or jolts when the camera moves: the picture slides clear of the question panel instead of stepping, the camera eases in and out, everything is loaded onto the graphics chip before play, the answer shake is a smooth sway, and all movement runs on the clock so it is the same on 60, 120 and 144 Hz screens. Checked by Rob on his laptop.
- **The home-screen sign** has a running marquee: a band of light runs round the bulbs.

## What changed overnight (1.0.0)

- **Outpace:** smoother (capped resolution, no shadows, an automatic quality step-down), the Final Sprint question in a side column on wide screens, an instant escape when the target is reached, racers that follow the subject, and a 3D studio: a chunky step track, a finish arch, a sprint track with a clock in the set (red for the last ten seconds), a camera that follows the class, a gap meter, slow-motion catches and escapes through the arch with confetti.
- **Over the Edge:** the correct teams take turns again, and whatever comes over the edge on a team's drop is theirs, in a random order shown on screen; the shelf starts emptier at the front, so the first drop of the game no longer pays a fortune; no more flickering under the sign; less glaring counters.
- **Category Clash:** columns are topics from your subject (choose up to four, or let it mix them); a real three-tier difficulty scale (recall, describe and apply, explain and extend) on every question; a lit wall of 3D tiles that flip, and a star-tile moment.
- **Hex Hunt:** every letter really is where the answer starts; one-press marking (the half with more right answers, or Neither); the gold keyboard outline only when the arrow keys are used; raised 3D hexes, glowing edges and a winning chain that lights up.
- **Every game:** a Start game button after setup, so no countdown runs before the class is ready; any big moment skips with Space or Enter.
- **Home screen:** compact Subject, Exam board and Questions dropdowns, the sign in the middle with twinkling bulbs, equal Play buttons, and a host who has no name unless you give him one.

## Known issues

- **Frame rate:** Outpace has been checked on Rob's laptop. On school computers with hardware acceleration switched off, the browser draws 3D in software; Outpace detects this and starts in its lightest mode, but it will be less smooth. Please check on the computers you'll use.
- **Over the Edge takes longer with more teams.** Turn-by-turn drops take about 3 seconds per team (sometimes 6). Whole games, timed with the game's own clock plus 38 seconds of classroom time per question (20 s thinking, 3 s show me, 10 s marking, 5 s reading the answer) and a minute for the intro and results: 2 teams about 8.6 minutes, 4 teams about 8.6, 6 teams 9.3 to 10.0. A slow class with six teams may run a little over 10 minutes, which is fine.
- **Luck matters in Over the Edge.** With "your drop, your counters", which team wins most depends on where counters land (over 12 test games teams ranged from about −30% to +60% of the average); dropping first is no advantage.
- **The jackpot is "about half the time" within sampling noise:** last measured at 45% (2 teams), 65% (4 teams) and 40% (6 teams) over 20 games each.
- **Untagged questions in teachers' own sets get an estimated tier** from their wording, which matches a hand tag about two times in three (and is within one tier 97% of the time); adding `Tier:` lines makes Category Clash exact.
- Not tested in Safari or on a real interactive whiteboard.

## Rob's checklist before selling

- [ ] **Classroom trials.** Play each game with at least one real class on the classroom PC and interactive whiteboard you will use (and ideally an older school laptop at Low graphics). Check that the 10-minute lengths, the 20-second countdown and the jackpot, sprint and board balance feel right with real pupils.
- [x] **Frame rate on a real laptop.** Outpace checked smooth on Rob's laptop (1.0.1). Still worth a look on the classroom PC.
- [ ] **Check the questions.** Read QUESTIONS-TO-CHECK.md first (nothing open for the AQA sciences yet), then sample the 1,425 built-in questions against the AQA Biology, Chemistry and Physics specifications (each question names its specification point). Make sure the "AQA-style" wording is right.
- [ ] **Trade mark searches.** Search the UK IPO (and EUIPO if selling beyond the UK) for "Showtime", "Over the Edge", "Outpace", "Category Clash" and "Hex Hunt" in classes 9, 28 and 41, and check that no listing text, image or file name suggests a link to any television programme, broadcaster or exam board.
- [ ] **Your employment contract.** Check what it says about intellectual property and outside work (resources made in your own time, using school equipment or for your own classes), and get written agreement from the school if needed before selling.
- [ ] **Tes setup.** Set up your Tes author shop and seller details, choose the price and licence, upload the HTML file, the PDF guide, the listing images and the description, and test-download it on a school computer (some school networks block downloading HTML files; a zip may be needed).
- [ ] Optional: try it in Safari and Firefox, and on a tablet, as well as Chrome and Edge.
