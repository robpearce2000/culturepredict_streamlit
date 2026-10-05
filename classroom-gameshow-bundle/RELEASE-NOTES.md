# Showtime: Classroom Gameshows 1.0.0: release notes

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

**Questions:** 194 AQA-style GCSE Combined Science questions (Biology, Chemistry, Physics, all hand-tagged by tier) and a 63-question Homeostasis lesson set, plus the teacher's own sets for any subject. The packs are AQA-style practice questions written for Showtime; they are not produced or endorsed by AQA.

## What changed overnight

- **Outpace:** smoother (capped resolution, no shadows, an automatic quality step-down), the Final Sprint question in a side column on wide screens, an instant escape when the target is reached, racers that follow the subject, and a 3D studio: a chunky step track, a finish arch, a sprint track with a clock in the set (red for the last ten seconds), a camera that follows the class, a gap meter, slow-motion catches and escapes through the arch with confetti.
- **Over the Edge:** the correct teams take turns again, and whatever comes over the edge on a team's drop is theirs, in a random order shown on screen; the shelf starts emptier at the front, so the first drop of the game no longer pays a fortune; no more flickering under the sign; less glaring counters.
- **Category Clash:** columns are topics from your subject (choose up to four, or let it mix them); a real three-tier difficulty scale (recall, describe and apply, explain and extend) on every question; a lit wall of 3D tiles that flip, and a star-tile moment.
- **Hex Hunt:** every letter really is where the answer starts; one-press marking (the half with more right answers, or Neither); the gold keyboard outline only when the arrow keys are used; raised 3D hexes, glowing edges and a winning chain that lights up.
- **Every game:** a Start game button after setup, so no countdown runs before the class is ready; any big moment skips with Space or Enter.
- **Home screen:** compact Subject, Exam board and Questions dropdowns, the sign in the middle with twinkling bulbs, equal Play buttons, and a host who has no name unless you give him one.

## Known issues

- **Frame rate has only been measured without a graphics card.** The test machine draws with a software renderer (about 8–9 fps in Outpace); real laptops are far faster, and Outpace steps its own quality down if a computer struggles. Please check on the computers you'll use.
- **Over the Edge with six teams takes about a minute and a half longer than before.** Turn-by-turn drops take about 3 seconds per team (sometimes 6), so six teams use about 24 seconds per question for the drops; a six-team game should still finish just under 10 minutes. This is an estimate from the measured drop times; a full six-team game wasn't timed end to end, so check it in your classroom trial.
- **Luck matters in Over the Edge.** With "your drop, your counters", which team wins most depends on where counters land (over 12 test games teams ranged from about −30% to +60% of the average); dropping first is no advantage.
- **The jackpot is "about half the time" within sampling noise:** last measured at 45% (2 teams), 65% (4 teams) and 40% (6 teams) over 20 games each.
- **Untagged questions in teachers' own sets get an estimated tier** from their wording, which matches a hand tag about two times in three (and is within one tier 97% of the time); adding `Tier:` lines makes Category Clash exact.
- Not tested in Safari or on a real interactive whiteboard.

## Rob's checklist before selling

- [ ] **Classroom trials.** Play each game with at least one real class on the classroom PC and interactive whiteboard you will use (and ideally an older school laptop at Low graphics). Check that the 10-minute lengths, the 20-second countdown and the jackpot, sprint and board balance feel right with real pupils.
- [ ] **Frame rate on a real laptop.** The test machine has no graphics card; open Outpace at High graphics on your own laptop and check it runs smoothly (the game steps its quality down by itself if not).
- [ ] **Check the questions.** Read all 257 built-in questions and answers against the current AQA Combined Science: Trilogy specification, including the tier given to each one and the five questions added overnight (Chemical analysis, The atmosphere and resources, Nervous vs hormonal, Nervous system). Make sure the Hex Hunt letters and the "AQA-style" wording are right.
- [ ] **Trade mark searches.** Search the UK IPO (and EUIPO if selling beyond the UK) for "Showtime", "Over the Edge", "Outpace", "Category Clash" and "Hex Hunt" in classes 9, 28 and 41, and check that no listing text, image or file name suggests a link to any television programme, broadcaster or exam board.
- [ ] **Your employment contract.** Check what it says about intellectual property and outside work (resources made in your own time, using school equipment or for your own classes), and get written agreement from the school if needed before selling.
- [ ] **Tes setup.** Set up your Tes author shop and seller details, choose the price and licence, upload the HTML file, the PDF guide, the listing images and the description, and test-download it on a school computer (some school networks block downloading HTML files; a zip may be needed).
- [ ] Optional: try it in Safari and Firefox, and on a tablet, as well as Chrome and Edge.
