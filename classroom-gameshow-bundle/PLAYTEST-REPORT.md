# Playtest report: simulated lessons with a class of 30

*Showtime: Classroom Gameshows 1.2.1, the shipped file `dist/showtime-classroom-gameshows.html`, October 2026.
This is a report only: no game code was changed.*

## Summary

33 simulated lessons, 34 games and 304 questions were played at real speed. The simulated teacher used only the
buttons and the keys in the teacher guide. The class was 30 students in 2, 4 or 6 teams, each team with its own
ability. All six subjects were covered, on both boards, plus Maths at 45 sec, 1 min 30 and 2 min. No console
errors appeared, and nothing froze. Every game stayed within 10 minutes for the non-Maths subjects. The problems
are mostly about what happens at the edges of a lesson: the end of a game, interruptions and quick double
presses. A few are about balance.

**The five most important problems**

1. **Outpace's Final Sprint loses the last question and then wipes the results** (Must fix). The 60-second clock
   runs out while the class is still writing or being marked. The game jumps straight to the results, and the
   teacher's next Enter, meant to confirm the marking, starts a new game. This happened in 4 of the 5 Outpace
   games played, so the class never sees how it did.
2. **Over the Edge runs at 3 to 4 frames a second without graphics acceleration** (Must fix). It runs at that
   speed at High and Low alike, and its physics then runs about 4 times slower than real time. Outpace detects
   this case and lightens itself; Over the Edge does not. Many school PCs and remote desktops run without
   acceleration.
3. **Category Clash shows the same questions all week** (Should fix, high). It prefers questions any playing team
   got wrong before. With 6 teams nearly every question is missed by someone, so on the same topic lessons 4 and
   5 were 100% repeats. Only 18 of the topic's 40 questions ever appeared.
4. **There is no pause, and the countdown keeps running behind the "Leave this game?" prompt** (Should fix). During
   a 2-minute interruption the countdown ran out and "show me" fired on its own (3 of 4 games). Behind the leave
   prompt it kept counting (18 s became 13 s). The guide's tip that "the show me only comes when you are ready" is
   not true: it fires at zero.
5. **Quick double presses skip choices** (Should fix). A double Enter on "Back to the board" in Category Clash and
   Hex Hunt opens the highlighted tile or hexagon before the captain has chosen. A double Space jumps past the
   whole countdown and the "3, 2, 1". Enter on any results screen restarts the game straight away.

**The five biggest opportunities**

1. **A Pause key (P) and a paused banner**, which also hold the countdown behind any prompt. Fire drills, a
   question from a student and a knock at the door are everyday events.
2. **"End game and show results" in every game.** Today only Category Clash can stop early and keep the scores,
   and only from the board. Menu discards the results in the other three games.
3. **A longer answer time for any subject, or automatically for calculations.** 208 of 1,509 Physics questions and
   60 of 1,403 Chemistry questions are calculations that get 20 seconds; the Maths answer-time choice would suit
   them too.
4. **Keep Hex Hunt close.** "The half with more right answers takes the hexagon" magnifies small differences: at
   65% against 55% right, the stronger half wins about two-thirds of hexagons and the weaker about a quarter.
   Proportional claims, a catch-up rule or best of three rounds would keep weaker halves in it. A best of three
   would also lengthen a game that often lasts only 3 to 4 minutes.
5. **Make the question the biggest thing on the Over the Edge screen while the class writes.** From the back of the
   room it is small text in a side panel, while the 3D machine fills about 70% of the screen.

## How the lessons were simulated

- **The file:** the shipped file in Chromium through Playwright, with no test shortcuts or speeded-up clocks. The
  teacher acted only through visible buttons and the guide's keys (Space, 1 to 6, Enter, U, R, 1 to 4, 1 to 3, F,
  Esc) and by clicking tiles and hexagons. The game's own read-only `state()` was used only to know which screen
  was showing and to log what happened. The harness is in `tools/playtest/`
  (`node tools/playtest/run-all.js tools/playtest/plan.json <out>`).
- **The teacher's pace:** 2.5 to 6 s to read a screen and act. Captains took 2.5 to 3.5 s to choose. On about a
  third of questions, writing time was cut short with Space; otherwise the countdown ran out. Marking took about
  1.2 s plus 0.9 s a team to look along the boards, and 0.45 s a key. The teacher spent 4 s reading out each
  answer.
- **The class:** team abilities of 85%, 40%, 70%, 55%, 62% and 48% (strong to weak). Accuracy dropped by 12
  points for each difficulty step, and a shared "tricky question" factor applied. In Hex Hunt each half had 15
  students, and the half with more right answers was pressed.
- **Classroom moments:** a mis-mark then Undo, a 2-minute interruption mid-question, an accidental Esc, a request
  to hear the question again, rapid double presses, ending early when time ran short, and a page refresh
  mid-game.
- **The hardest cases:** all teams always right, all teams always wrong, and two teams kept level.
- **The week:** the same History topic (AQA 1AA America 1840 to 1895, 40 questions) played in Category Clash five
  times, with the browser's memory kept between lessons.
- **A real-speed limit:** this container has no graphics chip, so the 3D games drew in software. Over the Edge ran
  at about 3 fps, and its physics clock, which slows when the frame rate drops below 15, ran about 4 times slow.
  Over the Edge times are therefore estimated for a normal-speed PC. Teacher and class time is taken from the
  real clock; counters dropping and settling come from the game's own clock, which is what a normal PC shows.
  Outpace's animations run on the real clock and were not affected.

## Over the Edge

| Severity | Finding |
|---|---|
| **Must fix before launch** | **3 to 4 fps without graphics acceleration, at High and Low alike.** At 1366×768 with Low graphics the median was 3.6 fps; at 1920×1080 with High it was 1.4 fps. In this state the physics clock runs about 4 times slow, so a 10-second drop takes about 40 seconds. In a classroom with hardware acceleration switched off, the game is unplayable. **Fix:** detect software rendering, as Outpace already does, and offer or force a lighter mode (fewer counters drawn, simpler materials) with a message. Check on a school PC with acceleration off. |
| **Should fix** | **The question is hard to read from the back.** At quarter size (`playtest-screenshots/readability/1366x768-ote-question-quarter.jpg`), the question is small type in the right-hand panel and the machine fills most of the screen. **Fix:** while the class writes, show the question large across the machine (as Outpace and Category Clash do), then shrink it back for the drop. |
| **Should fix** | **Ending early loses everything.** With time short, Menu then "Back to menu" ends the game with no results screen, so there is no "Reteach these" (`end-early-ote-end-early-modal.jpg`). **Fix:** add "End game and show results" to the Menu prompt, as in Category Clash. |
| **Should fix** | **Waiting while counters drop grows with the number of correct teams.** From marking to "settled", including captains choosing lanes, took 6 to 11 s with 2 teams, up to 20 s with 4 and up to 24.5 s with 6. It is a fun wait while the class watches, but it is the longest dead time in any game. **Fix:** consider a faster drop in Round 1 when four or more teams are correct, or let R (random lanes) be the default once three or more teams have dropped. |
| **Nice to have** | **Ties are shown as 1st and 2nd.** On the results screen, teams with the same money get different places (in the all-wrong game four teams on £0 were ranked 1 to 4). **Fix:** give equal scores the same place. |
| **Nice to have** | **Weaker teams can end on £0.** In the 4-team and 6-team games one team finished on £0 while the class pot was £6,800 and £8,100. The class jackpot moment keeps everyone involved, but a weak team's own row stays empty. |
| Works well | Undo after a slip worked, until the first counter drops, exactly as the guide says. Double presses during lane picking did no harm. The accidental-Esc prompt opens on "Keep playing". The jackpot fell in 3 of 6 games (2 of 3 normal-ability games), in line with the "about half" design. |

## Outpace

| Severity | Finding |
|---|---|
| **Must fix before launch** | **The sprint clock runs out mid-question; the question is lost, then Enter restarts the game.** In 4 of 5 games the 60-second clock reached zero while boards were being written or marked. The game went straight to the catch and the results, so that question's correct answers never counted. The teacher's next Enter, meant to confirm the marking, pressed Play again on the results, and a new game began. **Fix:** let the question on screen finish when the clock runs out (or stop accepting a new question after about 10 s), and make the results screen ignore keys for about 2 seconds. |
| **Must fix before launch** | **The sprint is too short at a realistic classroom pace.** Each sprint question took 30 to 33 s: about 15 to 20 s to write, 2.5 s for "show me", 6 to 8 s to mark and 1.4 s to move on. So only 2 questions fit into 60 seconds, not the 3 or 4 the target is tuned for. A class that got **every** question right (4 teams) reached 8 of the 9 steps and was caught. The normal-ability 2-team and 4-team classes reached 1 of 5 and 3 of 9. The only class that escaped was the Maths one, whose sprint lasts 6 minutes. **Fix:** tune targets to about 2.5 questions, give the sprint its own short countdown (10 to 12 s) so the pace is clear, or lengthen it to 90 s. |
| **Should fix** | **The Hunter is rarely a threat in the Deal Round.** In all three normal games the runner reached home; the Hunter gained only 2 steps in the 2-team and 4-team games and none with 6 teams. "At least half the teams correct" is nearly always met by an average class. **Fix:** the Hunter could also step forward every second question, or require more than half correct. |
| **Should fix** | **No pause.** In the 2-minute Maths game, a 2-minute interruption used up the question's whole answer time. |
| **Nice to have** | **"Also accept" text is very small from the back** (`1366x768-op-marked-quarter.jpg`). The question and the team results read well. |
| Works well | The Final Sprint card explains the rules clearly (`op-sprint-card.jpg`). Undo and the accidental-Esc prompt behaved well. About 22 to 29 fps even in software rendering, thanks to the existing quality step-down. Waits between questions were short (1.6 s while the racers move). |

## Category Clash

| Severity | Finding |
|---|---|
| **Should fix** | **The same questions all week.** Lessons 1 to 5 on one topic asked 9 questions each. Repeats from earlier lessons: 0, 1, 8, 9, 9. Only 18 of the 40 questions ever appeared. The board picks questions a playing team got wrong before, and with 6 teams almost every question is wrong for someone. **Fix:** prefer questions not asked recently, and limit "missed before" questions to two or three per board. |
| **Should fix** | **End game can't be reached during a question.** With time short, the End game button sits under the open question panel and cannot be clicked. From the board it works and the results are kept (`cc-ended-early-results.jpg`). **Fix:** keep End game above the question panel, or add it to the question's buttons. |
| **Should fix** | **A double Enter skips the captain's choice.** Enter on "Back to the board" followed straight away by another Enter opens whichever tile is highlighted. A double Space then skips the whole countdown. **Fix:** ignore Enter on the board for about half a second after returning, and treat a Space within half a second of the last one as a repeat. |
| **Should fix** | **The result is often decided after the first question.** In 5 of 11 complete games the team leading after question 1 never lost the lead. This is driven by the strong team (85%); the picking team getting full points while others get half adds to it. **Fix (optional):** let the last-placed team pick the final row, or make the star tile triple for the team in last place. |
| **Nice to have** | **A one-topic board can be 3×3.** The History topic has three subtopics, so the board had 9 tiles and the game took about 6 minutes. It works well, but it is shorter than the guide's "4 categories of 3 questions". |
| **Nice to have** | **"Team 1 win with 100 points"** on the results: "wins" reads better for a single team. |
| Works well | Everything else. Questions and answers are the most readable of all four games from the back (`1366x768-cc-marked-quarter.jpg`); undo worked; ties are handled ("It's a draw!", shared places); 60 fps throughout. |

## Hex Hunt

| Severity | Finding |
|---|---|
| **Should fix** | **No ending when nobody can answer.** In the all-wrong lesson nobody won a hexagon, so the game ran 31 questions over 15 minutes and would have gone on. **Fix:** after about 15 questions, or with an End game button, finish with the half holding more hexagons. |
| **Should fix** | **Small differences become one-sided games.** One press decides each hexagon, so per question: equal halves (60% against 60%) split 43% each; at 65% against 55% the split is 65% to 23%; at 70% against 50% it is 83% to 9%. In the simulation, where the halves were deliberately unequal (85% against 40%), the stronger half took every hexagon in all six games. **Fix:** tell teachers to balance the halves (the guide says nothing about it), and consider a catch-up rule (the half that is behind wins a level count). |
| **Should fix** | **The edges rely on colour alone.** The four edge bars are blue (left and right) and orange (top and bottom), with no ● or ■ mark, and the "left to right / top to bottom" note is tiny from the back (`1366x768-hh-board-quarter.jpg`). This contradicts the guide's "Nothing relies on colour alone". **Fix:** put ● on the blue edges and ■ on the orange ones. |
| **Should fix** | **A double Enter skips the captain's choice**, as in Category Clash: Enter after a claim, pressed twice, opens the highlighted hexagon. |
| **Nice to have** | **Short games.** Normal games lasted 3 to 6 minutes (6 to 11 questions). Best of three rounds would fill a 10-minute slot and give a losing half another chance. |
| Works well | Letters are big and clear from the back. Claims now change colour on the board where everyone sees them. The winning chain takes 2.2 s. 60 fps. |

## All games: teacher experience, guide, robustness

| Severity | Finding |
|---|---|
| **Should fix** | **No pause; the countdown runs behind prompts.** See the summary. The leave prompt is the right default ("Keep playing" is focused), but the countdown kept going behind it. |
| **Should fix** | **The guide's thinking-time tip is wrong.** The guide says "Need more thinking time? Let the countdown run out, or wait before pressing Space: the show me only comes when you are ready", but "show me" fires automatically at zero (checked in a game left untouched for 20 s). Either change the guide or let the countdown end in a "Time's up, press Space when ready" state. |
| **Should fix** | **Refreshing the page mid-game loses the game.** In all four games, a refresh went back to that game's setup screen with the game gone and no warning (`refresh-*-after-refresh.jpg`). **Fix:** a "leave this page?" warning while a game is in progress, or remembering the game in progress. |
| **Nice to have** | **No way to show the answer before marking.** The answer only appears after Confirm. Some teachers will want to reveal it, then mark; a key (A) to reveal it during marking would help. |
| **Nice to have** | **No way to add time when a team asks for the question again.** The countdown cannot be extended; a +10 s key would help. |
| **Nice to have** | **Guide wording.** It says "Press Start or Enter", but the button reads "Start the game". It mentions "sets made before version 1.4", but the product is 1.2.1. In full screen, the first Esc only leaves full screen, so the Menu needs a second Esc (as designed; the guide's Esc line could say so). |
| Works well | No console errors in any of the 33 lessons (only the browser's own software-3D warnings). No freezes or stuck screens, apart from the Outpace restart above. Undo was available and correct every time it was tried. The accidental-Esc prompt never left a game without a second confirmation. Running two games back to back (Category Clash then Hex Hunt) was smooth, with about 5 seconds to change game. Full screen works from the button and from F. |

## Readability from the back of the room

Screens were taken at 1920×1080 and 1366×768, normal and full screen, and also shrunk to a quarter of the area
(`playtest-screenshots/readability/*-quarter.jpg`). In full screen the game uses the same space as at that
window size, so the findings are the same.

- **Easy to read:** Category Clash questions, answers and scores; Hex Hunt letters; Outpace questions and team
  results; every team panel's ✓ and ✗ marks.
- **Too small:** the Over the Edge question; Outpace's and Category Clash's "Also accept" line; Hex Hunt's team
  bar ("left to right · 0 hexagons"); the keyboard hints at the bottom, which are for the teacher and don't
  matter.
- **Colour alone:** Hex Hunt's edge bars. Everything else pairs colour with a mark (● ■ ▲ ◆ ⬢ ✚, ✓ ✗).
- **Glare:** the Outpace track and the dark backgrounds of Over the Edge and Hex Hunt have light text on dark
  navy. That is high contrast on a screen, but it can wash out on a dim projector; Category Clash's cream question
  card holds up best.

## Questions in play

- **No repeats within a game:** 276 different questions came up across 33 lessons, and none was asked twice in
  the same game.
- **Across a week:** heavy repeats in Category Clash (see above). Over the Edge, Outpace and Hex Hunt only lean
  towards weak topics some of the time, so they should repeat much less, though this run did not measure it.
- **Length:** questions are short (median 12 to 16 words, longest 32), and answers are 1 or 2 words (longest 10).
  The longest takes about 10 seconds to read aloud; most take under 5.
- **Tight for 20 seconds:** explanation answers of 9 or 10 words (14 in the packs, for example "Carried by
  ice, not water, so not rounded or sorted"), and calculations: about 14% of Physics questions and 4% of
  Chemistry questions.
- **Maths:** the longer answer time works well. Games last longer, as the guide now says: Over the Edge with 6
  teams at 1 min 30 took about 19 minutes; Category Clash at 45 sec took about 11.5 minutes (13.5 minutes with
  the interruption).
- **Fit to each game:** Category Clash rows matched difficulty. Hex Hunt letters were all genuine first letters,
  with a mix of letters on each board.
- **Facts:** a spot check of 45 questions that came up found no errors. The Mappleton question should also
  accept "rock groynes" (added to QUESTIONS-TO-CHECK.md).

## Timing table

Times run from the Start game button to the results. Over the Edge times are estimated for a normal-speed PC (see
"How the lessons were simulated"). Per-question time covers the whole game divided by the number of questions.
Writing, show and marking are averages per question.

| Lesson | Game | Teams | Subject | Questions | Total | Per question | Writing | Show me | Marking | Longest wait |
|---|---|---|---|---|---|---|---|---|---|---|
| ote-2t | Over the Edge | 2 | Biology AQA (High, 1920) | 10 | **6:53** | 41 s | 20.1 s | 2.9 s | 7.8 s | marking to settled 11 s |
| ote-4t | Over the Edge | 4 | Physics Edexcel | 8 | **8:11** (6:10 without the 2-min pause) | 61 s | 34 s | 2.6 s | 9.1 s | 20 s |
| ote-6t | Over the Edge | 6 | Maths AQA, 1 min 30 | 10 | **18:56** | 114 s | 80 s | 2.8 s | 12.4 s | 24.5 s |
| edge all right | Over the Edge | 4 | Chemistry Edexcel | 8 | 7:12 | 54 s | 21.6 s | 3.0 s | 10.8 s | 22 s |
| edge all wrong | Over the Edge | 4 | Geography AQA | 10 | 6:27 | 39 s | 20.1 s | 3.1 s | 6.5 s | none |
| edge tie | Over the Edge | 2 | Biology AQA | 10 | 7:11 | 43 s | 21 s | 2.9 s | 5.8 s | 9 s |
| op-2t | Outpace | 2 | Chemistry AQA (High, 1920) | 10 | 5:20 | 32 s | 15.7 s | 2.5 s | 4.5 s | 1.6 s |
| op-4t | Outpace | 4 | History Edexcel | 11 | 6:39 | 36 s | 17.6 s | 2.6 s | 7.4 s | 1.6 s |
| op-6t | Outpace | 6 | Maths Edexcel, 2 min | 9 | 18:58 (about 17 without the pause) | 126 s | 106 s | 2.2 s | 9.8 s | 1.6 s |
| edge all right | Outpace | 4 | Physics AQA | 9 | 5:49 | 39 s | 19.1 s | 2.5 s | 8.0 s | 1.6 s |
| edge all wrong | Outpace | 4 | Biology Edexcel | 6 | 3:41 | 37 s | 18.2 s | 2.4 s | 5.8 s | 1.7 s |
| cc-2t | Category Clash | 2 | Geography AQA | 12 | 7:03 | 35 s | 19.2 s | 2.8 s | 4.7 s | none over 2 s |
| cc-4t | Category Clash | 4 | Biology Edexcel (High, 1920) | 11 | 6:36 | 36 s | 17.6 s | 2.7 s | 6.6 s | none over 2 s |
| cc-6t | Category Clash | 6 | Maths AQA, 45 sec | 12 | 13:32 (about 11:30 without the pause) | 68 s | 47.7 s | 2.4 s | 8.8 s | none over 2 s |
| edge all right | Category Clash | 6 | History Edexcel | 12 | 8:20 | 42 s | 20.1 s | 2.7 s | 10.2 s | none over 2 s |
| edge tie | Category Clash | 4 | Chemistry AQA | 12 | 6:52 | 34 s | 16.8 s | 2.7 s | 6.1 s | none over 2 s |
| week 1 to 5 | Category Clash | 6 | History AQA, one topic | 9 each | 5:34 to 5:55 | 37 to 39 s | 16.5 to 18.7 s | 2.7 s | 9.0 to 9.3 s | none over 2 s |
| hh-chem | Hex Hunt | 2 halves | Chemistry Edexcel | 8 | 4:25 | 33 s | 17.4 s | 2.7 s | 4.0 s | winning chain 2.2 s |
| hh-hist (1) | Hex Hunt | 2 halves | History AQA | 11 | 8:03 (6:00 without the pause) | 44 s | 28.4 s | 2.4 s | 4.0 s | 2.1 s |
| hh-hist (2) | Hex Hunt | 2 halves | History AQA | 7 | 4:08 | 35 s | 19.5 s | 2.7 s | 4.0 s | 2.2 s |
| hh-geog | Hex Hunt | 2 halves | Geography Edexcel (High, 1920) | 6 | 3:13 | 32 s | 16.1 s | 2.7 s | 4.0 s | 2.1 s |
| hh-maths | Hex Hunt | 2 halves | Maths Edexcel, 1 min 30 | 7 | 10:33 | 90 s | 74.3 s | 2.8 s | 4.0 s | 2.2 s |
| edge all wrong | Hex Hunt | 2 halves | Geography Edexcel | 31 (no end) | 15:10, stopped | 29 s | 17.5 s | 2.7 s | 4.0 s | none |
| back to back | Category Clash, then Hex Hunt | 6, then 2 | Physics AQA | 12 + 11 | 7:46 + 6:07 | 39 s, 33 s | 18 s | 2.7 s | 9.5 s, 4.0 s | 2.2 s |

## Fairness table

| Lesson | Game | Final scores | Spread (1st to last) | Lead last changed at | Outcome |
|---|---|---|---|---|---|
| ote-2t | Over the Edge | £100 / £800 | £700 | question 6 of 10 | jackpot not won (class £1,300) |
| ote-4t | Over the Edge | £600 / £0 / £100 / £300 | £600 | question 1 of 8 | jackpot won (class £6,800) |
| ote-6t Maths | Over the Edge | £200 / £200 / £350 / £0 / £0 / £550 | £550 | question 5 of 10 | jackpot won (class £8,100) |
| all right | Over the Edge | £100 / £400 / £300 / £200 | £300 | question 5 of 8 | jackpot won (class £6,600) |
| all wrong | Over the Edge | £0 each | £0 | (none) | not won (£0); shown as places 1 to 4 |
| tie | Over the Edge | £0 / £600 | £600 | question 4 of 10 | not won (class £800) |
| op-2t | Outpace | (class game) | | | caught in the sprint: 1 of 5 steps |
| op-4t | Outpace | (class game) | | | caught: 3 of 9 |
| op-6t Maths | Outpace | (class game) | | | **escaped: 14 of 14, pot 600** |
| all right | Outpace | (class game) | | | **caught: 8 of 9**, even with every answer right |
| all wrong | Outpace | (class game) | | | caught: 0 of 8 |
| cc-2t | Category Clash | 1,050 / 450 | 600 | question 1 of 12 | decided early |
| cc-4t | Category Clash | 1,100 / 250 / 1,050 / 850 | 850 | question 7 of 11 | close finish between two teams |
| cc-6t Maths | Category Clash | 1,450 / 800 / 750 / 850 / 900 / 150 | 1,300 | question 1 of 12 | decided early |
| all right | Category Clash | 1,600 / 1,600 / 1,500 / 1,550 / 1,700 / 1,500 | 200 | question 11 of 12 | very close |
| tie | Category Clash | 850 / 950 / 0 / 0 | 950 | question 7 of 12 | two leading teams level until late |
| back to back | Category Clash | 1,300 / 1,000 / 1,200 / 1,250 / 1,100 / 1,050 | 300 | question 11 of 12 | very close |
| week 1 to 5 | Category Clash | for example 1,250 / 250 / 1,050 / 500 / 1,100 / 750 | 500 to 1,050 | question 1 of 9 in weeks 3 to 5 | decided early in 3 of 5 |
| all six games | Hex Hunt | 6 to 11 hexagons against 0 | | first claim | the stronger half (85%) beat the weaker (40%) every time; see the calculation in the Hex Hunt section for balanced halves |

**Win rates.** Over the Edge jackpot: 3 of 6 games (2 of 3 normal-ability). Outpace class escape: 1 of 5 (1 of 3
normal-ability, and that was the Maths game with a 6-minute sprint). Category Clash: decided after the first
question in 5 of 11 complete games, and close to the end in 4.

## Frame rates

Measured in this container, which has **no graphics chip** (software 3D). A classroom PC with acceleration on will
be much faster; these figures show the worst case, a PC with acceleration off.

| Game | High graphics | Low graphics |
|---|---|---|
| Over the Edge (1366×768) | median 3.4 fps | median 3.6 fps (Low barely helps) |
| Outpace (1366×768) | median 28 fps (lowest 5 s: 23) | median 29 fps (lowest 23) |
| Category Clash, Hex Hunt | 60 fps | 60 fps |

## Limits of this test

- **Fun, noise and energy can't be simulated.** Whether students cheer, argue, lose interest during Over the
  Edge's drops, or find the Hunter exciting can only be seen in a real lesson.
- **The class is a model.** Real teams copy each other, misread questions, argue about whether "nearly right"
  counts, and take longer when a topic is new. Real teachers mark faster or slower than the 1 to 2 s a team used
  here, and talk between questions; real lessons will run longer than these timings.
- **Software 3D in this container.** Over the Edge timings are estimates for a normal-speed PC, and the frame
  rates above are a worst case. Please check both 3D games on the classroom PC, with acceleration on and off.
- **No projector or real room.** The quarter-size screenshots are only a rough stand-in for the back of a room,
  and glare or a dim projector can't be judged from a screenshot.
- **Small samples.** 3 to 6 games per game type. The fairness figures show tendencies, not reliable rates, and
  the Hex Hunt halves were deliberately unequal.
- **Not tested:** touchscreens and interactive whiteboards (all clicks were mouse clicks), Safari and Firefox,
  sound, very old PCs, and a class's reaction to the host's lines.

**What to watch for in your own lessons:** how long the Over the Edge drops feel with 6 teams; whether the Outpace
sprint feels winnable; how evenly Hex Hunt halves are matched; whether the back row can read the Over the Edge
question; and any moment you want to pause.
