# Decisions

Choices made while building the first edition without the owner available. Each has a one-line reason. Anything here can be changed.

## Names and branding

| Decision | Reason |
|---|---|
| **Game 2 is called Outpace**, not the working title in the brief. | The working title is the name of a long-running SEGA racing video game series, so it clashes with an existing trade mark in the same class (game software). "Outpace" keeps the meaning and found no clash in a web search. |
| **Game 1 keeps the name Over the Edge.** | A web search found no UK classroom quiz product with this name. A 1990s tabletop role-playing game uses it, but in a different product category. |
| **Bundle name kept: The Classroom Gameshow Bundle.** | As briefed; no conflict found. |
| Show-specific terms replaced: "lanes 1–4", "wildcard counters" (marked ★), "the Hunter", "Deal Round" with Cautious / Standard / Bold deals, "Final Sprint", "pass" (was "help"). | Brief's suggested terms, plus deal names that say what they do. |
| The "Lab grant" wildcard is now "Cash bonus". | Works for any subject, not just science. |
| Over the Edge set redesigned: deep-teal studio, aqua and coral light strips, sunflower shelves with a hazard-stripe front edge, a dark slate peg board with brass pegs, an "OVER THE EDGE" header sign. Geometry, tiers, peg layout and every physics constant are unchanged. | Removes the lilac-and-gold set, red shelves, white board and header sign of the TV look. The physics code was diffed against v1: the only change is the field name `mystery` → `wildcard`. |
| Outpace runner is **gold** and the Hunter is **magenta**, on a deep-indigo track with a teal home cell. | The original blue-versus-red pairing echoed the TV board. Gold/magenta also stays distinct for red–green colour blindness. Atom visuals kept, as briefed. |
| Default host renamed **Professor Pip** with curly ginger hair, glasses, a teal suit and an orange bow tie (new). | The previous default carried the owner's name and a generic look; the bow tie gives the mascot an identity that is not modelled on any presenter. All customisation options kept. |
| Bundle identity: midnight navy, tangerine, sunflower and teal; wordmark built as SVG in code; display font Lilita One, body font Nunito. | Bright, readable from the back of a room, fully offline. |
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

## Things not done

| Item | Reason |
|---|---|
| Manual playtesting on a real interactive whiteboard and in Safari. | Not possible from the build environment; recommended before release. |
| Trade-mark searches beyond a quick web search. | A formal UK IPO search for "Over the Edge" and "Outpace" in classes 9, 28 and 41 is worth doing before selling. |
