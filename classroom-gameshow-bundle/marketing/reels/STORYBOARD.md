# Reels storyboard

Five vertical reels (1080×1920, 60 fps) for Instagram Reels, TikTok and YouTube Shorts. Every shot is a real game
take recorded frame by frame on a controlled clock, so each one can be re-rendered exactly or varied.

- **Rebuild everything:** `sh tools/video/reels/build.sh` (about 2 hours with software 3D). One reel: run its line
  from that script, then its two `mix.py` lines.
- **Safe zones:** `sh tools/video/reels/check.sh` draws the top 250 px and bottom 450 px (where the apps put the account
  name, caption and buttons) over contact sheets of every reel, in `marketing/reels/check/`.
- **Takes:** a take is a seed plus a fixed list of actions. Over the Edge takes were chosen with
  `search-ote.js` / `search-jackpot.js`, which replay many seeds with no pictures and score every moment a counter goes
  over: how long it hung on the edge first and how many followed it.
- **Camera:** the reel camera (`reel.js`) wraps the page's WebGL renderer and places the 3D camera just before each
  frame is drawn; the game code is not changed. World units: Over the Edge's tray front edge is z = 5.5, its surface
  y = 0, x from −6 to 6 (physics x 0–480 maps to (x − 240) × 0.025). Outpace's track runs along x, HOME at about x = −1.4,
  each step 1.52 apart.
- **Sound:** the games' own effects, recorded as they play (`sounds.json`) and rebuilt by `mix.py`; the music is a short original score for each reel (`tools/video/reels/score.py`, FluidSynth with the FluidR3_GM soundfont), timed to that reel's cuts: 1 suspense (drone, heartbeat, snare roll) then a stab on the fall; 2 a bouncy groove with a stab on every cut; 3 a calm pad and celesta, no drums; 4 a tense A minor chase that turns major as the class reaches HOME; 5 a long build to a fanfare on the jackpot. Each ends with the same fanfare when the call to action starts. Every reel has a version with game sounds only.

## 1. Will it fall? (`reel-1-will-it-fall`, looping)

| Time | Shot | Camera | Caption |
|---|---|---|---|
| 0–5.5 s | A counter hangs over the front edge, tilting, as the shelf pushes the pile behind it | Low at the front, slow push-in (keys in `shots/reel-1-will-it-fall.json`) | "Will it fall?" (0–5.5 s) |
| 5.35–8 s | It tips; five more follow it over the edge into the chute | Rises and pulls back just before the fall, so the drop is in view | none |
| 8–8.5 s | Freeze, then the loop cuts back to the teeter | Held | none |

Seed 11, six questions played (every team right, random lanes), recorded physics frames 3450–3930; the counter falls
at frame 3780 (x = 250, the centre). Script: `render-ote.js shots/reel-1-will-it-fall.json`.

## 2. POV: you turned GCSE revision into a game show (`reel-2-pov-game-show`)

Cuts every 4 beats of the theme (2.14 s at 112 bpm) from the other shots; the caption is on screen throughout.

| Time | Cut | From |
|---|---|---|
| 0–2.1 s | The cascade | Reel 1, from just before the fall |
| 2.1–4.3 s | The jackpot falls | Reel 5, 1 s before the fall |
| 4.3–6.4 s | The class bursts through the HOME arch | Reel 4, the escape |
| 6.4–8.6 s | Category Clash: the hidden star tile spins round, double points | `render-boards.js cc-star 5` (Chemistry, star found at column 4, row 2) |
| 8.6–10.7 s | Hex Hunt: the last hexagon links the path and it lights up | `render-boards.js hh-path 5` (Biology, the middle row) |
| 10.7–12.9 s | +£5,000 and confetti | Reel 5, after the fall |

Built by `montage.py` (cut list at the top of the script); the caption layer comes from `stills.js`.

## 3. oddly satisfying (`reel-3-oddly-satisfying`, looping)

Half-speed slow motion (the game's frame clock runs at half speed, so every frame is a real render), a low, close view
along the front edge (camera about 3 m back, just above the tray) as seven counters tumble over one after another. Counter
sounds only, no music, so the main file and the `-sfx-only` file are the same; small lower-case caption. Seed 3, physics
frames 5031–5361 (the cascade starts at 5211, x = 128). Camera slides slowly sideways.

## 4. The Hunter is coming (`reel-4-hunter`)

Outpace, Standard deal (Hunter three steps behind), seed 3, Physics questions. Camera straight down the track from
behind the Hunter, so the class is ahead of it and HOME beyond. Each answer's step is kept as a quick cut (1–1.3 s):
two wrong answers (the Hunter closes to one step behind: the near-catch), then six right answers; the class bursts through
the arch with the game's own confetti. Caption "One wrong answer and it's over…" until the escape; a small "Showtime"
mark in the last second. Script: `render-outpace.js 3` (plan and cut lengths at the top).

## 5. The £5,000 jackpot (`reel-5-jackpot`)

The final round (test shortcut straight to it), seed 2, the gold counter placed 0.6 of its radius short of falling;
every team right and random lanes, so everyone's counters push in behind it. It hangs for about 20 s of play; the reel
keeps the last 6.7 s before it goes, then the fall, "+£5,000" and confetti (recording-only overlay), with the
"Showtime" mark in the last second. Physics frames 877–1467; camera low and close, rising and pulling back just before
the fall so the gold counter tips over in full view. Caption "£5,000 jackpot. / Last question." until the fall.

## Files

| Reel | Length | With music | Game sounds only | Cover |
|---|---|---|---|---|
| 1. Will it fall? | 12.5 s | reel-1-will-it-fall.mp4 (6.7 MB) | reel-1-will-it-fall-sfx-only.mp4 (6.5 MB) | reel-1-will-it-fall-cover.jpg |
| 2. POV: game show | 16.9 s | reel-2-pov-game-show.mp4 (9.5 MB) | reel-2-pov-game-show-sfx-only.mp4 (9.3 MB) | reel-2-pov-game-show-cover.jpg |
| 3. oddly satisfying | 15.0 s | reel-3-oddly-satisfying.mp4 (5.3 MB, soft calm score) | reel-3-oddly-satisfying-sfx-only.mp4 (counter sounds only, 5.3 MB) | reel-3-oddly-satisfying-cover.jpg |
| 4. The Hunter is coming | 15.85 s | reel-4-hunter.mp4 (8.5 MB) | reel-4-hunter-sfx-only.mp4 (8.2 MB) | reel-4-hunter-cover.jpg |
| 5. The £5,000 jackpot | 14.23 s | reel-5-jackpot.mp4 (9.1 MB) | reel-5-jackpot-sfx-only.mp4 (8.9 MB) | reel-5-jackpot-cover.jpg |
| 6. Answer before it falls | 14.23 s | reel-6-answer.mp4 (7.0 MB) | reel-6-answer-sfx-only.mp4 (6.9 MB) | none yet |

Each length includes a 4 s call to action (`tools/video/reels/cta.js` and `cta.py`): the last frame dims and blurs, then the wordmark, a hook for that reel, "Turn your classroom into a game show", the gold "Play free now on Tes" button and "Link in bio" pop in, and the host slides up bottom right and waves. The tail breaks the loops of reels 1 and 3.

All H.264 1080×1920 at 60 fps with AAC audio. Safe-zone contact sheets: `check/`.

## Reel 6: Answer before it falls (14.2 s)

The jackpot take again (`shots/reel-6-footage.json`: reel 5's take with no caption and no end mark), with the text laid
over it by `reel6.js` and `reel6.py`: the hook "Year 11s: Answer before it falls" on frame 1; the question "The
mitochondria is the ___ of the cell" popping in at 0.1 s; a timer ring and numbers 7 to 1 that hit zero as the counter
falls (a tick each second, an extra-loud last one); on the fall a white flash, "✓ Powerhouse" and "Site of aerobic
respiration"; then "Be honest: did you get it?" and "Tag the friend who forgot", fading out before the call to action
(cta.py 6, "Try this with your class"). Music: the reel 5 score, which builds to the same fall. Why it is built this
way is in `POSTING.md`.
