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
- **Sound:** the games' own effects, recorded as they play (`sounds.json`) and rebuilt by `mix.py`; the music is the
  Showtime theme (the launcher loop, draft B, 112 bpm) from `tools/music/loop.py`. Every reel has a version with game sounds only.

## 1. Will it fall? (`reel-1-will-it-fall`, looping)

| Time | Shot | Camera | Caption |
|---|---|---|---|
| 0–5.5 s | A counter hangs over the front edge, tilting, as the shelf pushes the pile behind it | Low at the front, slow push-in from 4 m to 2 m (keys in `shots/reel-1-will-it-fall.json`) | "Will it fall?" (0–5.6 s) |
| 5.5–8 s | It tips; five more follow it over the edge | Pulls back and tilts down to follow the fall | none |
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
along the front edge as seven counters tumble over one after another. Counter sounds only, no music; small lower-case
caption. Seed 3, physics frames 5031–5361 (the cascade starts at 5211, x = 128). Camera slides slowly sideways.

## 4. The Hunter is coming (`reel-4-hunter`)

Outpace, Standard deal (Hunter three steps behind), seed 3, Physics questions. Camera straight down the track from
behind the Hunter, so the class is ahead of it and HOME beyond. Each answer's step is kept as a quick cut (1–1.3 s):
two wrong answers (the Hunter closes to one step behind: the near-catch), then six right answers; the class bursts through
the arch with the game's own confetti. Caption "One wrong answer and it's over…" until the escape; a small "Showtime"
mark in the last second. Script: `render-outpace.js 3` (plan and cut lengths at the top).

## 5. The £5,000 jackpot (`reel-5-jackpot`)

The final round (test shortcut straight to it), seed 2, the gold counter placed 0.6 of its radius short of falling;
every team right and random lanes, so everyone's counters push in behind it. It hangs for about 20 s of play; the reel
keeps the last 6.5 s before it goes, then the fall, "+£5,000" and confetti (recording-only overlay), with the
"Showtime" mark in the last second. Physics frames 877–1467; camera low and close, pulling back for the fall.
Caption "£5,000 jackpot. Last question." until the fall.
