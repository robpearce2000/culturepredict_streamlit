# Demo videos: storyboard

Everything here can be rebuilt with one command: `sh tools/video/demo/build.sh` (a few hours on a 4-core machine with
software 3D). Each shot replays a fixed take: a fixed random seed, the same scripted teacher actions, and a page
clock that moves exactly 1/60 s per frame, so a shot can be rendered again after a game update and still match.

| File | Length | Size | What it is |
|---|---|---|---|
| `showtime-demo-60s-1080p.mp4` | 60.0 s | 28.6 MB | the main Tes demo, 1920×1080, 60 fps, H.264 + AAC |
| `showtime-demo-60s-1080p-silent.mp4` | 60.0 s | 27.1 MB | the same, no audio track |
| `showtime-reel-9x16.mp4` | 18.0 s | 9.6 MB | vertical cut, 1080×1920, for Instagram, TikTok and Shorts |
| `clip-over-the-edge.mp4` | 15.0 s | 11.5 MB | one game's best moment, 16:9 |
| `clip-outpace.mp4` | 15.0 s | 5.3 MB | 〃 |
| `clip-category-clash.mp4` | 15.0 s | 1.8 MB | 〃 |
| `clip-hex-hunt.mp4` | 15.0 s | 2.0 MB | 〃 |
| `thumbnail.png` | | | YouTube thumbnail, 1280×720 |
| `captions.srt` | | | the main video's on-screen text, timed |
| `showtime-theme.wav` / `.mp3` / `.mid` | 60.0 s | | the music on its own (script: `tools/video/music.py`) |

All footage is the Science mega pack's own test build (`test-build/editions/science.html`) on High graphics. Team
names are Electrons, Neutrons, Protons, Photons, Ions and Atoms. No people, faces, schools or show names appear.

## The shots (`marketing/video/work/<shot>/`, scripts in `tools/video/demo/`)

| Shot | Script | Seed | Game state and script |
|---|---|---|---|
| `hook-a`, `hook-b` | `reels/render-ote.js shots/hook-a.json`, `shots/hook.json` | 11 | Over the Edge, the reel 1 take (6 questions, every team right, random lanes): a counter hangs over the front edge from physics frame 3630; it tips at 3780 and the cascade follows. Camera: a slow push-in on the edge, then up and back to the whole machine. "+£500" pops on the fall and fades after 1.6 s. Two parts: hook-a is the teeter, hook-b starts one frame before the fall (same take, camera keys moved to match). |
| `jackpot` | `reels/render-ote.js shots/jackpot.json` | 2 | The final (the reel 5 take): the gold jackpot counter set 0.6 of its radius short of falling, every team right; it falls at physics frame 1277, with confetti and "+£5,000". |
| `ote` | `shot-ote.js 4` | 4 | AQA Biology, 6 teams. "Why do veins have valves?"; Space ("3, 2, 1, show me!"); teams marked ✓ ✓ ✗ ✓ ✓ ✗; confirm (the answer shows); random lanes; the counters drop down the pegs. Cues: ask, showme, mark, confirm, lanes. |
| `outpace`, `outpace-sprint` | `shot-outpace.js` | 3 | AQA Physics, 6 teams, Standard deal. Right (a step), wrong (the Hunter gains), then five right to HOME and "Escaped!". Then a new game straight to the Final Sprint: the clock running and a step. Only the moves are recorded. |
| `cc` | `shot-boards.js cc 5` | 5 | AQA Chemistry, 4 teams. A first pass finds the hidden ★ tile (Chemical analysis, 200); on a fresh board Electrons pick it: it spins round with gold light, worth 400. "Describe the test for hydrogen and its positive result." Three of four right; the scores jump. |
| `hh` | `shot-boards.js hh 5` | 5 | AQA Biology. Electrons claim the middle row: "three" fatty acids claims a T hexagon; the last claim lights the winning path and "Electrons take round 1!". |
| `launcher` | `shot-launcher.js` | 1 | The Science mega pack launcher. An on-screen pointer (recording only) picks Chemistry, Edexcel, Combined Science and four topics, then plays Category Clash and starts it. |
| `results` | `shot-results.js 2` | 2 | AQA Physics Category Clash, 4 teams, five questions with mixed marks, then End game: the class results with "Reteach these". |
| `v-ote` | `shot-vertical.js` | 4 | Over the Edge in its own portrait layout (1080×1920): six teams marked and confirmed. |
| `endcard` | `endcard.js` | | Wordmark, a still from each game, "Try it free on Tes", "No logins · Works offline", the host waving. Also draws `thumbnail.png`. |
| reels' takes | `tools/video/reels/` | | The vertical reel reuses the reels' vertical footage (see `marketing/reels/STORYBOARD.md`). |

Captions and logos are drawn by the app in its own fonts (`cards.js`), then laid over the footage with a quick
pop-in by the edit (`edit.py`), which also rebuilds the games' own sounds in sync and mixes in the theme.

## The main video (60 s)

| Time | Footage (shot, from) | Caption |
|---|---|---|
| 0–2.5 s | hook-a: the teeter, slow push-in | none |
| 2.5–6.4 s | hook-b: it tips, the cascade, "+£500"; the camera pulls back to the machine | 3.0–6.4 s: title card (wordmark, "Whole-class GCSE revision", the host) |
| 6.4–11 s | ote, from 1.4 s before "show me": the question, "3, 2, 1, show me!" | Every student answers every question |
| 11–15 s | ote, from marking: ✓ ✓ ✗ ✓ ✓ ✗ in seconds, confirm, the answer | Mark the whole class in seconds |
| 15–19 s | ote, from lanes + 0.8 s: counters bounce down the pegs | Over the Edge logo |
| 19–22 s | jackpot, from 1.8 s before the fall: teeter, fall, confetti, +£5,000 | none (music dips for the fall) |
| 22–30 s | outpace: surge (1.4 s), Hunter (1.6 s), sprint clock (1.6 s), sprint step (1 s), HOME (2.4 s) | Outpace logo · Race home before the Hunter catches you |
| 30–37 s | cc (framed): the star tile spins (3.6 s); marked, scores jump (3.4 s) | Category Clash logo |
| 37–44 s | hh (framed): a hexagon changes colour (2.6 s); the winning path (4.4 s) | Hex Hunt logo |
| 44–50 s | launcher: subject, board, course, topics, Play, Start | AQA & Edexcel · Separate & Combined / Biology · Chemistry · Physics |
| 50–55 s | results, slow push-in | See exactly what to reteach |
| 55–60 s | endcard | Try it free on Tes · No logins · Works offline (on the card) |

Music (`tools/video/music.py`, 120 bpm, beat = 0.5 s): snare roll and swell under the teeter, the stab on the
cascade at 2.5 s, the title hook at 3 s, a sting on each game's name (15, 22, 30, 37 s), a breakdown under the
launcher (44–50 s), the build (50–55 s) and the final fanfare on the end card, ending at 60 s.

## Other versions

- **Reel (18 s, 9:16):** the reel 1 cascade (3 s); every team marked in portrait (3.4 s, "Every student answers every
  question"); the jackpot falling (2.6 s, Over the Edge logo); the class reaching HOME (2.2 s, Outpace logo); the star
  tile (2 s, Category Clash); the winning path (2 s, Hex Hunt); "Free on Tes · No logins · Works offline" (2.8 s).
  Text is kept between y 250 and 1470 (the platforms' safe zone). Music: the theme's 18 s cut.
- **Clips (15 s each, 16:9):** each game's name, its best moment, and the wordmark with "Whole-class GCSE revision ·
  Free on Tes" over the last 2.4 s. Music: the theme's 15 s cut.

To change a cut, edit the segment lists in `tools/video/demo/edit.py` and run `python3 tools/video/demo/edit.py all`
(add `AUDIO_ONLY=1` when only the sound changed).
