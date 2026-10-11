#!/bin/sh
# Rebuilds every reel from scratch (about 2 hours on a 4-core machine with software 3D):
#   sh tools/video/reels/build.sh            render, mix and encode all five reels, both sound versions, and the covers
#   sh tools/video/reels/check.sh            contact sheets of every reel with the platform safe zones drawn in
# Needs: node + Playwright's Chromium, python3 + numpy, ffmpeg, fluidsynth and the FluidR3_GM soundfont, and the
# test build (node build.js).
set -e
cd "$(dirname "$0")/../../.."
R=tools/video/reels
OUT=marketing/reels
node build.js
# the shots (each replays its take exactly: fixed seed, same actions, fake clock)
node $R/render-ote.js $R/shots/reel-1-will-it-fall.json &
node $R/render-ote.js $R/shots/reel-3-oddly-satisfying.json &
wait
node $R/render-ote.js $R/shots/reel-5-jackpot.json &
node $R/render-outpace.js 3 &
wait
node $R/render-boards.js cc-star 5
node $R/render-boards.js hh-path 5
python3 $R/montage.py
node $R/stills.js
# mix and encode: with the Showtime theme, and game sounds only
python3 $R/mix.py reel-1-will-it-fall $OUT/reel-1-will-it-fall.mp4 --music
python3 $R/mix.py reel-1-will-it-fall $OUT/reel-1-will-it-fall-sfx-only.mp4
python3 $R/mix.py reel-2-pov-game-show $OUT/reel-2-pov-game-show.mp4 --music --overlay $OUT/work/caption-reel-2.png
python3 $R/mix.py reel-2-pov-game-show $OUT/reel-2-pov-game-show-sfx-only.mp4 --overlay $OUT/work/caption-reel-2.png
python3 $R/mix.py reel-3-oddly-satisfying $OUT/reel-3-oddly-satisfying.mp4 --sfx-gain 2.2
python3 $R/mix.py reel-3-oddly-satisfying $OUT/reel-3-oddly-satisfying-sfx-only.mp4 --sfx-gain 2.2
python3 $R/mix.py reel-4-hunter $OUT/reel-4-hunter.mp4 --music
python3 $R/mix.py reel-4-hunter $OUT/reel-4-hunter-sfx-only.mp4
python3 $R/mix.py reel-5-jackpot $OUT/reel-5-jackpot.mp4 --music
python3 $R/mix.py reel-5-jackpot $OUT/reel-5-jackpot-sfx-only.mp4
