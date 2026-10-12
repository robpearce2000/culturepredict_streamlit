#!/bin/sh
# Rebuilds every demo video from scratch (a few hours on a 4-core machine with software 3D):
#   sh tools/video/demo/build.sh
# Needs: node + Playwright's Chromium, python3 + numpy + Pillow + mido, ffmpeg, fluidsynth with the FluidR3_GM
# soundfont, and the test builds (node build.js --editions). The vertical reel also uses the reels' takes
# (sh tools/video/reels/build.sh renders them).
set -e
cd "$(dirname "$0")/../../.."
D=tools/video/demo
node build.js --editions
python3 tools/video/music.py                                   # the theme: 60 s, and 15 s / 18 s cuts
# the shots (each replays its take exactly: fixed seed, same actions, controlled clock)
node tools/video/reels/render-ote.js $D/shots/hook.json &       # hook-b: from just before the fall
node tools/video/reels/render-ote.js $D/shots/hook-a.json &     # hook-a: the teeter before it
node tools/video/reels/render-ote.js $D/shots/jackpot.json &
node $D/shot-ote.js 4 &
wait
node $D/shot-outpace.js &
node $D/shot-boards.js cc 5 &
node $D/shot-boards.js hh 5 &
node $D/shot-launcher.js &
wait
node $D/shot-results.js 2
node $D/shot-vertical.js
node $D/cards.js
node $D/endcard.js                                             # the end card and thumbnail.png (stills from the shots)
python3 $D/edit.py all                                         # cut, captions, sounds and music, encode
