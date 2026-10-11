#!/bin/sh
# Safe-zone check: a contact sheet of every reel (one frame each 0.75 s) with the areas the apps cover drawn in red:
# the top 250 px (account name) and the bottom 450 px (caption and buttons). Captions and the action must stay clear.
#   sh tools/video/reels/check.sh    ->  marketing/reels/check/<reel>.jpg
cd "$(dirname "$0")/../../.."
mkdir -p marketing/reels/check
for f in marketing/reels/reel-?-*[!y].mp4 marketing/reels/reel-?-*.mp4; do
  case "$f" in *-sfx-only.mp4) continue;; esac
  n=$(basename "$f" .mp4)
  ffmpeg -v error -y -i "$f" -vf "fps=4/3,drawbox=x=0:y=0:w=iw:h=250:color=red@0.35:t=fill,drawbox=x=0:y=ih-450:w=iw:h=450:color=red@0.35:t=fill,scale=270:480,tile=8x2" -frames:v 1 "marketing/reels/check/$n.jpg"
done
ls marketing/reels/check
