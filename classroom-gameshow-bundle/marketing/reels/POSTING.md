# Posting the reels

Post each reel natively on Instagram Reels, TikTok and YouTube Shorts (upload the mp4 itself; don't share links between
apps). Use the version with music, or the `-sfx-only` version and add a trending sound inside the app, which often helps
reach. Every reel now ends with a 4-second call to action in the video itself ("Turn your classroom into a game show", "Play free now on Tes", "Link in bio", the host waving bottom right). Repeat it in the caption: "Free taster on Tes, link in bio" (or your Tes shop link
where the platform allows one). The end card sits over the platforms' bottom buttons by design (the host is there on purpose), so keep the caption short. Pin a comment answering the question you ask.

Covers are in this folder (`*-cover.jpg`): the text sits inside the centre square, so they still read on a profile grid.

## Suggested order (about two weeks)

| Day | Reel | Why then |
|---|---|---|
| 1 (Tue) | 1. Will it fall? | The strongest scroll-stopper; a looping question invites rewatches and comments |
| 3 (Thu) | 3. oddly satisfying | Broad, non-teacher appeal; best with sound off and on loop |
| 5 (Sun) | 5. The £5,000 jackpot | Weekend lesson planning; payoff reel |
| 8 (Wed) | 2. POV: game show | Shows all four games once people have seen one |
| 10 (Fri) | 4. The Hunter is coming | Tension and a win; good for end-of-week |
| 12–14 | Repost the best performer with the other sound version | Fresh audio, same proven video |

## Captions and hashtags

**1. Will it fall?**
Caption: Revision, but every question could tip it over the edge. Would your class keep watching?
Hashtags: #teachersoftiktok #teachertok #gcse #revision #classroomgames #ukteachers #satisfying #edutok

**2. POV: you turned GCSE revision into a game show**
Caption: Four game shows, one question bank, the whole class answering on whiteboards. Which one would your class pick?
Hashtags: #pov #teachersoftiktok #gcserevision #classroomideas #ukteachers #teachertok #gameshow #edutok

**3. oddly satisfying**
Caption: Seven counters, one push. Did you watch it more than once?
Hashtags: #oddlysatisfying #satisfying #asmr #teachersoftiktok #classroomgames #gcse #slowmotion

**4. The Hunter is coming**
Caption: Two wrong answers and the Hunter was one step behind. Would your class have made it home?
Hashtags: #teachersoftiktok #teachertok #gcse #revisiongame #classroomgames #ukteachers #edutok #quiz

**5. The £5,000 jackpot**
Caption: Last question, jackpot on the edge, whole class shouting. Would your class have got the jackpot?
Hashtags: #teachersoftiktok #gcse #revision #classroomgames #ukteachers #teachertok #jackpot #edutok

The games are original; no TV show is named or imitated. No pupils, faces or schools appear in any reel.

## Reel 6: "Answer before it falls" (the one built to travel)

Nobody can promise a video goes viral; this one is built on what reliably helps a short video spread, and it is
worth posting first and at the best time you have (a weekday around 7-8 pm UK, or Sunday evening: teachers are on
their phones then).

What each part is doing:

- **Text on frame 1, a direct call-out** ("Year 11s: answer before it falls"). Viewers decide in about a second; a
  named audience plus a task is a reason to stop scrolling.
- **A question with a deadline.** The countdown ring, the ticking and the gold £5,000 counter teetering on the edge
  make a loop you want closed (curiosity gap plus loss aversion): most people stay to see whether it falls.
- **Everyone can answer it.** "The mitochondria is the ___ of the cell" is a shared school memory, so people
  answer in the comments (comments and shares weigh most), tag friends and feel clever (social currency).
- **A payoff and a twist.** The reveal lands with the fall, confetti and a chime, and adds the real science
  ("site of aerobic respiration") so teachers see it is proper revision, not just a meme.
- **Two prompts to act.** "Be honest: did you get it?" (comments) then "Tag the friend who forgot" (shares), then the
  call to action to play free on Tes.
- **Short and it replays.** 14 s, and it cuts straight back to a teetering counter and a question, so many viewers
  watch it twice (watch time and replays are what the platforms reward most).
- Sound works with or without audio: every word is on screen.

Caption (Instagram, TikTok and Shorts): `Year 11s: can you get this before it falls? Teachers: save this for Monday's
starter. Free taster on Tes, link in bio.`

Pin this comment yourself right away: `Answer in 3, 2, 1... PS: no Googling.` Then reply to the first replies in the
first hour (early replies push the video wider). Hashtags (keep 3 to 5): `#gcse #revision #teachersoftiktok
#scienceteacher #classroomgames`.

Make a follow-up: post the same format with a different question every few days (Maths, History, Geography), since
a repeatable series builds followers; change the question text in `tools/video/reels/reel6.js` and re-run `reel6.py`,
`cta.py 6` and the `mix.py` lines for reel 6 in `build.sh` (the footage does not need re-rendering).
