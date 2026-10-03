# Changelog

## 1.0.0 – first edition (October 2026)

Two stand-alone tutoring games packaged as one product.

### New
- **Launcher**: Showtime wordmark, the host as mascot, a card for each game, space for future games, global settings, the question bank and an About panel with credits and licences.
- **Shared question bank** used by both games: list, choose, add by pasting text, rename, edit, delete (with a confirm tap), copy a built-in pack, and save or load a JSON backup of all sets and history.
- **GCSE Combined Science starter pack (AQA-style)**: 191 questions across all seven Biology, ten Chemistry and seven Physics topic areas, in separate subject packs and a full mix. Builds on the v1 mixed set; every question checked for GCSE accuracy.
- **Shared wrong-answer history** per player name across both games, with the "pick more questions from weak topics" option in both.
- **Global settings**: sound, text size (normal or large), reduced motion, graphics quality (high or low).
- **Single offline file** with fonts, Three.js and every script inlined, and a security policy that blocks network access.
- **Teacher guide** (PDF) and **Tes listing** images and text.
- Automated browser tests, contrast check and banned-term check.

### Over the Edge (the counter-pusher game)
- New name, logo, header sign and set: teal studio, coral and aqua lighting, sunflower shelves with a hazard-stripe edge, slate peg board, brass pegs.
- Drop zones are now **lanes 1–4**; mystery counters are **★ wildcard counters**.
- Host renamed Professor Pip with a new default look and a bow tie; still fully customisable, and shared with the launcher.
- Menu, sound and settings buttons on screen; Esc returns to the menu (with a confirm during a game). Enter starts and replays.
- Darker button colours for WCAG AA contrast; players also marked ● and ■.
- Speech bubble kept clear of the menu bar, tail follows the host.
- Physics, scoring and difficulty unchanged.

### Outpace (the race game; working title replaced)
- New name and logo. The chaser is now **the Hunter**; the offer round is the **Deal Round** with Cautious, Standard and Bold deals; the final is the **Final Sprint**; "help" is now **pass**.
- Runner and Hunter recoloured gold and magenta with name tags; track and home cell restyled.
- Full keyboard control (there were no shortcuts before).
- New bottom dock layout: the pass buttons can no longer cover the question card.
- Camera keeps both atoms on screen on portrait tablets.
- Wrong answers saved as they happen; short synthesised sound effects added; softer catch flash, off with reduced motion.
- Uses the Showtime UI kit and fonts (no web font downloads).
- Rules and timings unchanged.

### Removed
- All references to the TV programmes the games were inspired by, in text, titles, comments and visuals.
- Web font and CDN requests.
- The one-off host-style reset from v1 of the counter-pusher game.
