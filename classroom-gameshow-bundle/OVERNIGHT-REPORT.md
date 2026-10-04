# Overnight report

(Summary to be completed at the end of the night.)

## Part 1: Outpace — done

- Ported the prototype's fixes from `outpace-reference.html` into `src/games/outpace/` rather than copying the built file, onto the current game (one mode, 10-minute length, main-screen subject).
- **Performance:** resolution capped at 1.25×, no shadows, name tags measured only when they change and moved by transform (no page layout every frame), HUD-panel resize observer, and an automatic quality step-down (bloom off, then 1:1 resolution) after two seconds of slow frames. Measured on the test machine's software renderer at 1366×768: High 6.0 → 8.8 fps, Low 8.2 → 9.1 fps. A real laptop GPU will be far faster; please check on your laptop.
- **Final Sprint:** the question is a side column on screens 900px and wider; the race, camera and tags are centred in the space left. Portrait tablets keep the bottom card.
- **Instant win:** reaching the target stops the clock; after a 1.6 s undo window the escape plays.
- **Subject looks:** Science, Maths, English, History, Geography and General, following the main-screen subject; runner gold, Hunter magenta in every look.
- **Smaller Deal Round tags.**
- Tests: `tests/outpace.spec.js` (instant win with undo, every subject's look, frame rate at Low and High, recorded). Outpace screenshot sweep at all three sizes and the full-graphics Outpace playthrough pass.
