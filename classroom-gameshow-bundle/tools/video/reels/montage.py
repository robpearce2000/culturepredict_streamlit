#!/usr/bin/env python3
"""
Reel 2, "POV: you turned GCSE revision into a game show": fast cuts on the beat of the Showtime theme (112 bpm,
so 2 beats = 1.071 s), taken from the other reels' frames (the parts with no caption) and the two board shots.
Builds work/reel-2-pov-game-show/ (frames linked in order, the game sounds of each cut moved to its place) for mix.py.
  python3 tools/video/reels/montage.py
"""
import json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'marketing', 'reels', 'work'))
FPS = 60
BEAT = 60 / 112
def beats(n): return round(n * BEAT * FPS)

# (source shot, cue in its log ('' = frame 0), frames after the cue, length in beats): every cut lands on a beat
CUTS = [
    ('reel-1-will-it-fall', 'fall', -12, 4),   # the counter tips and the cascade goes
    ('reel-5-jackpot', 'fall', -60, 4),        # the jackpot falls
    ('reel-4-hunter', 'escape', 10, 4),        # the class bursts through the HOME arch
    ('board-cc-star', '', 24, 4),              # the star tile spins: double points
    ('board-hh-path', '', 24, 4),              # the winning path lights up
    ('reel-5-jackpot', 'fall', 68, 4),         # +£5,000 and confetti, to the end
]

def main():
    out = os.path.join(WORK, 'reel-2-pov-game-show')
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    sounds, n = [], 0
    for name, cue, off, nb in CUTS:
        d = os.path.join(WORK, name); rec = json.load(open(os.path.join(d, 'sounds.json')))
        start = max(0, (rec.get('cues', {}).get(cue, 0) if cue else 0) + off)
        length = beats(nb)
        for i in range(length):
            src = os.path.join(d, '%05d.jpg' % min(start + i, rec['frames'] - 1))
            os.symlink(src, os.path.join(out, '%05d.jpg' % (n + i)))
        a, b = rec['t0'] + start / FPS, rec['t0'] + (start + length) / FPS
        for s in rec['sounds']:
            if s.get('start') and a <= s['start'] < b:
                t = dict(s); off = n / FPS - start / FPS - rec['t0']
                t['start'] = s['start'] + off; t['stop'] = s['stop'] + off
                t['freq'] = [[k, tt + off, v] for k, tt, v in s.get('freq', [])]
                if s.get('gain'): t['gain'] = {'gain': [[k, tt + off, v] for k, tt, v in s['gain'].get('gain', [])]}
                sounds.append(t)
        n += length
    json.dump({'t0': 0, 'frames': n, 'fps': FPS, 'sounds': sounds}, open(os.path.join(out, 'sounds.json'), 'w'))
    print('reel-2 frames', n, f'{n / FPS:.2f}s')

if __name__ == '__main__':
    main()
