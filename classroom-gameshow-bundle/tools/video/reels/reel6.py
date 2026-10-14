#!/usr/bin/env python3
"""
Reel 6, "Answer before it falls": lays the text from reel6.js over the caption-free jackpot footage (reel-6-footage),
on the beat of its fall. The hook is on the very first frame; the question pops in; a timer ring drains and its number
ticks down so it reaches zero as the jackpot counter falls; on the fall the answer is revealed (and a short flash
marks it); then "Be honest: did you get it?" and "Tag the friend who forgot". Everything fades out before the call
to action (cta.py 6). Adds the clock ticks and a chime to sounds.json.

  python3 tools/video/reels/reel6.py     (after render-ote.js shots/reel-6-footage.json and reel6.js)
"""
import json, math, os, shutil
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'marketing', 'reels', 'work'))
L = os.path.join(WORK, 'reel6')
FPS = 60

def back_out(t):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def lay(name):
    im = Image.open(os.path.join(L, name + '.png')).convert('RGBA')
    return im.crop(im.getbbox()), im.getbbox()

def put(canvas, layer, scale=1.0, alpha=1.0, dy=0):
    im, box = layer
    w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    r = im.resize((w, h), Image.LANCZOS)
    if alpha < 1: r.putalpha(r.getchannel('A').point(lambda v: int(v * alpha)))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + dy
    canvas.alpha_composite(r, (int(cx - w / 2), int(cy - h / 2)))

def pop(canvas, layer, f, start, out_from, dur=9, pulse=0.0):
    """pops in at frame `start` (back-out), fades from frame `out_from`"""
    t = (f - start) / float(dur)
    if t <= 0: return
    t = min(1.0, t)
    a = min(1.0, t * 2.5) * max(0.0, 1 - (f - out_from) / 8.0) if f >= out_from else min(1.0, t * 2.5)
    if a <= 0: return
    put(canvas, layer, (0.72 + 0.28 * back_out(t)) * (1 + pulse), a)

def ring(frac, hot):
    S = 4; R = 138; W = 20; size = (2 * R + 40)
    im = Image.new('RGBA', (size * S, size * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    box = [20 * S, 20 * S, (2 * R + 20) * S, (2 * R + 20) * S]
    d.ellipse(box, outline=(20, 24, 52, 235), width=W * S)
    if frac > 0:
        col = (255, 70, 60, 255) if hot else (255, 201, 60, 255)
        d.arc(box, -90, -90 + 360 * frac, fill=col, width=W * S)
    return im.resize((size, size), Image.LANCZOS)

def build():
    import sys
    src = os.path.join(WORK, sys.argv[1] if sys.argv[1:] else 'reel-6-footage')
    rec = json.load(open(os.path.join(src, 'sounds.json')))
    N, fall = rec['frames'], rec['cues']['fall']
    out = os.path.join(WORK, 'reel-6-answer')
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    A = {n: lay(n) for n in ['hook', 'question', 'comment', 'answer', 'tag', 'tag2'] + ['num%d' % i for i in range(1, 8)]}
    REVEAL, FADE = fall + 3, N - 12
    for f in range(N):
        bg = Image.open(os.path.join(src, '%05d.jpg' % f)).convert('RGBA')
        if f < fall + 6 or True:
            pop(bg, A['hook'], f, -1, FADE, dur=1)                        # on the first frame
        pop(bg, A['question'], f, 6, FADE)
        left = (fall - f) / FPS
        if 10 <= f < fall:
            n = max(1, min(7, math.ceil(left)))
            since = (fall - f) % FPS if n > 1 else 0
            age = (FPS - since) % FPS                                      # frames since the number changed
            bump = 0.28 * max(0.0, 1 - age / 8.0) if age < 8 and f > 14 else 0.0
            hot = left < 2.0
            bg.alpha_composite(ring(left / (fall / FPS), hot), (540 - 178, 1010 - 178))
            put(bg, A['num%d' % n], 1 + bump)
            pop(bg, A['comment'], f, 28, fall)
        if f >= REVEAL:
            pop(bg, A['answer'], f, REVEAL, FADE)
            pop(bg, A['tag'], f, REVEAL + 40, REVEAL + 130)
            pop(bg, A['tag2'], f, REVEAL + 132, FADE)
        if fall <= f < fall + 7:                                            # a white flash on the reveal
            fl = Image.new('RGBA', bg.size, (255, 255, 255, int(110 * (1 - (f - fall) / 7.0))))
            bg.alpha_composite(fl)
        bg.convert('RGB').save(os.path.join(out, '%05d.jpg' % f), quality=93)
    # the clock: a tick each second (alternating pitch), the last one lower and longer, then a chime on the answer
    t0 = rec['t0']; sounds = list(rec['sounds'])
    def osc(kind, at, dur, freq, gain):
        s = t0 + at
        sounds.append({'type': kind, 'start': s, 'stop': s + dur + 0.02, 'freq': [['set', s, freq]],
                       'gain': {'gain': [['set', s, 0.0001], ['exp', s + 0.004, gain], ['exp', s + dur, 0.0001]]}})
    for k in range(1, 8):
        at = (fall - k * FPS) / FPS
        if at < 0.15: continue
        last = k == 1
        osc('square', at, 0.05 if not last else 0.12, 1500 if k % 2 else 1100, 0.05 if not last else 0.08)
        osc('sine', at, 0.09, 220, 0.12)
    for i, fr in enumerate((784, 988, 1175)): osc('triangle', (REVEAL + i * 4) / FPS, 0.7, fr, 0.05)
    rec2 = dict(rec); rec2['sounds'] = sounds
    json.dump(rec2, open(os.path.join(out, 'sounds.json'), 'w'))
    print('reel-6-answer', N, 'frames; fall at', fall)

if __name__ == '__main__':
    build()
