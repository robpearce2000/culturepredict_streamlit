#!/usr/bin/env python3
"""
Adds the call to action to the end of every reel: after the last picture the footage holds, dims and softens, and the
layers from cta.js (wordmark, a headline for that reel, a line under it, the "Play free now on Tes" button, "Link in bio") pop in one
after another, the button pulsing, and the host slides up from the bottom right corner, waving. Builds work/<reel>-cta/ (the reel's own frames, then the CTA frames) and its
sounds.json (frames extended; every game sound where it was), for mix.py.

  python3 tools/video/reels/cta.py [reel ...]       (after cta.js and montage.py; reel 6 after reel6.py)

The card stays 4 s: long enough to read the headline (6 words), the button (5) and the line under it.
"""
import json, math, os, shutil
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'marketing', 'reels', 'work'))
CTA = os.path.join(WORK, 'cta')
FPS = 60
TAIL = 4 * FPS
REELS = {1: 'reel-1-will-it-fall', 2: 'reel-2-pov-game-show', 3: 'reel-3-oddly-satisfying', 4: 'reel-4-hunter', 5: 'reel-5-jackpot', 6: 'reel-6-answer'}
# when each layer arrives (frame in the tail) and how it moves in
ARRIVE = {'wm': 4, 'head': 10, 'sub': 22, 'btn': 30, 'bio': 44}
PIP_AT = 52   # the host slides up from the bottom right corner and waves

def ease_out(t): return 1 - (1 - t) ** 3
def back_out(t):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def pip_frames():
    """the host's wave as a loop: the run of captured frames up to the one that looks most like the first"""
    meta = json.load(open(os.path.join(CTA, 'pip.json')))
    fr = [Image.open(os.path.join(CTA, 'pip-%02d.png' % i)).convert('RGBA') for i in range(meta['frames'])]
    def diff(a, b):
        pa, pb = a.resize((90, 112)).tobytes(), b.resize((90, 112)).tobytes()
        return sum(abs(x - y) for x, y in zip(pa, pb)) / len(pa)
    best = min(range(12, len(fr)), key=lambda k: diff(fr[0], fr[k]))
    return fr[:best], meta['x'], meta['y']

def layer(n, name):
    im = Image.open(os.path.join(CTA, f'{n}-{name}.png')).convert('RGBA')
    return im.crop(im.getbbox()), im.getbbox()

def paste(canvas, lay, f, name):
    im, box = lay
    t = (f - ARRIVE[name]) / 9.0
    if t <= 0: return
    t = min(1.0, t)
    s = 0.7 + 0.3 * (back_out(t) if name in ('btn', 'head') else ease_out(t))
    if name == 'btn' and f > 36: s *= 1 + 0.028 * math.sin((f - 36) / 42 * 2 * math.pi)   # the button breathes
    w, h = max(1, int(im.width * s)), max(1, int(im.height * s))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    r = im.resize((w, h), Image.LANCZOS)
    a = r.getchannel('A').point(lambda v: int(v * min(1.0, t * 2.2)))
    r.putalpha(a)
    canvas.alpha_composite(r, (int(cx - w / 2), int(cy - h / 2)))

def build(n, name):
    src = os.path.join(WORK, name)
    rec = json.load(open(os.path.join(src, 'sounds.json')))
    out = os.path.join(WORK, name + '-cta')
    shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    N = rec['frames']
    for i in range(N):
        os.symlink(os.path.realpath(os.path.join(src, '%05d.jpg' % i)), os.path.join(out, '%05d.jpg' % i))
    last = Image.open(os.path.join(src, '%05d.jpg' % (N - 1))).convert('RGB')
    dim = ImageEnhance.Brightness(last.filter(ImageFilter.GaussianBlur(9))).enhance(0.38)
    layers = {k: layer(n, k) for k in ARRIVE}
    pips, px, py = pip_frames()
    for f in range(TAIL):
        k = min(1.0, f / 12.0)                                            # the picture dims and softens as the card arrives
        bg = Image.blend(last, dim, ease_out(k)).convert('RGBA')
        for nm in ('wm', 'head', 'sub', 'btn', 'bio'): paste(bg, layers[nm], f, nm)
        if f >= PIP_AT:      # the host peeks in from the bottom right corner, waving
            t = min(1.0, (f - PIP_AT) / 16.0)
            bg.paste(pips[(f - PIP_AT) % len(pips)], (px, py + int((1 - back_out(t)) * 560)), pips[(f - PIP_AT) % len(pips)])
        bg.convert('RGB').save(os.path.join(out, '%05d.jpg' % (N + f)), quality=93)
    rec2 = dict(rec); rec2['frames'] = N + TAIL; rec2['cta'] = {'from': N, 'frames': TAIL}
    json.dump(rec2, open(os.path.join(out, 'sounds.json'), 'w'))
    print(name + '-cta', N + TAIL, 'frames', f'{(N + TAIL) / FPS:.2f}s')

if __name__ == '__main__':
    import sys
    for n, name in REELS.items():
        if not sys.argv[1:] or str(n) in sys.argv[1:]: build(n, name)
