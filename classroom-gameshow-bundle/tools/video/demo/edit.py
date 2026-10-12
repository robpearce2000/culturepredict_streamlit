#!/usr/bin/env python3
"""
The demo videos' edit (recording only): cuts the rendered shots together, lays the captions over them with a
quick pop-in, rebuilds the games' own sounds in sync (from what each shot recorded), mixes in the Showtime
theme and encodes the mp4s.

  python3 tools/video/demo/edit.py main      showtime-demo-60s-1080p.mp4 (+ -silent), captions.srt
  python3 tools/video/demo/edit.py reel      showtime-reel-9x16.mp4
  python3 tools/video/demo/edit.py clips     clip-over-the-edge.mp4, clip-outpace.mp4, clip-category-clash.mp4, clip-hex-hunt.mp4
  python3 tools/video/demo/edit.py all

Every edit is a list of segments: frames taken from a shot (from a named cue in its log, plus an offset), with an
optional layout ('inset': the footage framed on a branded background, the caption above it) and a slow zoom; and a
list of captions (card name, from, to; in seconds of the finished video). Shots live in marketing/video/work/<name>/
(16:9) or marketing/reels/work/<name>/ (9:16, the reels' takes), each with sounds.json (frames, cues, sounds).
"""
import json, math, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
WORK = os.path.join(ROOT, 'marketing', 'video', 'work')
RWORK = os.path.join(ROOT, 'marketing', 'reels', 'work')
OUT = os.path.join(ROOT, 'marketing', 'video')
CARDS = os.path.join(WORK, 'cards')
FPS = 60
sys.path.insert(0, os.path.join(HERE, '..', 'reels'))
from mix import render_sfx, SR  # noqa: E402  (the reels' sound rebuilder: the games' oscillators, offline)

def S(t): return int(round(t * FPS))

# ---------------------------------------------------------------- the edits
# a segment: (shot, start, frames, options); start is a frame number or (cue, offset in frames)
MAIN = {
    'size': (1920, 1080), 'music': os.path.join(OUT, 'showtime-theme.wav'), 'length': 60,
    'segments': [
        ('hook-a', 0, 149, {}),                                    # 0-2.5 s: the teeter, the push-in
        ('hook-b', 0, S(6.4) - 149, {}),                           # 2.5 s: it tips, the cascade, +£500; the title card
        ('ote', ('showme', -84), S(11.0) - S(6.4), {}),            # the question, "3, 2, 1, show me!"
        ('ote', ('mark', -12), S(4.0), {}),                        # six teams marked in seconds
        ('ote', ('lanes', 50), S(4.0), {'punch': True}),           # Over the Edge: the drops push counters off
        ('jackpot', ('fall', -110), S(3.0), {}),                   # the jackpot teeters and falls
        ('outpace', ('move0', 0), 84, {'punch': True}),            # Outpace: the class surges a step
        ('outpace', ('move1', 0), 96, {}),                         # a wrong answer: the Hunter lunges
        ('outpace-sprint', ('clock', 0), 96, {}),                  # the Final Sprint clock
        ('outpace-sprint', ('step', 0), 60, {}),                   # a sprint step
        ('outpace', ('escape', 0), S(30.0) - S(22.0) - 336, {}),   # HOME: the burst through the arch
        ('cc', ('pick', -36), S(3.6), {'layout': 'inset', 'punch': True}),     # Category Clash: the star tile
        ('cc', ('mark', -6), S(37.0) - S(30.0) - S(3.6), {'layout': 'inset'}),  # marked, the scores jump
        ('hh', ('pick', 0), S(2.6), {'layout': 'inset', 'punch': True, 'zoom': (1.15, 1.15, 960, 520)}),       # Hex Hunt: a hexagon changes colour
        ('hh', ('win', -66), S(44.0) - S(37.0) - S(2.6), {'layout': 'inset', 'zoom': (1.15, 1.15, 960, 520)}),  # the winning path lights up
        ('launcher', ('subject', -20), S(6.0), {}),                # the launcher: subject, board, course, topics, play
        ('results', ('results', 30), S(5.0), {'zoom': (1.0, 1.08, 960, 650)}),  # the results: reteach these
        ('endcard', 0, S(5.0), {}),                                # the end card
    ],
    'captions': [
        ('title', 3.0, 6.4), ('answers', 6.5, 10.95), ('mark', 11.05, 14.95), ('ote', 15.05, 18.95),
        ('outpace', 22.1, 29.9), ('cc', 30.05, 36.95), ('hh', 37.05, 43.95), ('launcher', 44.2, 49.9), ('results', 50.3, 54.95),
    ],
    # moments where the game's own sounds must come through: the music dips under them
    'duck': [(2.3, 3.6), (20.0, 21.6)],
    'srt': [
        (3.0, 6.4, 'Showtime: Classroom Gameshows\nWhole-class GCSE revision'),
        (6.5, 10.95, 'Every student answers every question'),
        (11.05, 14.95, 'Mark the whole class in seconds'),
        (15.05, 18.95, 'Over the Edge'),
        (22.1, 29.9, 'Outpace\nRace home before the Hunter catches you'),
        (30.05, 36.95, 'Category Clash'),
        (37.05, 43.95, 'Hex Hunt'),
        (44.2, 49.9, 'AQA & Edexcel · Separate & Combined\nBiology · Chemistry · Physics'),
        (50.3, 54.95, 'See exactly what to reteach'),
        (55.0, 60.0, 'Showtime: Classroom Gameshows\nTry it free on Tes · No logins · Works offline'),
    ],
}

# the vertical reel (1080×1920): the cascade, "every student answers", one shot from each game, "Free on Tes"
REEL = {
    'size': (1080, 1920), 'music': os.path.join(WORK, 'music', 'theme-18s.wav'), 'length': 18,
    'segments': [
        ('R:reel-1-will-it-fall', ('fall', -90), S(3.0), {}),                    # the teeter, the cascade
        ('v-ote', ('mark', 0), S(3.4), {}),                                      # every team marked
        ('R:reel-5-jackpot', ('fall', -40), S(2.6), {'punch': True}),             # Over the Edge: the jackpot
        ('R:reel-4-hunter', ('escape', -40), S(2.2), {'punch': True}),            # Outpace: HOME
        ('R:board-cc-star', 20, S(2.0), {'punch': True}),                         # Category Clash: the star tile
        ('R:board-hh-path', 20, S(2.0), {'punch': True}),                         # Hex Hunt: the winning path
        ('R:reel-5-jackpot', ('fall', 90), S(18.0) - S(15.2), {}),                # Free on Tes, over the confetti
    ],
    'captions': [
        ('v-answers', 3.05, 6.35), ('v-ote', 6.45, 8.95), ('v-outpace', 9.05, 11.15), ('v-cc', 11.25, 13.15), ('v-hh', 13.25, 15.15), ('v-free', 15.2, 18.0),
    ],
    'duck': [(1.2, 2.4)],
}

def clip(name, segs, caps):
    return {'size': (1920, 1080), 'music': os.path.join(WORK, 'music', 'theme-15s.wav'), 'length': 15,
            'segments': segs, 'captions': caps + [('clip-end', 12.6, 15.0)], 'duck': [], 'name': name}

CLIPS = {
    'clip-over-the-edge': clip('Over the Edge', [
        ('ote', ('showme', -30), S(2.6), {}),
        ('ote', ('mark', 0), S(2.4), {}),
        ('ote', ('lanes', 50), S(3.4), {}),
        ('jackpot', ('fall', -200), S(15.0) - S(8.4), {}),
    ], [('ote', 0.1, 3.0)]),
    'clip-outpace': clip('Outpace', [
        ('outpace', ('move0', 0), ('escape', 0), {}),             # every move of the deal round, in order
        ('outpace-sprint', ('clock', 0), 216, {}), ('outpace-sprint', ('step', 0), 60, {}),
        ('outpace', ('escape', 0), ('end', 0), {}),
    ], [('outpace', 0.1, 4.5)]),
    'clip-category-clash': clip('Category Clash', [
        ('cc', ('board', 0), ('end', 0), {'layout': 'inset'}),
    ], [('cc', 0.1, 12.4)]),
    'clip-hex-hunt': clip('Hex Hunt', [
        ('hh', ('pick', 0), ('lastpick', 0), {'layout': 'inset', 'zoom': (1.15, 1.15, 960, 520)}),
        ('hh', ('lastpick', 0), ('end', 0), {'layout': 'inset', 'zoom': (1.15, 1.15, 960, 520)}),
    ], [('hh', 0.1, 12.4)]),
}

# ---------------------------------------------------------------- shots
_shots = {}
def shot(name):
    if name not in _shots:
        d = os.path.join(RWORK, name[2:]) if name.startswith('R:') else os.path.join(WORK, name)
        rec = json.load(open(os.path.join(d, 'sounds.json')))
        fps = rec.get('fps', FPS)
        times = rec.get('times') or [rec['t0'] + (i + 1) / fps for i in range(rec['frames'])]
        _shots[name] = {'dir': d, 'rec': rec, 'times': times, 'frames': rec['frames'], 'cues': rec.get('cues', {})}
    return _shots[name]

def at(sh, ref):
    if isinstance(ref, int): return ref
    cue, off = ref
    if cue == 'end': return sh['frames'] + off
    return sh['cues'][cue] + off

def plan(edit):
    """the finished video's frames: (shot, source frame, segment options, frame within the segment, segment length)"""
    out = []
    for name, start, n, opt in edit['segments']:
        sh = shot(name); a = at(sh, start)
        n = at(sh, n) - a if isinstance(n, tuple) else n
        for i in range(n):
            out.append((name, min(max(0, a + i), sh['frames'] - 1), opt, i, n))
    # a clip shorter than its music holds its last picture (under the end card) to the end
    want = int(round(edit['length'] * FPS))
    if out and len(out) < want:
        name, f, opt, i, n = out[-1]; extra = want - len(out)
        out = [(a, b, c, d, e + extra) if a == name and e == n and len(out) - n <= k else (a, b, c, d, e) for k, (a, b, c, d, e) in enumerate(out)]
        out += [(name, f, opt, n + j, n + extra) for j in range(extra)]
    return out

# ---------------------------------------------------------------- pictures
_cards = {}
def card(name):
    if name not in _cards:
        im = Image.open(os.path.join(CARDS, name + '.png')).convert('RGBA')
        _cards[name] = (im, im.getbbox())
    return _cards[name]

_bg = {}
def inset_bg(size):
    if size not in _bg:
        w, h = size
        y = np.linspace(0, 1, h)[:, None]; x = np.linspace(-1, 1, w)[None, :]
        top, bot = np.array([0x2E, 0x1E, 0x5E]), np.array([0x12, 0x14, 0x38])
        g = top[None, None, :] * (1 - y[..., None]) + bot[None, None, :] * y[..., None]
        ang = np.arctan2(y - 1.1, x * 1.6)[..., None]                     # soft rays from below, like the launcher
        g = g * (0.94 + 0.06 * (np.sin(ang * 22) > 0))
        _bg[size] = Image.fromarray(np.clip(g, 0, 255).astype('uint8'), 'RGB')
    return _bg[size]

def ease_out(t): return 1 - (1 - t) ** 3

def compose(job):
    """one finished frame: the shot's frame (zoomed / inset), then the captions on screen at this moment"""
    k, (name, f, opt, i, n), caps, size, path = job
    w, h = size
    im = Image.open(os.path.join(shot(name)['dir'], '%05d.jpg' % f)).convert('RGB')
    if im.size != size: im = im.resize(size, Image.LANCZOS)
    z = 1.0
    if opt.get('zoom'):
        z0, z1, cx, cy = opt['zoom']; z = z0 + (z1 - z0) * (i / max(1, n - 1))
    if opt.get('punch') and i < 10:                                 # a quick push on the cut into each game
        z *= 1 + 0.06 * (1 - ease_out(i / 10))
        cx, cy = (opt['zoom'][2], opt['zoom'][3]) if opt.get('zoom') else (w / 2, h / 2)
    if z != 1.0:
        cw, ch = w / z, h / z
        x0 = min(max(0, cx - cw / 2), w - cw); y0 = min(max(0, cy - ch / 2), h - ch)
        im = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize(size, Image.LANCZOS)
    if opt.get('layout') == 'inset':                                 # framed on the brand background, room above for the logo
        s = 0.84; fw, fh = int(w * s), int(h * s); fx, fy = (w - fw) // 2, h - fh - 18
        bg = inset_bg(size).copy()
        sh_ = Image.new('L', size, 0); ImageDraw.Draw(sh_).rounded_rectangle((fx - 4, fy + 6, fx + fw + 4, fy + fh + 14), 26, fill=150)
        bg.paste((8, 8, 24), (0, 0), sh_.filter(ImageFilter.GaussianBlur(10)))
        m = Image.new('L', (fw + 14, fh + 14), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, fw + 13, fh + 13), 26, fill=255)
        bg.paste((0xFF, 0xC9, 0x3C), (fx - 7, fy - 7), m)
        m2 = Image.new('L', (fw, fh), 0); ImageDraw.Draw(m2).rounded_rectangle((0, 0, fw - 1, fh - 1), 20, fill=255)
        bg.paste(im.resize((fw, fh), Image.LANCZOS), (fx, fy), m2)
        im = bg
    t = k / FPS
    out = im.convert('RGBA')
    for cname, t0, t1 in caps:
        if not (t0 <= t < t1): continue
        c, box = card(cname)
        a_in = min(1.0, (t - t0) * FPS / 8); a_out = min(1.0, (t1 - t) * FPS / 6)
        a = min(a_in, a_out); sc = 0.86 + 0.14 * ease_out(a_in)
        layer = c
        if sc < 0.999 and box:
            bx = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            crop = c.crop(box); cw, ch = max(1, int(crop.width * sc)), max(1, int(crop.height * sc))
            layer = Image.new('RGBA', c.size, (0, 0, 0, 0)); layer.paste(crop.resize((cw, ch), Image.LANCZOS), (int(bx[0] - cw / 2), int(bx[1] - ch / 2)))
        if a < 0.999:
            r, g, b, al = layer.split(); layer = Image.merge('RGBA', (r, g, b, al.point(lambda v: int(v * a))))
        out.alpha_composite(layer)
    out.convert('RGB').save(path, quality=93)
    return k

# ---------------------------------------------------------------- sounds
def sounds(frames):
    """the game sounds of each segment, moved to where its frames land in the finished video"""
    out = []
    seg_start = 0
    while seg_start < len(frames):
        name, f0, opt, i, n = frames[seg_start]
        sh = shot(name); rec = sh['rec']; times = sh['times']
        src = [frames[seg_start + j][1] for j in range(n)]
        lo, hi = times[src[0]] - 1 / FPS, times[src[-1]]
        for s in rec['sounds']:
            st = s.get('start') or 0
            if not (lo <= st < hi): continue
            # the first frame that shows the moment the sound starts, and where that frame is in the video
            j = next((j for j, fr in enumerate(src) if times[fr] >= st), n - 1)
            dt = (seg_start + j) / FPS - (times[src[j]] - st) - st
            t = dict(s); t['start'] = s['start'] + dt; t['stop'] = (s.get('stop') or 0) + dt
            t['freq'] = [[kk, tt + dt, v] for kk, tt, v in s.get('freq', [])]
            if s.get('gain'): t['gain'] = {'gain': [[kk, tt + dt, v] for kk, tt, v in s['gain'].get('gain', [])]}
            out.append(t)
        seg_start += n
    return out

def audio(edit, frames, path, music=True):
    length = len(frames) / FPS
    sfx = render_sfx({'t0': 0, 'sounds': sounds(frames)}, length)[:int(length * SR)] * 1.6
    mix = sfx.copy()
    if music and edit.get('music'):
        data = subprocess.run(['ffmpeg', '-v', 'error', '-i', edit['music'], '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True).stdout
        m = np.frombuffer(data, dtype=np.float32).astype(np.float64)
        m = np.pad(m, (0, max(0, len(mix) - len(m))))[:len(mix)]
        env = np.ones(len(mix))
        for a, b in edit.get('duck', []):                           # the music dips under the key sound effects
            ia, ib = int(a * SR), int(b * SR); r = int(0.15 * SR)
            env[ia:ib] = np.minimum(env[ia:ib], 0.55)
            env[max(0, ia - r):ia] = np.minimum(env[max(0, ia - r):ia], np.linspace(1, 0.55, ia - max(0, ia - r)))
            env[ib:ib + r] = np.minimum(env[ib:ib + r], np.linspace(0.55, 1, len(env[ib:ib + r])))
        mix = mix + 0.6 * m * env
    peak = max(1e-6, np.abs(mix).max()); mix = mix * min(1.0, 0.89 / peak)
    import wave
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix, -1, 1) * 32767).astype('<i2').tobytes())

# ---------------------------------------------------------------- build
def build(key, edit, outs):
    frames = plan(edit)
    want = int(round(edit['length'] * FPS))
    if len(frames) != want: print(f'{key}: {len(frames)} frames, expected {want}')
    d = os.path.join(WORK, 'edit-' + key); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith('.jpg'): os.remove(os.path.join(d, f))
    jobs = [(k, fr, edit['captions'], edit['size'], os.path.join(d, '%05d.jpg' % k)) for k, fr in enumerate(frames)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get('JOBS', 3))) as ex:
        for _ in ex.map(compose, jobs, chunksize=24): pass
    for out, music, silent in outs:
        wav = os.path.join(d, 'mix.wav')
        if not silent: audio(edit, frames, wav, music)
        cmd = ['ffmpeg', '-y', '-v', 'error', '-framerate', str(FPS), '-i', os.path.join(d, '%05d.jpg')]
        cmd += [] if silent else ['-i', wav]
        cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', os.environ.get('CRF', '20'), '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-r', str(FPS), '-movflags', '+faststart']
        cmd += ['-an'] if silent else ['-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-shortest']
        subprocess.run(cmd + [os.path.join(OUT, out)], check=True)
        print('wrote', out, f'{len(frames) / FPS:.2f}s', f'{os.path.getsize(os.path.join(OUT, out)) / 1e6:.1f} MB')

def srt(edit, path):
    def ts(t): return '%02d:%02d:%02d,%03d' % (t // 3600, t % 3600 // 60, int(t % 60), round((t - int(t)) * 1000))
    with open(path, 'w') as f:
        for i, (a, b, text) in enumerate(edit['srt'], 1): f.write(f'{i}\n{ts(a)} --> {ts(b)}\n{text}\n\n')

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('main', 'all'):
        build('main', MAIN, [('showtime-demo-60s-1080p.mp4', True, False), ('showtime-demo-60s-1080p-silent.mp4', False, True)])
        srt(MAIN, os.path.join(OUT, 'captions.srt'))
    if what in ('reel', 'all'):
        build('reel', REEL, [('showtime-reel-9x16.mp4', True, False)])
    if what in ('clips', 'all'):
        for k, e in CLIPS.items(): build(k, e, [(k + '.mp4', True, False)])
