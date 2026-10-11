#!/usr/bin/env python3
"""
Rebuilds a reel's game sounds from what the page recorded (every effect is an oscillator with frequency and gain
automation), mixes in the Showtime theme (the launcher loop, draft B, from tools/music) and encodes the mp4.

  python3 tools/video/reels/mix.py <name> <out.mp4> [--music] [--loop] [--speed 0.5] [--start s] [--end s]

  --music   theme music under the game sounds (otherwise game sounds only)
  --speed   slow-motion factor the sounds were recorded at (pitch kept, timing stretched); 1 = real time
"""
import argparse, json, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
WORK = os.path.join(ROOT, 'marketing', 'reels', 'work')
SR = 48000

def automation(events, t, default):
    """value of a Web Audio parameter at times t (set / exponential / linear ramps)"""
    v = np.full_like(t, default, dtype=np.float64)
    if not events: return v
    ev = sorted(events, key=lambda e: e[1])
    cur_t, cur_v = None, default
    for kind, et, ev_v in ev:
        if kind == 'set' or cur_t is None:
            v[t >= et] = ev_v; cur_t, cur_v = et, ev_v; continue
        seg = (t >= cur_t) & (t <= et)
        if et > cur_t:
            f = (t[seg] - cur_t) / (et - cur_t)
            if kind == 'exp' and cur_v > 0 and ev_v > 0: v[seg] = cur_v * (ev_v / cur_v) ** f
            else: v[seg] = cur_v + (ev_v - cur_v) * f
        v[t > et] = ev_v; cur_t, cur_v = et, ev_v
    return v

def wave(kind, phase):
    p = phase % 1.0
    if kind == 'square': return np.where(p < 0.5, 1.0, -1.0)
    if kind == 'sawtooth': return 2 * p - 1
    if kind == 'triangle': return 4 * np.abs(p - 0.5) - 1
    return np.sin(2 * math.pi * p)

def render_sfx(rec, length, speed=1.0):
    out = np.zeros(int(length * SR) + SR)
    t0 = rec['t0']
    for s in rec['sounds']:
        start, stop = s.get('start') or 0, s.get('stop') or 0
        if not stop or stop <= start: continue
        a, b = (start - t0) / speed, (stop - t0) / speed
        if b < 0 or a > length: continue
        i0, i1 = max(0, int(a * SR)), min(len(out), int(b * SR))
        if i1 <= i0: continue
        tt = t0 + (np.arange(i0, i1) / SR) * speed            # page time of each sample (slow motion keeps pitch)
        f = automation(s.get('freq', []), tt, 440.0)
        g = automation((s.get('gain') or {}).get('gain', []), tt, 1.0)
        phase = np.cumsum(f / SR)
        out[i0:i1] += wave(s.get('type', 'sine'), phase) * g
    return out

def theme(length):
    """the Showtime theme (launcher loop, draft B), looped to length"""
    path = os.path.join(WORK, 'theme-b.wav')
    if not os.path.exists(path):
        sys.path.insert(0, os.path.join(ROOT, 'tools', 'music'))
        import loop
        cfg = loop.DRAFTS['b']; L = 24 * 240 / cfg['bpm']
        raw = path + '.raw.wav'; loop.render(loop.build('b', cfg, 3), raw)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{L:.5f}', '-t', f'{L:.5f}', '-i', raw, '-ar', str(SR), '-ac', '1', path], check=True)
        os.remove(raw)
    data = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True, check=True).stdout
    m = np.frombuffer(data, dtype=np.float32).astype(np.float64)
    n = int(length * SR); reps = n // len(m) + 1
    return np.tile(m, reps)[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('out')
    ap.add_argument('--music', action='store_true'); ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--music-offset', type=float, default=0.0); ap.add_argument('--sfx-gain', type=float, default=1.6)
    ap.add_argument('--overlay', default=None, help='PNG laid over every frame (a caption layer)')
    ap.add_argument('--frames', default=None, help='ffmpeg input pattern (default work/<name>/%%05d.jpg)')
    a = ap.parse_args()
    d = os.path.join(WORK, a.name)
    rec = json.load(open(os.path.join(d, 'sounds.json')))
    fps = rec['fps']; length = rec['frames'] / fps
    sfx = render_sfx(rec, length, a.speed)[:int(length * SR)] * a.sfx_gain
    mix = sfx.copy()
    if a.music:
        m = theme(length + a.music_offset)[int(a.music_offset * SR):][:len(mix)]
        fade = int(0.15 * SR); m[:fade] *= np.linspace(0, 1, fade) if rec.get('fadeIn') else 1; m[-fade:] *= np.linspace(1, 0, fade) if rec.get('fadeOut', True) else 1
        mix = mix + 0.55 * m
    peak = max(1e-6, np.abs(mix).max()); mix = mix * min(1.0, 0.89 / peak)
    wav = os.path.join(d, 'mix.wav')
    pcm = (np.clip(mix, -1, 1) * 32767).astype('<i2')
    import wave as wv
    with wv.open(wav, 'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    frames = a.frames or os.path.join(d, '%05d.jpg')
    vin = ['-framerate', str(fps), '-i', frames] + (['-i', a.overlay] if a.overlay else [])
    vf = ['-filter_complex', '[0:v][1:v]overlay=0:0[v]', '-map', '[v]', '-map', '2:a'] if a.overlay else []
    subprocess.run(['ffmpeg', '-y', '-v', 'error'] + vin + ['-i', wav] + vf + ['-c:v', 'libx264', '-preset', 'slow', '-crf', '19',
                    '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-r', str(fps), '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-shortest', '-movflags', '+faststart', a.out], check=True)
    print('wrote', a.out, f'{length:.2f}s')

if __name__ == '__main__':
    main()
