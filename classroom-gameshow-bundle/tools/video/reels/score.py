#!/usr/bin/env python3
"""
A short original score for each reel, composed in code (MIDI, rendered with FluidSynth and the FluidR3_GM soundfont,
MIT licence) and timed to that reel's own cuts, so the music has the reel's mood instead of one loop under everything:

  reel 1  Will it fall?     suspense: a low drone, a heartbeat and a snare roll, then a big stab on the fall, then a bright groove
  reel 2  POV game show     bouncy game-show groove at 112 bpm, a stab on every cut, a roll into the call to action
  reel 3  oddly satisfying  calm: soft pad and celesta, no drums, warm resolve on the call to action
  reel 4  The Hunter        tense A minor chase, a roll as the class reaches HOME, then it turns major and triumphant
  reel 5  £5,000 jackpot    long build to the jackpot fall, a fanfare on it, then celebration

Every reel ends with the same short fanfare tag when the call to action starts (the last 4 s), held to the last frame.
Entirely original: no existing tune is quoted.

  python3 tools/video/reels/score.py [reel ...]   ->  marketing/reels/work/music/reel-<n>.wav  (mix.py --music-wav)
"""
import os, subprocess, sys
import mido

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(ROOT, 'marketing', 'reels', 'work', 'music')
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
TPB = 480
KICK, SNARE, CLAP, HAT, OHAT, CRASH, TOM_L, TOM_M, TOM_H = 36, 38, 39, 42, 46, 49, 45, 47, 50

# reel: (bpm, length s, call to action starts s, cue s)  -- bpm is chosen so the cue (the fall, the escape) is on a beat
REELS = {
    1: (120.0, 12.5, 8.5, 5.5),
    2: (112.0, 16.9, 12.9, None),
    3: (84.0, 15.0, 11.0, None),
    4: (134.16, 15.85, 11.85, 8.05),
    5: (144.0, 14.233, 10.233, 6.667),
}

class Score:
    def __init__(self, bpm):
        self.bpm = bpm
        self.m = mido.MidiFile(ticks_per_beat=TPB)
        tempo = mido.MidiTrack(); self.m.tracks.append(tempo)
        tempo.append(mido.MetaMessage('set_tempo', tempo=int(60e6 / bpm), time=0))
        self.tr = {}
    def b(self, sec): return sec * self.bpm / 60.0
    def track(self, name, ch, prog):
        t = mido.MidiTrack(); self.m.tracks.append(t)
        t.append(mido.MetaMessage('track_name', name=name, time=0))
        if ch != 9: t.append(mido.Message('program_change', channel=ch, program=prog, time=0))
        self.tr[name] = {'t': t, 'ch': ch, 'ev': []}
        return name
    def note(self, name, beat, dur, pitch, vel=96):
        if beat < 0: return
        e = self.tr[name]['ev']
        e.append((round(beat * TPB), 1, pitch, max(1, min(127, int(vel))))); e.append((round((beat + dur) * TPB), 0, pitch, 0))
    def save(self, path):
        for tr in self.tr.values():
            last = 0
            for tick, on, pitch, vel in sorted(tr['ev'], key=lambda e: (e[0], e[1])):
                tr['t'].append(mido.Message('note_on' if on else 'note_off', channel=tr['ch'], note=pitch, velocity=vel, time=tick - last)); last = tick
        self.m.save(path)

def instruments(s, extra=()):
    names = {'brass': (0, 61), 'bass': (1, 34), 'keys': (2, 4), 'lead': (3, 56), 'drums': (9, 0)}
    names.update(dict(extra))
    for n, (ch, prog) in names.items(): s.track(n, ch, prog)

# C G Am F (bass root, triad); A minor version for the chase: Am F C G / Am F Dm E
MAJOR = [(36, (60, 64, 67)), (43, (59, 62, 67)), (45, (57, 60, 64)), (41, (57, 60, 65))]
MINOR = [(33, (57, 60, 64)), (29, (57, 60, 65)), (36, (55, 60, 64)), (31, (55, 59, 62))]

def groove(s, b0, b1, chords=MAJOR, drums='full', brass_hits=True, hat16=False):
    """bass, offbeat keys, drums and brass hits from beat b0 to b1 (a bar cut short stops at b1)"""
    def note(tr, beat, dur, pitch, vel=96):
        if b0 <= beat < b1: s.note(tr, beat, min(dur, b1 - beat), pitch, vel)
    for bar in range(int(-(-(b1 - b0) // 4))):
        b = b0 + bar * 4; root, tri = chords[bar % 4]
        for i, p in enumerate([0, 0, 12, 0, 0, 7, 12, 7]): note('bass', b + i * 0.5, 0.42, root + p, 92 if i % 2 == 0 else 80)
        for off in (0.5, 1.5, 2.5, 3.5):
            for p in tri: note('keys', b + off, 0.22, p + 12 if off in (1.5, 3.5) else p, 70)
        if drums != 'none':
            n = 16 if hat16 else 8; st = 4.0 / n
            for i in range(n): note('drums', b + i * st, 0.2, HAT if i % 2 == 0 or drums == 'full' else OHAT, 70 if i % 2 == 0 else 50)
            for off in ((0, 2, 2.5) if drums == 'full' else (0, 2)): note('drums', b + off, 0.3, KICK, 110)
            if drums == 'full':
                for off in (1, 3): note('drums', b + off, 0.3, SNARE, 105); note('drums', b + off, 0.3, CLAP, 70)
        if brass_hits:
            for off, d in ((0, 0.5), (2.5, 0.4)):
                for p in tri: note('brass', b + off, d, p + 12, 100)

def roll(s, b0, b1, v0, v1, step=0.25):
    n = int((b1 - b0) / step)
    for i in range(n): s.note('drums', b0 + i * step, 0.2, SNARE, v0 + (v1 - v0) * i / max(1, n - 1))

def rise(s, b0, n, lo=60, vel0=70, step=0.25, track='lead', scale=(0, 2, 4, 5, 7, 9, 11)):
    """a rising scale run of n notes starting at beat b0"""
    for i in range(n): s.note(track, b0 + i * step, step, lo + 12 * (i // 7) + scale[i % 7], vel0 + i * 3)

def stab(s, beat, pitches=(48, 55, 60, 64, 67, 72), dur=1.0, vel=122, crash=True, tom=False):
    for p in pitches: s.note('brass', beat, dur, p, vel)
    s.note('bass', beat, dur, pitches[0] - 12 if pitches[0] > 40 else pitches[0], 118)
    s.note('drums', beat, 0.3, KICK, 120); s.note('drums', beat, 0.3, SNARE, 112)
    if crash: s.note('drums', beat, 1.5, CRASH, 120)

def cta_tag(s, t_c, length, soft=False):
    """the same short fanfare at the start of the call to action, then a held final chord to the last frame"""
    b0, end = s.b(t_c), s.b(length)
    if soft:
        for p in (48, 55, 60, 64, 67, 71): s.note('pad', b0, end - b0, p, 52)
        for i, p in enumerate((84, 83, 79, 76, 79, 83, 88)): s.note('bells', b0 + 0.5 + i * 0.5, 1.5, p, 62)
        s.note('bells', b0 + 4.5, end - b0 - 4.5, 91, 56)
        return
    stab(s, b0, (48, 55, 60, 64, 67, 72, 76), dur=1.0, vel=124, tom=True)
    # the hook: G G A G C E G (over the headline and the button)
    for beat, dur, p in ((0.5, .5, 67), (1.0, .5, 67), (1.5, .5, 69), (2.0, .5, 67), (2.5, 1, 72), (3.5, 1, 76)):
        s.note('lead', b0 + beat, dur * 0.9, p, 108); s.note('brass', b0 + beat, dur * 0.9, p - 3 if p > 70 else p - 4, 84)
    # drums keep time under the hook, then the last chord holds with a crash
    for i in range(int(4.5 * 2)): s.note('drums', b0 + 1 + i * 0.5, 0.2, HAT, 64)
    for off in (1, 2, 3): s.note('drums', b0 + off, 0.3, KICK, 100)
    for off in (2, 4): s.note('drums', b0 + off, 0.3, SNARE, 100)
    last = b0 + 4.5
    stab(s, last, (48, 55, 60, 64, 67, 72, 76), dur=max(1.0, end - last), vel=124)
    s.note('lead', last, max(1.0, end - last), 84, 116)
    for i in range(6): s.note('drums', last + i * 0.5, 0.3, TOM_H if i % 2 else TOM_M, 96)

def reel1(s, bpm, length, t_c, cue):
    instruments(s, {'strings': (4, 48)})
    fall = s.b(cue)
    # 0 to the fall: a low drone that swells, a heartbeat that quickens, a snare roll, a rising run
    for p in (36, 43, 48, 51): s.note('strings', 0, fall, p, 50)
    for p in (60, 63, 67, 70): s.note('strings', 0, fall, p, 40)
    for i in range(int(fall)): s.note('drums', i, 0.4, KICK, 60 + 50 * i / fall); s.note('drums', i + 0.5, 0.2, KICK, 40 + 40 * i / fall)
    roll(s, fall - 4, fall, 30, 115)
    rise(s, fall - 2, 8, 60, 70, 0.25, scale=(0, 2, 3, 5, 7, 8, 10))
    stab(s, fall, (48, 55, 60, 64, 67, 72), 1.5)
    groove(s, fall + 1.5, s.b(t_c) - 0.0, drums='full')
    cta_tag(s, t_c, length)

def reel2(s, bpm, length, t_c, cue):
    instruments(s)
    cut = 4                                   # a cut every 4 beats
    stab(s, 0, (48, 55, 60, 64, 67, 72), 1.0)
    groove(s, 1, s.b(t_c) - 2, hat16=False)
    for beat in range(cut, int(s.b(t_c)) - 2, cut):
        for p in (60, 64, 67, 72): s.note('brass', beat, 0.9, p, 112)
        s.note('drums', beat, 1.2, CRASH, 90)
    roll(s, s.b(t_c) - 2, s.b(t_c), 40, 120)
    rise(s, s.b(t_c) - 2, 8, 67, 70, 0.25)
    cta_tag(s, t_c, length)

def reel3(s, bpm, length, t_c, cue):
    instruments(s, {'pad': (4, 89), 'bells': (5, 8), 'harp': (6, 46)})
    # calm: a soft pad on C, G, Am, F, a slow harp arpeggio, and a bell on the first beat of each bar
    chords = [(48, (60, 64, 67, 71)), (43, (59, 62, 67, 71)), (45, (57, 60, 64, 67)), (41, (57, 60, 65, 69))]
    end = s.b(t_c)
    for bar in range(int(end // 4) + 1):
        b = bar * 4; root, tri = chords[bar % 4]
        if b >= end: break
        for p in tri[:3]: s.note('pad', b, min(4.1, end - b), p, 44)
        s.note('pad', b, min(4.1, end - b), root - 12, 40)
        for i in range(8):
            if b + i * 0.5 < end: s.note('harp', b + i * 0.5, 0.9, tri[[0, 1, 2, 3, 2, 1, 2, 1][i]] + 12, 52 + (i % 2) * 6)
        if b + 0.0 < end: s.note('bells', b, 1.5, tri[0] + 24, 50)
    cta_tag(s, t_c, length, soft=True)

def reel4(s, bpm, length, t_c, cue):
    instruments(s, {'strings': (4, 44)})
    esc = s.b(cue)
    # the chase: A minor, a kick on every beat, tremolo strings stabbing, the Hunter's low brass; the first two answers are wrong
    groove(s, 0, esc - 2, chords=MINOR, hat16=True, brass_hits=False)
    for beat in range(0, int(esc - 2), 2):
        root, tri = MINOR[(beat // 4) % 4]
        for p in tri: s.note('strings', beat + 1.5, 0.45, p, 78)
        s.note('brass', beat, 0.9, root + 12, 96); s.note('brass', beat, 0.9, root + 19, 90)
        s.note('drums', beat, 0.3, KICK, 112); s.note('drums', beat + 1, 0.3, KICK, 100)
    roll(s, esc - 2, esc, 50, 124, 0.25)
    rise(s, esc - 2, 8, 57, 70, 0.25, scale=(0, 2, 3, 5, 7, 8, 11))
    # HOME: it turns major and triumphant
    stab(s, esc, (48, 55, 60, 64, 67, 72, 76), 2.0)
    groove(s, esc + 2, s.b(t_c), chords=MAJOR, hat16=False)
    cta_tag(s, t_c, length)

def reel5(s, bpm, length, t_c, cue):
    instruments(s, {'strings': (4, 48)})
    fall = s.b(cue)
    # a long build: drone, quickening heartbeat, ticking hats, a roll, a high run, then the jackpot fanfare
    for p in (36, 43, 48, 52): s.note('strings', 0, fall, p, 46)
    for p in (64, 67, 71, 74): s.note('strings', 0, fall, p, 38)
    for i in range(int(fall * 2)): s.note('drums', i * 0.5, 0.15, HAT, 40 + 36 * i / (fall * 2))
    for i in range(int(fall)): s.note('drums', i, 0.4, KICK, 56 + 54 * i / fall)
    roll(s, fall - 6, fall, 28, 118, 0.25)
    rise(s, fall - 3, 12, 60, 64, 0.25)
    stab(s, fall, (48, 55, 60, 64, 67, 72, 76), 2.0, vel=127)
    for i in range(8): s.note('drums', fall + i * 0.5, 0.3, TOM_H if i % 2 else TOM_M, 104)
    for i, p in enumerate((72, 74, 76, 77, 79, 81, 83, 84)): s.note('lead', fall + 2 + i * 0.25, 0.25, p, 100 + i * 2)
    groove(s, fall + 4, s.b(t_c), hat16=False)
    cta_tag(s, t_c, length)

BUILD = {1: reel1, 2: reel2, 3: reel3, 4: reel4, 5: reel5}

def render(n):
    bpm, length, t_c, cue = REELS[n]
    s = Score(bpm); BUILD[n](s, bpm, length, t_c, cue)
    os.makedirs(OUT, exist_ok=True)
    mid = os.path.join(OUT, f'reel-{n}.mid'); s.save(mid)
    tmp, wav = os.path.join(OUT, f'reel-{n}.tmp.wav'), os.path.join(OUT, f'reel-{n}.wav')
    subprocess.run(['fluidsynth', '-ni', '-g', '0.8', '-r', '48000', '-F', tmp, SF2, mid], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-t', f'{length:.3f}', '-af', f'afade=t=out:st={length - 0.5:.3f}:d=0.5', '-ac', '1', wav], check=True)
    os.remove(tmp); print('wrote', wav)

if __name__ == '__main__':
    for n in ([int(a) for a in sys.argv[1:]] or REELS): render(n)
