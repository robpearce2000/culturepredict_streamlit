#!/usr/bin/env python3
"""
Original game-show style score for the Showtime demo video (about 60 s, 120 bpm, C major), composed in code.
It is entirely original: no existing tune, theme or sample is quoted or imitated. Written as MIDI with
mido, then rendered with FluidSynth and the FluidR3_GM soundfont (MIT licence; see LICENSES.md).

  python3 tools/video/music.py     ->  marketing/video/showtime-theme.mid  and  showtime-theme.wav

Beats: 1 beat = 0.5 s, so beat 6 is 3 s, beat 30 is 15 s, beat 110 is 55 s. The cues follow the video:
  0-3 s    snare roll and a rising brass swell, a big stab on the cascade (beat 5)
  3-6 s    the title hook
  6-44 s   the groove, with a short sting on each game's name (beats 30, 44, 60, 74)
  44-50 s  breakdown under the launcher montage; 50-55 s the build
  55-60 s  final fanfare, ending cleanly at 60 s
"""
import os, subprocess, sys
import mido

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
OUT = os.path.join(ROOT, 'marketing', 'video')
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
TPB = 480
mid = mido.MidiFile(ticks_per_beat=TPB)
def track(ch, prog, name):
    t = mido.MidiTrack(); mid.tracks.append(t); t.append(mido.MetaMessage('track_name', name=name, time=0))
    if ch != 9: t.append(mido.Message('program_change', channel=ch, program=prog, time=0))
    return {'t': t, 'ch': ch, 'ev': []}
tempo = mid.tracks.append(mido.MidiTrack()) or mid.tracks[0]
tempo.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
BRASS, BASS, KEYS, LEAD, DR = track(0, 61, 'Brass'), track(1, 34, 'Bass'), track(2, 4, 'Keys'), track(3, 56, 'Trumpet'), track(9, 0, 'Drums')
def note(tr, beat, dur, pitch, vel=96):
    tr['ev'].append((round(beat * TPB), 1, pitch, vel)); tr['ev'].append((round((beat + dur) * TPB), 0, pitch, 0))
def flush(tr):
    last = 0
    for tick, on, pitch, vel in sorted(tr['ev'], key=lambda e: (e[0], e[1])):
        tr['t'].append(mido.Message('note_on' if on else 'note_off', channel=tr['ch'], note=pitch, velocity=vel, time=tick - last)); last = tick
KICK, SNARE, CLAP, HAT, OHAT, CRASH, TOM_L, TOM_M, TOM_H = 36, 38, 39, 42, 46, 49, 45, 47, 50
CHORDS = [(36, (60, 64, 67)), (43, (59, 62, 67)), (45, (57, 60, 64)), (41, (57, 60, 65))]   # C G Am F: bass root, keys triad
def chord_at(bar): return CHORDS[bar % 4]

def groove(b0, b1, bass=True, keys=True, drums='full', brass_hits=True):
    for bar in range((b1 - b0) // 4):
        b = b0 + bar * 4; root, tri = chord_at((b - 6) // 4 if b >= 6 else bar)
        if bass:
            for i, p in enumerate([0, 0, 12, 0, 0, 7, 12, 7]): note(BASS, b + i * 0.5, 0.42, root + p, 92 if i % 2 == 0 else 80)
        if keys:
            for off in (0.5, 1.5, 2.5, 3.5):
                for p in tri: note(KEYS, b + off, 0.22, p + 12 if off in (1.5, 3.5) else p, 70)
        if drums != 'none':
            for i in range(8): note(DR, b + i * 0.5, 0.2, HAT if i % 2 == 0 or drums == 'full' else OHAT, 70 if i % 2 == 0 else 52)
            for off in ((0, 2, 2.5) if drums == 'full' else (0, 2)): note(DR, b + off, 0.3, KICK, 110)
            if drums == 'full':
                for off in (1, 3): note(DR, b + off, 0.3, SNARE, 105); note(DR, b + off, 0.3, CLAP, 70)
        if brass_hits:
            for off, d in ((0, 0.5), (2.5, 0.4)):
                for p in tri: note(BRASS, b + off, d, p + 12, 100)
def roll(b0, b1, v0, v1, step=0.25):
    n = int((b1 - b0) / step)
    for i in range(n): note(DR, b0 + i * step, 0.2, SNARE, int(v0 + (v1 - v0) * i / max(1, n - 1)))
def sting(beat, high=False):
    for p in ((67, 72) if not high else (72, 76)): pass
    note(LEAD, beat, 0.5, 67 if not high else 72, 105); note(LEAD, beat + 0.5, 1.5, 72 if not high else 76, 108)
    for p in (60, 64, 67, 72): note(BRASS, beat + 0.5, 1.2, p, 96)
    note(DR, beat, 0.5, CRASH, 100)

# --- 0-3 s: snare roll, swell, the stab on the cascade (beat 5)
roll(0, 5, 30, 112)
for p in (48, 55, 60, 64): note(BRASS, 0, 5, p, 60)
for i, p in enumerate((60, 62, 64, 65, 67, 69, 71, 72, 74, 76)): note(LEAD, 2.5 + i * 0.25, 0.25, p, 70 + i * 4)
note(DR, 4, 0.4, KICK, 112); note(DR, 4.5, 0.4, KICK, 112)
for p in (48, 55, 60, 64, 67, 72): note(BRASS, 5, 1, p, 120)
note(BASS, 5, 1, 36, 120); note(DR, 5, 1, CRASH, 120); note(DR, 5, 0.4, KICK, 120); note(DR, 5, 0.4, SNARE, 118)
# --- 3-6 s: the title hook (beats 6-12), over a light groove
hook = [(6, .5, 67), (6.5, .5, 67), (7, .5, 69), (7.5, .5, 67), (8, 1, 72), (9, 1, 76), (10, .5, 74), (10.5, .5, 74), (11, .5, 76), (11.5, .5, 74), (12, 1.5, 79), (13.5, .5, 76)]
for b, d, p in hook:
    note(LEAD, b, d * 0.9, p, 108)
    note(BRASS, b, d * 0.9, p - 4 if p in (72, 76, 79) else p - 3, 84)
groove(6, 14, brass_hits=False)
# --- 14-44 s: the groove, a sting on each game's name (beats 30 Over the Edge, 44 Outpace, 60 Category Clash, 74 Hex Hunt)
groove(14, 30)
sting(30); groove(34, 44)
sting(44, high=True); groove(48, 60)
sting(60); groove(64, 74)
sting(74, high=True); groove(78, 88)
# the bars between a sting and the next groove: keep time with drums and bass
for b0, b1 in ((30, 34), (44, 48), (60, 64), (74, 78)):
    groove(b0, b1, keys=False, brass_hits=False)
# --- 44-50 s: breakdown under the launcher montage (beats 88-100): bass, keys, light kick and hats
groove(88, 100, drums='light', brass_hits=False)
# --- 50-55 s: the build back in (beats 100-110), a roll into the fanfare
groove(100, 106)
roll(106, 110, 50, 120, 0.25)
for i, p in enumerate((55, 57, 59, 60, 62, 64, 65, 67)): note(LEAD, 106 + i * 0.5, 0.5, p + 12, 80 + i * 5)
# --- 55-60 s: final fanfare (beats 110-120), then silence on the 60 s mark
def hit(beat, dur, pitches, vel=122):
    for p in pitches: note(BRASS, beat, dur, p, vel)
hit(110, 1, (48, 55, 60, 64, 67, 72)); note(DR, 110, 1, CRASH, 125); note(DR, 110, .3, KICK, 125); note(DR, 110, .3, SNARE, 120); note(BASS, 110, 1, 36, 120)
hit(111.5, .5, (53, 57, 60, 65, 69, 72)); hit(112, 1, (55, 59, 62, 67, 71, 74)); note(BASS, 112, 1, 43, 118)
for i, p in enumerate((72, 74, 76, 77, 79, 81, 83, 84)): note(LEAD, 113 + i * 0.25, 0.25, p, 100 + i * 2)
hit(115, 3.8, (48, 55, 60, 64, 67, 72, 76), 125); note(LEAD, 115, 3.8, 84, 118); note(BASS, 115, 3.8, 36, 120)
note(DR, 115, 3.8, CRASH, 125); note(DR, 115, .3, KICK, 125)
for i in range(8): note(DR, 115 + i * 0.5, 0.3, TOM_H if i % 2 else TOM_M, 100)
for tr in (BRASS, BASS, KEYS, LEAD, DR): flush(tr)
os.makedirs(OUT, exist_ok=True)
mid_path = os.path.join(OUT, 'showtime-theme.mid'); mid.save(mid_path)
wav = os.path.join(OUT, 'showtime-theme.wav')
subprocess.run(['fluidsynth', '-ni', '-g', '0.8', '-r', '44100', '-F', wav, SF2, mid_path], check=True, stdout=subprocess.DEVNULL)
# trim to exactly 60 s with a short fade at the very end, so the music ends cleanly with the video
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-t', '60', '-af', 'afade=t=out:st=59.4:d=0.6', os.path.join(OUT, 'showtime-theme-60s.wav')], check=True)
os.replace(os.path.join(OUT, 'showtime-theme-60s.wav'), wav)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-b:a', '192k', os.path.join(OUT, 'showtime-theme.mp3')], check=True)
print('wrote', mid_path, wav)
