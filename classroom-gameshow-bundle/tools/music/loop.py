#!/usr/bin/env python3
"""
Original launcher music for Showtime: a seamless game-show loop (24 bars, about 50 s) and a short fanfare,
composed in code, written as MIDI with mido and rendered with FluidSynth + FluidR3_GM (MIT licence; see
LICENSES.md). Entirely original: no existing tune is quoted or imitated.

  python3 tools/music/loop.py            ->  music-drafts/ : draft-{a,b,c}.mp3, draft-{a,b,c}-x2.mp3 (loops twice),
                                             fanfare-{a,b,c}.mp3 and the MIDI sources in tools/music/midi/

The loop is rendered three times from MIDI and the middle pass is cut out at exact bar lines, so the reverb and
any held notes wrap round and the join cannot be heard. Layout: 8 bars hook / 8 bars answer / 8 bars hook, harmonised.
"""
import os, subprocess, sys, shutil
import mido

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
TPB = 480
KICK, SNARE, CLAP, HAT, OHAT, CRASH, RIDE, SHAKER, COW = 36, 38, 39, 42, 46, 49, 51, 70, 56

# name: tempo, melody voice, harmony voice, bass, keys, drum feel
DRAFTS = {
    'a': dict(bpm=118, lead=56, harm=61, bass=34, keys=4,  feel='bold',   brass_vel=100, lead_vel=104, swing=0.0),    # brassier and bolder
    'b': dict(bpm=112, lead=73, harm=56, bass=33, keys=4,  feel='bouncy', brass_vel=80,  lead_vel=92,  swing=0.12),   # lighter and bouncier (flute-led, trumpet colour)
    'c': dict(bpm=115, lead=56, harm=60, bass=34, keys=0,  feel='mid',    brass_vel=90,  lead_vel=98,  swing=0.06),   # in between (trumpet and horn, piano)
}
# 8-bar chord loop: C Am F G | C Em F G   (root, triad)
PROG = [(36, (60, 64, 67)), (33, (57, 60, 64)), (41, (60, 65, 69)), (43, (59, 62, 67)),
        (36, (60, 64, 67)), (40, (59, 64, 67)), (41, (60, 65, 69)), (43, (59, 62, 67))]
# the hook, as (beat in bar, length, MIDI pitch), one list per bar
HOOK = [
    [(0, .5, 76), (.5, .5, 76), (1, .5, 79), (2, 1, 76), (3, .5, 74), (3.5, .5, 72)],
    [(0, .5, 72), (.5, .5, 76), (1, 1.5, 81), (3, 1, 79)],
    [(0, .5, 77), (.5, .5, 77), (1, .5, 81), (2, 1, 77), (3, .5, 76), (3.5, .5, 74)],
    [(0, .5, 74), (.5, .5, 79), (1, .5, 83), (2, 2, 79)],
    [(0, .5, 76), (.5, .5, 76), (1, .5, 79), (2, .5, 84), (2.5, .5, 83), (3, 1, 79)],
    [(0, .5, 76), (.5, .5, 79), (1, 1, 83), (3, .5, 81), (3.5, .5, 79)],
    [(0, 1, 81), (1, .5, 79), (1.5, .5, 77), (2, 1, 79), (3, 1, 81)],
    [(0, .5, 79), (.5, .5, 74), (1, .5, 71), (1.5, .5, 74), (2, 1.5, 79), (3.5, .5, 76)],
]

def build(name, cfg, loops=1):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    meta = mido.MidiTrack(); mid.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(cfg['bpm']), time=0))
    tr = {}
    for ch, key, prog in ((0, 'lead', cfg['lead']), (1, 'harm', cfg['harm']), (2, 'bass', cfg['bass']), (3, 'keys', cfg['keys']), (9, 'drums', 0)):
        t = mido.MidiTrack(); mid.tracks.append(t); t.append(mido.MetaMessage('track_name', name=key, time=0))
        if ch != 9: t.append(mido.Message('program_change', channel=ch, program=prog, time=0))
        tr[key] = dict(t=t, ch=ch, ev=[])
    def note(k, beat, dur, pitch, vel):
        sw = cfg['swing'] if (beat * 2) % 2 == 1 else 0     # push the off-beat eighths late for bounce
        b = beat + sw
        tr[k]['ev'].append((round(b * TPB), 1, pitch, int(vel))); tr[k]['ev'].append((round((b + dur * 0.92) * TPB), 0, pitch, 0))
    feel = cfg['feel']
    for lp in range(loops):
        for bar in range(24):
            b0 = (lp * 24 + bar) * 4
            sec, pb = bar // 8, bar % 8
            root, tri = PROG[pb]
            # bass: bouncy octave pattern
            pat = [0, 12, 0, 7, 0, 12, 7, 5] if feel != 'mid' else [0, 0, 12, 0, 0, 7, 12, 7]
            for i, p in enumerate(pat):
                note('bass', b0 + i * .5, .5 if feel == 'bouncy' else .45, root + p + (12 if root < 36 else 0), 96 if i % 2 == 0 else 80)
            # keys: off-beat chord stabs (up an octave on the "and" of 2 and 4)
            for off in (.5, 1.5, 2.5, 3.5):
                for p in tri: note('keys', b0 + off, .25, p + (12 if off in (1.5, 3.5) else 0), 66)
            # drums
            for i in range(8):
                note('drums', b0 + i * .5, .2, HAT, 74 if i % 2 == 0 else 54)
            kicks = (0, 2) if feel == 'bouncy' else (0, 1.5, 2) if feel == 'mid' else (0, 1, 2, 3)
            for off in kicks: note('drums', b0 + off, .3, KICK, 108)
            for off in (1, 3): note('drums', b0 + off, .3, SNARE, 100); note('drums', b0 + off, .3, CLAP, 62)
            if feel == 'bouncy':
                for i in range(8): note('drums', b0 + i * .5 + .25, .1, SHAKER, 38)
            if pb == 7 and sec < 2:   # fill into the next section, and on the last bar into the loop point
                for i in range(4): note('drums', b0 + 3 + i * .25, .2, SNARE, 70 + i * 10)
            if pb == 0 and sec == 0 and lp == 0: note('drums', b0, 1, CRASH, 90)
            # melody
            if sec in (0, 2):
                for (o, d, p) in HOOK[pb]:
                    note('lead', b0 + o, d, p, cfg['lead_vel'])
                    if sec == 2: note('harm', b0 + o, d, p - 4 if p % 12 in (0, 5, 7) else p - 3, cfg['brass_vel'] - 12)   # harmonised a third below
                # sustained pad-like brass chord under the hook
                for p in tri: note('harm', b0, 3.6, p, cfg['brass_vel'] - 30)
            else:
                # answer section: syncopated brass stabs on the chord, melody climbs on the chord notes
                for off, d in ((0, .75), (.75, .75), (1.5, .5), (2.5, .5), (3, .5)):
                    for p in tri: note('harm', b0 + off, d, p + 12, cfg['brass_vel'])
                for i, off in enumerate((0, .75, 1.5, 2.5, 3)):
                    note('lead', b0 + off, .5, tri[i % 3] + 24 if pb % 2 == 0 else tri[(i + 1) % 3] + 24, cfg['lead_vel'] - 6)
    for k in tr:
        last = 0
        for tick, on, pitch, vel in sorted(tr[k]['ev'], key=lambda e: (e[0], e[1])):
            tr[k]['t'].append(mido.Message('note_on' if on else 'note_off', channel=tr[k]['ch'], note=pitch, velocity=vel, time=tick - last)); last = tick
    return mid

def fanfare(cfg):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    meta = mido.MidiTrack(); mid.tracks.append(meta); meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(cfg['bpm'] * 1.6), time=0))
    t = mido.MidiTrack(); mid.tracks.append(t)
    t.append(mido.Message('program_change', channel=0, program=cfg['harm'], time=0))
    t.append(mido.Message('program_change', channel=1, program=cfg['lead'], time=0))
    ev = []
    def n(ch, b, d, p, v): ev.append((round(b * TPB), 1, ch, p, v)); ev.append((round((b + d) * TPB), 0, ch, p, 0))
    # up-beat run, then a big C major chord: G4 C5 E5 | G5 held with C major underneath
    for i, p in enumerate((67, 72, 76)): n(1, i * .5, .5, p, 100)
    for p in (60, 64, 67, 72): n(0, 1.5, 1.6, p, 110)
    n(1, 1.5, 1.6, 79, 112)
    ev.sort(key=lambda e: (e[0], e[1])); last = 0
    for tick, on, ch, p, v in ev:
        t.append(mido.Message('note_on' if on else 'note_off', channel=ch, note=p, velocity=v, time=tick - last)); last = tick
    return mid

def render(mid, wav):
    path = wav + '.mid'; mid.save(path)
    subprocess.run(['fluidsynth', '-ni', '-g', '0.9', '-r', '44100', '-F', wav, SF2, path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(path)

def main():
    out = os.path.join(ROOT, 'music-drafts'); os.makedirs(out, exist_ok=True)
    midi = os.path.join(HERE, 'midi'); os.makedirs(midi, exist_ok=True)
    tmp = os.path.join(HERE, '_work'); os.makedirs(tmp, exist_ok=True)
    for name, cfg in DRAFTS.items():
        bar = 240 / cfg['bpm']; L = 24 * bar
        build(name, cfg, 1).save(os.path.join(midi, f'loop-{name}.mid'))
        fanfare(cfg).save(os.path.join(midi, f'fanfare-{name}.mid'))
        raw = os.path.join(tmp, f'{name}.wav'); render(build(name, cfg, 3), raw)
        one = os.path.join(tmp, f'{name}-loop.wav')
        # cut the middle pass at exact bar lines: its start already carries the tail of the pass before
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', f'{L:.5f}', '-t', f'{L:.5f}', '-i', raw, '-af', 'volume=0.62,alimiter=limit=0.9', one], check=True)
        two = os.path.join(tmp, f'{name}-x2.wav')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-stream_loop', '1', '-i', one, '-c', 'copy', two], check=True)
        for src, dst in ((one, f'draft-{name}.mp3'), (two, f'draft-{name}-x2.mp3')):
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-codec:a', 'libmp3lame', '-b:a', '160k', os.path.join(out, dst)], check=True)
        fw = os.path.join(tmp, f'fanfare-{name}.wav'); render(fanfare(cfg), fw)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', fw, '-t', '2.2', '-af', 'volume=0.62,afade=t=out:st=1.7:d=0.5,alimiter=limit=0.9', '-codec:a', 'libmp3lame', '-b:a', '160k', os.path.join(out, f'fanfare-{name}.mp3')], check=True)
        print(name, f'loop {L:.2f}s at {cfg["bpm"]} bpm')
    shutil.rmtree(tmp)

if __name__ == '__main__':
    main()
