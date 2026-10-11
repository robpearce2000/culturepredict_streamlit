#!/usr/bin/env python3
"""
Independent check of numerical answers for one pack's review file.

  python3 tools/review/calc_check.py review/science/<key>.json

The review file holds, for the pack it names:
  "calcs":   { id: "python expression giving the correct numerical answer", ... }
             or { id: ["expression", tolerance] } (relative tolerance; default 0.01)
  "notcalc": { id: "why the number in this answer is not worked out (a year, a recalled value, a count...)" }
Every question is found as a candidate by its content (the answer contains a number, or the question
gives two or more numbers), whatever its Calc tag. Each candidate must appear in calcs or notcalc.
The stored answer's number is read (standard form, fractions, ratios a : b as a/b, unicode minus,
thousands spaces) and compared with the expression. Mismatches and uncovered candidates are printed.
"""
import json, math, re, subprocess, sys
from fractions import Fraction

SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')
def norm(s):
    s = s.replace('−', '-').replace('–', '-').replace('×', 'x').replace(' ', '').replace(' ', '')
    s = re.sub(r'(?<=\d)[ ,](?=\d{3}\b)', '', s)
    return s
def numbers(text):
    return re.findall(r'-?\d+(?:\.\d+)?', norm(text))
def read_answer(a):
    t = norm(a)
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*x\s*10\s*(?:\^|\*\*)?\s*([⁻⁰¹²³⁴⁵⁶⁷⁸⁹\-]+|\(?-?\d+\)?)', t)
    if m:
        e = m.group(2).translate(SUP).strip('()')
        return float(m.group(1)) * 10 ** int(e)
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)', t)
    if m: return float(m.group(1)) / float(m.group(2))
    m = re.search(r'(-?\d+)\s*/\s*(\d+)', t)
    if m and not re.search(r'[a-zA-Z]\s*/', t[:m.start() + 1]): return float(Fraction(int(m.group(1)), int(m.group(2))))
    t2 = t.replace('½', '.5') if re.search(r'\d½', t) else t.replace('½', '0.5')
    m = re.search(r'-?\d+(?:\.\d+)?', t2)
    return float(m.group(0)) if m else None
def candidate(q):
    return bool(re.search(r'\d|½|¼', q['a'])) or len(numbers(q['q'])) >= 2

def main(path):
    rv = json.load(open(path))
    data = json.loads(subprocess.check_output(['node', 'tools/review/export.js', rv['pack']]))
    qs = {q['id']: q for q in data['questions']}
    calcs, notcalc = rv.get('calcs', {}), rv.get('notcalc', {})
    bad = 0
    for i in list(calcs) + list(notcalc):
        if i not in qs: print('UNKNOWN ID', i); bad += 1
    env = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
    for i, q in qs.items():
        if candidate(q) and i not in calcs and i not in notcalc:
            print('UNCOVERED', i, '|', q['q'], '|', q['a']); bad += 1
    for i, spec in calcs.items():
        if i not in qs: continue
        expr, tol = (spec if isinstance(spec, list) else [spec, 0.01])
        try: want = float(eval(str(expr), {'__builtins__': {}}, env))
        except Exception as e: print('BAD EXPR', i, expr, e); bad += 1; continue
        got = read_answer(qs[i]['a'])
        ok = got is not None and (abs(got - want) <= tol * max(abs(want), 1e-12) or (want == 0 and abs(got) < 1e-12))
        if not ok:
            print('MISMATCH', i, '| stored:', qs[i]['a'], '| read:', got, '| computed:', want, '|', qs[i]['q']); bad += 1
    print(f"{len(qs)} questions; {sum(candidate(q) for q in qs.values())} numerical candidates; {len(calcs)} recalculated; {len(notcalc)} not calculations; {bad} problems")
    return bad

if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1]) else 0)
