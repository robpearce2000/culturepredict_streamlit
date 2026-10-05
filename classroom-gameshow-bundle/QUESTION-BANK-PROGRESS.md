# Question bank progress

The job (QUESTION-BANKS-BRIEF): exam-board-accurate GCSE question packs, one per specification
topic, for AQA and Pearson Edexcel in Biology, Chemistry, Physics, Maths, History and Geography.
This file says what is finished and what is left, so any run can carry on where the last stopped.

## How a pack is made

1. **Specification.** The board's current published PDF, fetched from the board's website. Its
   numbered points, Higher-tier-only headings and required practicals are saved as a registry in
   `packs/specs/<board>-<code>.json` (the PDFs themselves are not kept in the repository).
2. **Write.** One writer per topic, working only from that topic's specification text, following
   the rules in the brief (correct with one answer, on the specification, exam-useful, a
   20-second mini-whiteboard answer, easy to mark, original). File: `packs/<board>/<subject>/<ref>-<topic>.txt`.
3. **Check.** A separate checker reads every question against the specification text, then does a
   second, error-hunting read (wrong facts, units, arithmetic re-done independently, ambiguity,
   more than one right answer, copied wording, wrong Higher-tier tags, wrong difficulty) and fixes
   or removes what it finds. Anything it cannot settle goes to `QUESTIONS-TO-CHECK.md`.
4. **Automated checks** (`node tools/packs.js check`, and `tests/packs.spec.js` in the test suite):
   every field present and valid; every Ref a real point of the specification; Higher-tier-only
   points tagged; no duplicates or near-duplicates; answer length; Hex Hunt letters; at least 30
   questions per topic (3 per specification point for larger topics); each difficulty at least a
   fifth; at least a third of answers usable in Hex Hunt; every point and required practical covered.
5. **Ids** are added with `node tools/packs.js ids <folder>` once a pack is checked, and never change.

## Specifications used

| Board | Subject | Code | Version | Date | Status |
|---|---|---|---|---|---|
| AQA | Biology | 8461 | 1.0 | 21 April 2016 | Fetched from aqa.org.uk (confirmed as the current version on the specification page) |
| AQA | Chemistry | 8462 | 1.1 | 4 October 2019 | Fetched from aqa.org.uk |
| AQA | Physics | 8463 | 1.1 | 30 September 2019 | Fetched from aqa.org.uk |
| AQA | Mathematics | 8300 | | | Not started |
| AQA | History | 8145 | | | Not started |
| AQA | Geography | 8035 | | | Not started |
| Pearson Edexcel | Biology, Chemistry, Physics | 1BI0, 1CH0, 1PH0 | | | Not started. Note: qualifications.pearson.com dropped connections from this environment on the first try; a direct PDF link returned an HTML page. To retry. |
| Pearson Edexcel | Mathematics | 1MA1 | | | Not started |
| Pearson Edexcel | History | 1HI0 | | | Not started |
| Pearson Edexcel | Geography A, Geography B | 1GA0, 1GB0 | | | Not started |

## Packs

Written = questions the writer produced; Checked = questions after the checking pass;
Removed / Rewritten / Added = changes made in checking; Flagged = items in QUESTIONS-TO-CHECK.md.

| Pack | Target | Written | Checked | Removed | Rewritten | Added | Flagged | Status |
|---|---|---|---|---|---|---|---|---|
| AQA Biology 4.1 Cell biology | 36 | 47 | 47 | 0 | 8 | 0 | 0 | Done |
| AQA Biology 4.2 Organisation | 30 | 42 | 42 | 0 | 8 | 0 | 0 | Done |
| AQA Biology 4.3 Infection and response | 39 | 49 | 50 | 0 | 3 | 1 | 0 | Done |
| AQA Biology 4.4 Bioenergetics | 30 | 39 | 39 | 0 | 9 | 0 | 0 | Done |
| AQA Biology 4.5 Homeostasis and response | 42 | 60 | 68 | 0 | 22 | 8 | 0 | Done |
| AQA Biology 4.6 Inheritance, variation and evolution | 63 | 82 | 84 | 0 | 11 | 2 | 0 | Done |
| AQA Biology 4.7 Ecology | 63 | 80 | 80 | 0 | 14 | 0 | 0 | Done |
| AQA Chemistry 4.1 Atomic structure and the periodic table | 45 | 55 | 56 | 0 | 12 | 1 | 0 | Done |
| AQA Chemistry 4.2 Bonding, structure, and the properties of matter | 54 | 88 | 89 | 0 | 3 | 1 | 0 | Done |
| AQA Chemistry 4.3 Quantitative chemistry | 39 | 52 | 52 | 0 | 4 | 0 | 0 | Done |
| AQA Chemistry 4.4 Chemical changes | 45 | 55 | 55 | 0 | 15 | 0 | 0 | Done |
| AQA Chemistry 4.5 Energy changes | 30 | 46 | 46 | 0 | 6 | 0 | 0 | Done |
| AQA Chemistry 4.6 The rate and extent of chemical change | 33 | 46 | 46 | 0 | 5 | 0 | 0 | Done |
| AQA Chemistry 4.7 Organic chemistry | 36 | 60 | 60 | 0 | 2 | 0 | 0 | Done |
| AQA Chemistry 4.8 Chemical analysis | 42 | 61 | 61 | 0 | 4 | 0 | 0 | Done |
| AQA Chemistry 4.9 Chemistry of the atmosphere | 30 | 44 | 46 | 0 | 9 | 2 | 0 | Done |
| AQA Chemistry 4.10 Using resources | 33 | 46 | 47 | 0 | 2 | 1 | 0 | Done |

Topic 4.8 "Key ideas" in the AQA science specifications is a list of cross-cutting ideas examined
through the other topics, not a topic of its own, so it has no pack (see DECISIONS.md).

## Next

AQA Physics (written; checking in progress), then Pearson Edexcel sciences, AQA and
Edexcel Maths, Geography, History (the brief's order).
