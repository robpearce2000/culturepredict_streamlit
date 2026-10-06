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
| AQA | Physics | 8463 | 1.1 | 30 September 2019 | Fetched from aqa.org.uk (with Appendix A, the equations students recall or select from the sheet) |
| AQA | Mathematics | 8300 | 1.1 | September 2026 | Fetched from aqa.org.uk. Higher-only content is the spec's HIGHER ONLY column |
| AQA | History | 8145 | | | Not started |
| AQA | Geography | 8035 | | | Not started |
| Pearson Edexcel | Biology, Chemistry, Physics | 1BI0, 1CH0, 1PH0 | Issue 4 | March 2024 | Pearson's own published PDFs, as captured by the Internet Archive in September 2026 (qualifications.pearson.com cannot be reached from the build environment). To confirm against Pearson's site that Issue 4 is still current |
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
| AQA Physics 4.1 Energy | 30 | 63 | 63 | 0 | 4 | 0 | 0 | Done |
| AQA Physics 4.2 Electricity | 36 | 51 | 51 | 0 | 6 | 0 | 0 | Done |
| AQA Physics 4.3 Particle model of matter | 30 | 41 | 42 | 0 | 9 | 1 | 0 | Done |
| AQA Physics 4.4 Atomic structure | 36 | 47 | 49 | 1 | 4 | 3 | 0 | Done |
| AQA Physics 4.5 Forces | 75 | 98 | 102 | 0 | 6 | 4 | 0 | Done |
| AQA Physics 4.6 Waves | 39 | 55 | 58 | 0 | 10 | 3 | 0 | Done |
| AQA Physics 4.7 Magnetism and electromagnetism | 30 | 48 | 48 | 0 | 3 | 0 | 0 | Done |
| AQA Physics 4.8 Space physics | 30 | 42 | 44 | 5 | 6 | 7 | 0 | Done |
| Edexcel Biology 1 Key concepts in biology | 51 | 62 | 64 | 0 | 15 | 2 | 0 | Done |
| Edexcel Biology 2 Cells and control | 51 | 62 | 62 | 0 | 2 | 0 | 0 | Done |
| Edexcel Biology 3 Genetics | 69 | 83 | 83 | 0 | 8 | 0 | 0 | Done |
| Edexcel Biology 4 Natural selection and genetic modification | 42 | 51 | 52 | 0 | 6 | 1 | 0 | Done |
| Edexcel Biology 5 Health, disease and the development of medicines | 75 | 88 | 90 | 0 | 7 | 2 | 0 | Done |
| Edexcel Biology 6 Plant structures and their functions | 48 | 55 | 55 | 1 | 6 | 1 | 0 | Done |
| Edexcel Biology 7 Animal coordination, control and homeostasis | 66 | 76 | 76 | 1 | 7 | 1 | 0 | Done |
| Edexcel Biology 8 Exchange and transport in animals | 36 | 47 | 47 | 0 | 10 | 0 | 0 | Done |
| Edexcel Biology 9 Ecosystems and material cycles | 57 | 68 | 68 | 0 | 5 | 0 | 0 | Done |
| AQA Maths 3.1 Number | 48 | 59 | 62 | 0 | 5 | 3 | 0 | Done |
| AQA Maths 3.2 Algebra | 75 | 134 | 91 | 43 | 2 | 0 | 0 | Done |
| AQA Maths 3.3 Ratio, proportion and rates of change | 48 | 56 | 56 | 0 | 4 | 0 | 0 | Done |
| AQA Maths 3.4 Geometry and measures | 75 | 93 | 94 | 0 | 6 | 1 | 0 | Done |
| AQA Maths 3.5 Probability | 30 | 46 | 46 | 0 | 2 | 0 | 1 | Done |
| AQA Maths 3.6 Statistics | 30 | 52 | 53 | 0 | 2 | 1 | 0 | Done |
| Edexcel Chemistry 1 Key concepts in chemistry | 159 | 184 | 184 | 0 | 26 | 0 | 0 | Done |
| Edexcel Chemistry 2 States of matter and mixtures | 36 | 43 | 43 | 2 | 6 | 2 | 0 | Done |
| Edexcel Chemistry 3 Chemical changes | 93 | 108 | 108 | 0 | 3 | 0 | 0 | Done |
| Edexcel Chemistry 4 Extracting metals and equilibria | 51 | 59 | 59 | 0 | 3 | 0 | 0 | Done |
| Edexcel Chemistry 5 Separate chemistry 1 | 81 | 96 | 97 | 2 | 6 | 3 | 0 | Done |
| Edexcel Chemistry 6 Groups in the periodic table | 48 | 56 | 56 | 1 | 6 | 1 | 0 | Done |
| Edexcel Chemistry 7 Rates of reaction and energy changes | 48 | 61 | 62 | 0 | 7 | 1 | 0 | Done |
| Edexcel Chemistry 8 Fuels and Earth science | 78 | 97 | 97 | 3 | 9 | 3 | 0 | Done |
| Edexcel Chemistry 9 Separate chemistry 2 | 117 | 139 | 139 | 0 | 13 | 0 | 0 | Done |

Topic 4.8 "Key ideas" in the AQA science specifications is a list of cross-cutting ideas examined
through the other topics, not a topic of its own, so it has no pack (see DECISIONS.md).

Totals so far: AQA sciences, 25 packs, 1,425 questions; Edexcel Biology, 9 packs, 597 questions;
AQA Maths, 6 packs, 402 questions; Edexcel Chemistry, 9 packs, 845 questions.

## Next

Edexcel Physics and Edexcel Maths (being written and checked), then Geography (AQA, Edexcel A,
Edexcel B) and History (AQA, Edexcel), in the brief's order.
