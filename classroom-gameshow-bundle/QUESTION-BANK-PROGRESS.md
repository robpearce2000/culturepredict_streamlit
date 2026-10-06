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
| AQA | History | 8145 | 1.3 | 24 September 2019 | Fetched from aqa.org.uk. The historic environment site (it changes each year) is left out |
| AQA | Geography | 8035 | 1.1 | 12 September 2023 | Fetched from aqa.org.uk. Fieldwork, the pre-release issue evaluation and geographical skills are left out |
| Pearson Edexcel | Biology, Chemistry, Physics | 1BI0, 1CH0, 1PH0 | Issue 4 | March 2024 | Pearson's own published PDFs, as captured by the Internet Archive in September 2026 (qualifications.pearson.com cannot be reached from the build environment). To confirm against Pearson's site that Issue 4 is still current |
| Pearson Edexcel | Mathematics | 1MA1 | Issue 2 | June 2015 | Internet Archive copy of Pearson's PDF (the latest it holds); to confirm on Pearson's site |
| Pearson Edexcel | History | 1HI0 | Issue 6 | June 2024 | Internet Archive copy of Pearson's PDF; the fixed historic environments are included with their thematic studies |
| Pearson Edexcel | Geography A, Geography B | 1GA0, 1GB0 | Issue 5; Issue 4 | September 2025; September 2025 | Internet Archive copies of Pearson's PDFs; fieldwork and skills sections are left out |

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
| Edexcel Maths 1 Number | 48 | 79 | 58 | 30 | 28 | 9 | 1 | Done |
| Edexcel Maths 2 Algebra | 75 | 146 | 92 | 54 | 15 | 0 | 2 | Done |
| Edexcel Maths 3 Ratio, proportion and rates of change | 48 | 59 | 62 | 2 | 11 | 5 | 0 | Done |
| Edexcel Maths 4 Geometry and measures | 75 | 95 | 95 | 0 | 3 | 0 | 0 | Done |
| Edexcel Maths 5 Probability | 30 | 42 | 42 | 0 | 7 | 0 | 0 | Done |
| Edexcel Maths 6 Statistics | 30 | 50 | 55 | 1 | 5 | 6 | 0 | Done |
| AQA Geography 3.1.1 The challenge of natural hazards | 33 | 52 | 56 | 0 | 5 | 4 | 1 | Done |
| AQA Geography 3.1.2 The living world: Ecosystems and tropical rainforests | 30 | 44 | 44 | 0 | 4 | 0 | 0 | Done |
| AQA Geography 3.1.2.3 The living world: Hot deserts (option) | 30 | 39 | 39 | 1 | 1 | 1 | 0 | Done |
| AQA Geography 3.1.2.4 The living world: Cold environments (option) | 30 | 37 | 37 | 3 | 13 | 3 | 0 | Done |
| AQA Geography 3.1.3.2 Physical landscapes in the UK: Coastal landscapes (option) | 30 | 46 | 49 | 0 | 5 | 3 | 0 | Done |
| AQA Geography 3.1.3.3 Physical landscapes in the UK: River landscapes (option) | 30 | 49 | 52 | 3 | 7 | 6 | 1 | Done |
| AQA Geography 3.1.3.4 Physical landscapes in the UK: Glacial landscapes (option) | 30 | 44 | 44 | 0 | 11 | 0 | 0 | Done |
| AQA Geography 3.2.1 Urban issues and challenges | 30 | 47 | 45 | 2 | 5 | 0 | 0 | Done |
| AQA Geography 3.2.2 The changing economic world | 30 | 48 | 49 | 1 | 10 | 2 | 0 | Done |
| AQA Geography 3.2.3.1 The challenge of resource management: Resource management | 30 | 35 | 35 | 1 | 7 | 1 | 0 | Done |
| AQA Geography 3.2.3.2 The challenge of resource management: Food (option) | 30 | 35 | 36 | 0 | 6 | 1 | 0 | Done |
| AQA Geography 3.2.3.3 The challenge of resource management: Water (option) | 30 | 35 | 35 | 0 | 3 | 0 | 1 | Done |
| AQA Geography 3.2.3.4 The challenge of resource management: Energy (option) | 30 | 34 | 37 | 0 | 2 | 3 | 0 | Done |
| AQA History 1AA America, 1840–1895: Expansion and consolidation | 30 | 40 | 40 | 0 | 0 | 0 | 0 | Done |
| AQA History 1AB Germany, 1890–1945: Democracy and dictatorship | 30 | 51 | 51 | 0 | 4 | 0 | 0 | Done |
| AQA History 1AC Russia, 1894–1945: Tsardom and communism | 30 | 42 | 42 | 0 | 2 | 0 | 0 | Done |
| AQA History 1AD America, 1920–1973: Opportunity and inequality | 30 | 42 | 42 | 0 | 0 | 0 | 0 | Done |
| AQA History 1BA Conflict and tension: the First World War, 1894–1918 | 30 | 44 | 48 | 0 | 1 | 4 | 0 | Done |
| AQA History 1BB Conflict and tension: the inter-war years, 1918–1939 | 30 | 55 | 55 | 0 | 1 | 0 | 0 | Done |
| AQA History 1BC Conflict and tension between East and West, 1945–1972 | 30 | 53 | 53 | 0 | 4 | 0 | 0 | Done |
| AQA History 1BD Conflict and tension in Asia, 1950–1975 | 30 | 40 | 40 | 0 | 1 | 0 | 0 | Done |
| AQA History 1BE Conflict and tension in the Gulf and Afghanistan, 1990–2009 | 30 | 37 | 37 | 0 | 4 | 0 | 1 | Done |
| AQA History 2AA Britain: Health and the people: c1000 to the present day | 36 | 49 | 49 | 0 | 2 | 0 | 0 | Done |
| AQA History 2AB Britain: Power and the people: c1170 to the present day | 36 | 52 | 52 | 0 | 2 | 0 | 0 | Done |
| AQA History 2AC Britain: Migration, empires and the people: c790 to the present day | 36 | 55 | 55 | 0 | 2 | 0 | 0 | Done |
| AQA History 2BA Norman England, c1066–c1100 | 30 | 37 | 37 | 0 | 6 | 0 | 0 | Done |
| AQA History 2BB Medieval England - the reign of Edward I, 1272–1307 | 30 | 45 | 45 | 0 | 3 | 0 | 0 | Done |
| AQA History 2BC Elizabethan England, c1568–1603 | 30 | 44 | 44 | 0 | 2 | 0 | 0 | Done |
| AQA History 2BD Restoration England, 1660–1685 | 30 | 42 | 42 | 0 | 2 | 0 | 0 | Done |
| Edexcel Geography A 1A The changing landscapes of the UK: Coastal landscapes and processes (option) | 30 | 37 | 39 | 0 | 5 | 2 | 0 | Done |
| Edexcel Geography A 1B The changing landscapes of the UK: River landscapes and processes (option) | 30 | 47 | 54 | 1 | 8 | 8 | 1 | Done |
| Edexcel Geography A 1C The changing landscapes of the UK: Glaciated upland landscapes and processes (option) | 30 | 45 | 44 | 2 | 9 | 1 | 1 | Done |
| Edexcel Geography A 2 Weather hazards and climate change | 30 | 50 | 48 | 6 | 7 | 4 | 0 | Done |
| Edexcel Geography A 3 Ecosystems, biodiversity and management | 30 | 38 | 38 | 0 | 7 | 0 | 0 | Done |
| Edexcel Geography A 4 Changing cities | 30 | 45 | 48 | 0 | 2 | 3 | 0 | Done |
| Edexcel Geography A 5 Global development | 30 | 48 | 48 | 1 | 7 | 2 | 0 | Done |
| Edexcel Geography A 6 Resource management: Overview of the global and UK distribution of food, energy and water | 30 | 40 | 40 | 1 | 2 | 1 | 0 | Done |
| Edexcel Geography A 6A Resource management: Energy resource management (option) | 30 | 39 | 39 | 0 | 6 | 0 | 0 | Done |
| Edexcel Geography A 6B Resource management: Water resource management (option) | 30 | 38 | 39 | 1 | 9 | 2 | 0 | Done |
| Edexcel Geography A 8 Geographical investigations − UK challenges | 30 | 46 | 50 | 0 | 9 | 4 | 0 | Done |
| Edexcel Geography B 1 Hazardous Earth | 30 | 38 | 38 | 0 | 3 | 0 | 0 | Done |
| Edexcel Geography B 2 Development dynamics | 30 | 48 | 51 | 0 | 7 | 3 | 0 | Done |
| Edexcel Geography B 3 Challenges of an urbanising world | 30 | 39 | 39 | 0 | 3 | 0 | 0 | Done |
| Edexcel Geography B 4 The UK’s evolving physical landscape | 30 | 47 | 47 | 1 | 6 | 1 | 0 | Done |
| Edexcel Geography B 5 The UK’s evolving human landscape | 30 | 43 | 50 | 2 | 9 | 9 | 0 | Done |
| Edexcel Geography B 7 People and the biosphere | 30 | 35 | 35 | 0 | 10 | 0 | 0 | Done |
| Edexcel Geography B 8 Forests under threat | 30 | 42 | 44 | 0 | 8 | 2 | 0 | Done |
| Edexcel Geography B 9 Consuming energy resources | 30 | 39 | 40 | 0 | 6 | 1 | 0 | Done |
| Edexcel History 10 Crime and punishment in Britain, c1000–present and Whitechapel, c1870–c1900: crime, policing and the inner city | 42 | 61 | 61 | 0 | 0 | 0 | 0 | Done |
| Edexcel History 11 Medicine in Britain, c1250–present and The British sector of the Western Front, 1914–18: injuries, treatment and the trenches | 42 | 55 | 55 | 0 | 4 | 0 | 0 | Done |
| Edexcel History 12 Warfare and British society, c1250–present and London and the Second World War, 1939–45 | 42 | 74 | 74 | 0 | 0 | 0 | 0 | Done |
| Edexcel History 13 Migrants in Britain, c800–present and Notting Hill, c1948–c1970 | 42 | 52 | 55 | 0 | 4 | 3 | 0 | Done |
| Edexcel History B1 Anglo-Saxon and Norman England, c1060–88 | 36 | 50 | 49 | 1 | 3 | 0 | 0 | Done |
| Edexcel History B2 The reigns of King Richard I and King John, 1189–1216 | 36 | 53 | 55 | 0 | 2 | 2 | 0 | Done |
| Edexcel History B3 Henry VIII and his ministers, 1509–40 | 36 | 52 | 52 | 0 | 1 | 0 | 0 | Done |
| Edexcel History B4 Early Elizabethan England, 1558–88 | 36 | 46 | 46 | 0 | 4 | 0 | 0 | Done |
| Edexcel History P1 Spain and the ‘New World’, c1490–c1555 | 30 | 45 | 45 | 1 | 5 | 1 | 0 | Done |
| Edexcel History P2 British America, 1713–83: empire and revolution | 30 | 36 | 39 | 0 | 3 | 3 | 0 | Done |
| Edexcel History P3 The American West, c1835–c1895 | 30 | 47 | 46 | 1 | 1 | 0 | 0 | Done |
| Edexcel History P4 Superpower relations and the Cold War, 1941–91 | 30 | 43 | 43 | 0 | 0 | 0 | 0 | Done |
| Edexcel History P5 Conflict in the Middle East, 1945–95 | 30 | 36 | 40 | 0 | 2 | 4 | 0 | Done |
| Edexcel History 30 Russia and the Soviet Union, 1917–41 | 48 | 62 | 62 | 0 | 1 | 0 | 0 | Done |
| Edexcel History 31 Weimar and Nazi Germany, 1918–39 | 48 | 60 | 63 | 0 | 3 | 3 | 0 | Done |
| Edexcel History 32 Mao’s China, 1945–76 | 48 | 64 | 68 | 0 | 8 | 4 | 0 | Done |
| Edexcel History 33 The USA, 1954–75: conflict at home and abroad | 48 | 85 | 90 | 0 | 17 | 5 | 0 | Done |
| Edexcel Physics 1 Key concepts of physics | 30 | 46 | 38 | 8 | 8 | 0 | 0 | Done |
| Edexcel Physics 2 Motion and forces | 99 | 115 | 115 | 0 | 6 | 0 | 0 | Done |
| Edexcel Physics 3 Conservation of energy | 42 | 52 | 48 | 5 | 3 | 1 | 0 | Done |
| Edexcel Physics 4 Waves | 51 | 61 | 61 | 1 | 3 | 1 | 0 | Done |
| Edexcel Physics 5 Light and the electromagnetic spectrum | 72 | 90 | 83 | 7 | 7 | 0 | 0 | Done |
| Edexcel Physics 6 Radioactivity | 138 | 161 | 157 | 4 | 4 | 0 | 0 | Done |
| Edexcel Physics 7 Astronomy | 57 | 67 | 67 | 0 | 6 | 0 | 0 | Done |
| Edexcel Physics 8 Energy – forces doing work | 45 | 53 | 53 | 2 | 9 | 2 | 0 | Done |
| Edexcel Physics 9 Forces and their effects | 30 | 40 | 40 | 0 | 3 | 0 | 0 | Done |
| Edexcel Physics 10 Electricity and circuits | 126 | 161 | 143 | 21 | 6 | 3 | 0 | Done |
| Edexcel Physics 11 Static electricity | 30 | 36 | 34 | 2 | 2 | 0 | 0 | Done |
| Edexcel Physics 12 Magnetism and the motor effect | 42 | 48 | 48 | 0 | 1 | 0 | 0 | Done |
| Edexcel Physics 13 Electromagnetic induction | 33 | 42 | 38 | 4 | 3 | 0 | 0 | Done |
| Edexcel Physics 14 Particle model | 60 | 69 | 69 | 0 | 4 | 0 | 0 | Done |
| Edexcel Physics 15 Forces and matter | 51 | 59 | 58 | 1 | 1 | 0 | 0 | Done |

Topic 4.8 "Key ideas" in the AQA science specifications is a list of cross-cutting ideas examined
through the other topics, not a topic of its own, so it has no pack (see DECISIONS.md).

Totals: 135 packs, 7,792 questions, covering every specification in the brief.

## Next

All packs in the brief are written and checked. Next: Rob's review of QUESTIONS-TO-CHECK.md, and
confirming the Pearson issue numbers on qualifications.pearson.com.
