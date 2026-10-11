# Showtime editions (version 1.7.0)

Built from the same source by `node build.js --editions` (tools/editions.js). Every edition has all four games, the question bank and "write your own"; they differ only in the built-in packs. Checksums: SHA256SUMS.txt. Upload files: ../downloads/. Listing text and images: ../listings/<edition>/.

| Edition | File | Size | Subjects | Topic packs | Questions | Load time |
|---|---|---|---|---|---|---|
| Free taster | `showtime-free-taster.html` | 2.09 MB (zip 2.29 MB) | Biology, Chemistry, Physics, Maths, History, Geography | 13 | 843 | 250 ms |
| Biology edition | `showtime-biology.html` | 2.11 MB (zip 2.30 MB) | Biology | 16 | 1,011 | 232 ms |
| Chemistry edition | `showtime-chemistry.html` | 2.13 MB (zip 2.32 MB) | Chemistry | 19 | 1,437 | 238 ms |
| Physics edition | `showtime-physics.html` | 2.15 MB (zip 2.33 MB) | Physics | 23 | 1,602 | 256 ms |
| Maths edition | `showtime-maths.html` | 2.08 MB (zip 2.28 MB) | Maths | 12 | 868 | 258 ms |
| History edition | `showtime-history.html` | 2.17 MB (zip 2.35 MB) | History | 33 | 1,698 | 247 ms |
| Geography edition | `showtime-geography.html` | 2.14 MB (zip 2.33 MB) | Geography | 32 | 1,393 | 281 ms |
| Science mega pack | `showtime-science.html` | 2.35 MB (zip 2.48 MB) | Biology, Chemistry, Physics | 58 | 4,050 | 325 ms |
| Mega bundle | `showtime-mega-bundle.html` | 2.69 MB (zip 2.73 MB) | Biology, Chemistry, Physics, Maths, History, Geography | 135 | 8,009 | 324 ms |

Topic packs and questions count each separate-science or other specification topic once; the Combined Science views of the science packs use the same questions. Load time: from opening the file to the launcher showing, in headless Chromium on the build machine (median of three); every edition is well under the 3-second target.
