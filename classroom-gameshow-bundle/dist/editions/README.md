# Showtime editions (version 1.8.3)

Built from the same source by `node build.js --editions` (tools/editions.js). Every edition has all four games, the question bank and "write your own"; they differ only in the built-in packs. Checksums: SHA256SUMS.txt. Upload files: ../downloads/. Listing text and images: ../listings/<edition>/.

| Edition | File | Size | Subjects | Topic packs | Questions | Load time |
|---|---|---|---|---|---|---|
| Free taster | `showtime-free-taster.html` | 2.09 MB (zip 1.80 MB) | Biology, Chemistry, Physics, Maths, History, Geography | 13 | 861 | 404 ms |
| Biology edition | `showtime-biology.html` | 2.12 MB (zip 1.83 MB) | Biology | 16 | 1,006 | 446 ms |
| Chemistry edition | `showtime-chemistry.html` | 2.14 MB (zip 1.84 MB) | Chemistry | 19 | 1,424 | 493 ms |
| Physics edition | `showtime-physics.html` | 2.15 MB (zip 1.85 MB) | Physics | 23 | 1,580 | 412 ms |
| Maths edition | `showtime-maths.html` | 2.09 MB (zip 1.81 MB) | Maths | 12 | 868 | 422 ms |
| History edition | `showtime-history.html` | 2.18 MB (zip 1.87 MB) | History | 33 | 1,698 | 349 ms |
| Geography edition | `showtime-geography.html` | 2.15 MB (zip 1.85 MB) | Geography | 32 | 1,393 | 355 ms |
| Science mega pack | `showtime-science.html` | 2.36 MB (zip 2.00 MB) | Biology, Chemistry, Physics | 58 | 4,010 | 419 ms |
| Mega bundle | `showtime-mega-bundle.html` | 2.70 MB (zip 2.25 MB) | Biology, Chemistry, Physics, Maths, History, Geography | 135 | 7,969 | 411 ms |

Topic packs and questions count each separate-science or other specification topic once; the Combined Science views of the science packs use the same questions. Load time: from opening the file to the launcher showing, in headless Chromium on the build machine (median of three); every edition is well under the 3-second target.
