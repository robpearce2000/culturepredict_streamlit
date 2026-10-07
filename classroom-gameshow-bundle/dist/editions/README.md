# Showtime editions (version 1.5.0)

Built from the same source by `node build.js --editions` (tools/editions.js). Every edition has all four games, the question bank and "write your own"; they differ only in the built-in packs. Checksums: SHA256SUMS.txt. Upload files: ../downloads/. Listing text and images: ../listings/<edition>/.

| Edition | File | Size | Subjects | Topic packs | Questions | Load time |
|---|---|---|---|---|---|---|
| Free taster | `showtime-free-taster.html` | 1.25 MB (zip 1.67 MB) | Biology, Chemistry, Physics, Maths, History, Geography | 13 | 806 | 213 ms |
| Biology edition | `showtime-biology.html` | 1.28 MB (zip 1.69 MB) | Biology | 16 | 1,007 | 237 ms |
| Chemistry edition | `showtime-chemistry.html` | 1.30 MB (zip 1.71 MB) | Chemistry | 19 | 1,437 | 217 ms |
| Physics edition | `showtime-physics.html` | 1.31 MB (zip 1.72 MB) | Physics | 23 | 1,555 | 229 ms |
| Maths edition | `showtime-maths.html` | 1.24 MB (zip 1.67 MB) | Maths | 12 | 809 | 239 ms |
| History edition | `showtime-history.html` | 1.33 MB (zip 1.73 MB) | History | 33 | 1,675 | 217 ms |
| Geography edition | `showtime-geography.html` | 1.31 MB (zip 1.72 MB) | Geography | 32 | 1,389 | 239 ms |
| Science mega pack | `showtime-science.html` | 1.52 MB (zip 1.87 MB) | Biology, Chemistry, Physics | 58 | 3,999 | 252 ms |
| Mega bundle | `showtime-mega-bundle.html` | 1.85 MB (zip 2.11 MB) | Biology, Chemistry, Physics, Maths, History, Geography | 135 | 7,872 | 241 ms |

Topic packs and questions count each separate-science or other specification topic once; the Combined Science views of the science packs use the same questions. Load time: from opening the file to the launcher showing, in headless Chromium on the build machine (median of three); every edition is well under the 3-second target.
